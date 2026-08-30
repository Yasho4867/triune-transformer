import torch
import torch.nn.functional as F

import triune.model.config as defaults
from triune.model import MoE_FFN, FP4Linear


try:
    from bitsandbytes.optim import AdamW8bit
    HAS_8BIT = True
except ImportError:
    HAS_8BIT = False
    AdamW8bit = None

class CentroidSteerOptimizer(torch.optim.Optimizer):
    def __init__(self, model, lr, betas, weight_decay,
                 rank=defaults.GALORE_RANK, update_gap=defaults.GALORE_UPDATE_GAP,
                 steer_scale=0.1,
                 expert_lr=defaults.GALORE_LR, expert_betas=defaults.GALORE_BETAS,
                 expert_wd=defaults.GALORE_WEIGHT_DECAY):
        defaults_dict = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([{'params': [dummy_param]}], defaults_dict)
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd

        self.non_expert_params = []
        self.layer_groups = []
        seen_param_ids = set()

        num_groups = 0
        # 1. Count MoE routed experts
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2 and id(p) not in seen_param_ids:
                                seen_param_ids.add(id(p))
                                num_groups += 1

        # 2. Count all remaining 2D parameters (Attention, Shared Experts, Dense FFNs, Exit Heads)
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2 and id(param) not in seen_param_ids:
                seen_param_ids.add(id(param))
                num_groups += 1

        seen_param_ids.clear()
        group_idx = 0

        # 1. Populate MoE routed experts (with Centroid Steering)
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2 and id(p) not in seen_param_ids:
                                seen_param_ids.add(id(p))
                                stagger = -(group_idx * (update_gap // max(1, num_groups)))
                                self.layer_groups.append({
                                    'module': module,
                                    'expert_idx': expert_idx,
                                    'param': p,
                                    'projection': None,
                                    'projection_side': 'left',
                                    'proj_step': stagger,
                                    'state': {'momentum': None, 'variance': None, 'step': 0}
                                })
                                group_idx += 1

        # 2. Populate remaining 2D parameters (Standard GaLore)
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2 and id(param) not in seen_param_ids:
                seen_param_ids.add(id(param))
                stagger = -(group_idx * (update_gap // max(1, num_groups)))
                self.layer_groups.append({
                    'module': None,
                    'expert_idx': None,
                    'param': param,
                    'projection': None,
                    'projection_side': 'left',
                    'proj_step': stagger,
                    'state': {'momentum': None, 'variance': None, 'step': 0}
                })
                group_idx += 1

        # 3. 1D parameters (RMSNorm, biases) -> Base Optimizer
        for name, param in model.named_parameters():
            if id(param) not in seen_param_ids:
                self.non_expert_params.append(param)

        if HAS_8BIT and AdamW8bit is not None:
            self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.AdamW(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
        for group in self.layer_groups:
            p = group['param']
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()

    def set_lr(self, lr):
        for pg in self.base_optimizer.param_groups:
            pg['lr'] = lr
        self.expert_lr = lr

    @torch.no_grad()
    def step(self, closure=None):
        self.step_count += 1
        self.base_optimizer.step()

        expert_lr = self.expert_lr
        expert_beta1, expert_beta2 = self.expert_betas
        expert_wd = self.expert_wd

        for group in self.layer_groups:
            module = group['module']
            expert_idx = group['expert_idx']
            p = group['param']
            state = group['state']

            if p.grad is None:
                continue

            grad = p.grad.data
            if grad.abs().sum() == 0:
                continue
            m, n = grad.shape
            if (self.step_count - group['proj_step']) >= self.update_gap or group['projection'] is None:
                grad_fp32 = grad.float()
                U, S, V = torch.svd_lowrank(grad_fp32, q=min(self.rank + 10, m, n), niter=2)
                rank = min(self.rank, m, n)
                
                # Determine projection side based on matrix shape to minimize memory footprint
                if m >= n:
                    # Left Projection: P (m x rank)
                    group['projection_side'] = 'left'
                    group['projection'] = U[:, :rank].to(grad.dtype).contiguous()
                else:
                    # Right Projection: Q (n x rank)
                    group['projection_side'] = 'right'
                    group['projection'] = V[:, :rank].to(grad.dtype).contiguous()
                    
                del grad_fp32, U, S, V
                # Preserve stagger offset on first initialization
                group['proj_step'] = self.step_count
                state['momentum'] = None
                state['variance'] = None
                state['step'] = 0

            proj = group['projection']
            side = group.get('projection_side', 'left')

            # ─── Centroid steering ────────────────────────────
            centroids = module.last_centroids if module is not None else None
            steer_applied = False

            if centroids is not None and expert_idx is not None and expert_idx < centroids.size(0) and self.steer_scale > 0:
                c = centroids[expert_idx]
                target_dim = m if side == 'left' else n
                if c.size(0) != target_dim:
                    expert = module.experts[expert_idx]
                    first_linear = expert[0]
                    w = first_linear.linear.weight if isinstance(first_linear, FP4Linear) else first_linear.weight
                    b = first_linear.linear.bias if isinstance(first_linear, FP4Linear) else first_linear.bias
                    c_projected = F.linear(c.unsqueeze(0).to(device=w.device, dtype=w.dtype), w, b).squeeze(0)
                else:
                    c_projected = c.to(device=proj.device, dtype=proj.dtype)

                c_norm = c_projected.norm()
                if c_norm > 1e-8:
                    c_hat = (c_projected / c_norm).to(device=proj.device, dtype=proj.dtype)
                    c_proj = proj @ (proj.T @ c_hat)
                    c_res = c_hat - c_proj
                    c_res_norm = c_res.norm()
                    if c_res_norm > 1e-8:
                        c_orth = (c_res / c_res_norm).to(device=proj.device, dtype=proj.dtype)
                        proj_aug = torch.cat([proj, (self.steer_scale * c_orth).unsqueeze(1)], dim=1)
                        steer_applied = True

            if not steer_applied:
                zero_vec = torch.zeros(m if side == 'left' else n, 1, dtype=proj.dtype, device=proj.device)
                proj_aug = torch.cat([proj, zero_vec], dim=1)

            # Gradient projection & Adam step
            if side == 'left':
                g_lr = proj_aug.T @ grad  # Shape: (rank+1) x n
            else:
                g_lr = grad @ proj_aug    # Shape: m x (rank+1)

            state['step'] += 1
            if state['momentum'] is None:
                state['momentum'] = g_lr.clone()
                state['variance'] = g_lr.pow(2).clone()
            else:
                state['momentum'] = expert_beta1 * state['momentum'] + (1 - expert_beta1) * g_lr
                state['variance'] = expert_beta2 * state['variance'] + (1 - expert_beta2) * g_lr.pow(2)

            step = state['step']
            m_hat = state['momentum'] / (1 - expert_beta1 ** step)
            v_hat = state['variance'] / (1 - expert_beta2 ** step)
            delta_lr = m_hat / (v_hat.sqrt() + 1e-8)

            # Back-project updates to parameters
            if side == 'left':
                delta_full = proj_aug @ delta_lr
            else:
                delta_full = delta_lr @ proj_aug.T

            if expert_wd != 0:
                p.data -= expert_lr * expert_wd * p.data
            p.data -= expert_lr * delta_full.reshape(p.shape)

        if torch.cuda.is_available() and (self.step_count <= 2 or self.step_count % 50 == 0):
            torch.cuda.empty_cache()

    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'layer_groups': [
                {
                    'projection': g['projection'],
                    'projection_side': g.get('projection_side', 'left'),
                    'proj_step': g['proj_step'],
                    'state': g['state']
                }
                for g in self.layer_groups
            ]
        }

    def load_state_dict(self, state_dict):
        assert len(state_dict['layer_groups']) == len(self.layer_groups), \
            f"Layer group count mismatch: saved {len(state_dict['layer_groups'])}, current {len(self.layer_groups)}"
        for saved, cur in zip(state_dict['layer_groups'], self.layer_groups):
            if saved['projection'] is not None:
                rows, cols = cur['param'].shape
                side = saved.get('projection_side', 'left')
                if side == 'left':
                    expected_shape = (rows, min(self.rank, rows, cols))
                else:
                    expected_shape = (cols, min(self.rank, rows, cols))
                assert tuple(saved['projection'].shape) == expected_shape, \
                    f"Projection shape mismatch: saved {saved['projection'].shape}, expected {expected_shape}"
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        for g, sd in zip(self.layer_groups, state_dict['layer_groups']):
            target_device = g['param'].device
            g['projection'] = sd['projection'].to(target_device) if sd['projection'] is not None else None
            g['projection_side'] = sd.get('projection_side', 'left')
            g['proj_step'] = sd['proj_step']
            restored_state = {}
            for k, v in sd['state'].items():
                if isinstance(v, torch.Tensor):
                    restored_state[k] = v.to(target_device)
                else:
                    restored_state[k] = v
            g['state'].update(restored_state)
