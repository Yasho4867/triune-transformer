import torch
import torch.nn.functional as F

import config as defaults
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
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([{'params': [dummy_param]}], defaults)
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd

        self.non_expert_params = []
        self.layer_groups = []

        group_idx = 0
        num_groups = 0
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2:
                                num_groups += 1

        group_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2:
                                stagger = -(group_idx * (update_gap // max(1, num_groups)))
                                self.layer_groups.append({
                                    'module': module,
                                    'expert_idx': expert_idx,
                                    'param': p,
                                    'projection': None,
                                    'proj_step': stagger,
                                    'state': {'momentum': None, 'variance': None, 'step': 0}
                                })
                                group_idx += 1

        expert_param_ids = {id(g['param']) for g in self.layer_groups}
        for name, param in model.named_parameters():
            if id(param) not in expert_param_ids:
                self.non_expert_params.append(param)

        if HAS_8BIT and AdamW8bit is not None:
            self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.AdamW(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)

    def zero_grad(self):
        self.base_optimizer.zero_grad()
        for group in self.layer_groups:
            p = group['param']
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def set_lr(self, lr):
        for pg in self.base_optimizer.param_groups:
            pg['lr'] = lr

    @torch.no_grad()
    def step(self):
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
            m, n = grad.shape
            grad_fp32 = grad.float()

            if (self.step_count - group['proj_step']) >= self.update_gap or group['projection'] is None:
                U, S, V = torch.svd_lowrank(grad_fp32, q=min(self.rank + 10, m, n), niter=4)
                rank = min(self.rank, m, n)
                P = U[:, :rank] @ torch.diag(S[:rank])
                group['projection'] = P.to(grad.dtype)
                group['proj_step'] = self.step_count
                state['momentum'] = None
                state['variance'] = None
                state['step'] = 0

            P = group['projection']
            zero_col = torch.zeros(m, 1, dtype=grad.dtype, device=grad.device)

            # ─── Centroid steering ────────────────────────────
            centroids = module.last_centroids
            steer_applied = False
            if centroids is not None and expert_idx < centroids.size(0) and self.steer_scale > 0:
                c = centroids[expert_idx]
                if c.size(0) != m:
                    expert = module.experts[expert_idx]
                    first_linear = expert[0]
                    if isinstance(first_linear, FP4Linear):
                        w = first_linear.linear.weight
                        b = first_linear.linear.bias
                    else:
                        w = first_linear.weight
                        b = first_linear.bias
                    c_projected = F.linear(c.unsqueeze(0), w, b).squeeze(0)
                else:
                    c_projected = c

                c_norm = c_projected.norm()
                if c_norm > 1e-8:
                    c_hat = c_projected / c_norm
                    c_proj = P @ (P.T @ c_hat)
                    c_res = c_hat - c_proj
                    c_res_norm = c_res.norm()
                    if c_res_norm > 1e-8:
                        c_orth = c_res / c_res_norm
                        P_aug = torch.cat([P, self.steer_scale * c_orth.unsqueeze(1)], dim=1)
                        steer_applied = True

            if not steer_applied:
                P_aug = torch.cat([P, zero_col], dim=1)

            g_lr = P_aug.T @ grad

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
            delta_full = P_aug @ delta_lr

            if expert_wd != 0:
                p.data -= expert_lr * expert_wd * p.data
            p.data -= expert_lr * delta_full.reshape(p.shape)

    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'layer_groups': [
                {
                    'projection': g['projection'],
                    'proj_step': g['proj_step'],
                    'state': g['state']
                }
                for g in self.layer_groups
            ]
        }

    def load_state_dict(self, state_dict):
        # Integrity checks
        assert len(state_dict['layer_groups']) == len(self.layer_groups), \
            f"Layer group count mismatch: saved {len(state_dict['layer_groups'])}, current {len(self.layer_groups)}"
        for saved, cur in zip(state_dict['layer_groups'], self.layer_groups):
            # Shape check on projection if saved
            if saved['projection'] is not None:
                rows, cols = cur['param'].shape
                expected_shape = (rows, min(self.rank, rows, cols))
                assert tuple(saved['projection'].shape) == expected_shape, \
                    f"Projection shape mismatch: saved {saved['projection'].shape}, expected {expected_shape}"
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        for g, sd in zip(self.layer_groups, state_dict['layer_groups']):
            g['projection'] = sd['projection']
            g['proj_step'] = sd['proj_step']
            g['state'].update(sd['state'])
