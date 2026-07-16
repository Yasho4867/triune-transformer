import torch
import torch.nn as nn
import torch.nn.functional as F

from .fp4 import *
from .norms import *

class MoE_FFN(nn.Module):
    def __init__(self, dim, num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS, use_fp4=True, shared_expert=SHARED_EXPERT):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.dim = dim
        self.shared_expert = shared_expert
        self.shared_scale = SHARED_EXPERT_SCALE
        LinearCls = FP4Linear if use_fp4 else nn.Linear

        self.experts = nn.ModuleList([
            nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )
            for _ in range(num_experts)
        ])

        if shared_expert:
            self.shared = nn.Sequential(
                LinearCls(dim, dim * EXPERT_HIDDEN_MULTIPLIER),
                nn.GELU(),
                LinearCls(dim * EXPERT_HIDDEN_MULTIPLIER, dim)
            )

        self.router = nn.Linear(dim, num_experts)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.last_centroids = None
        self.overflow_counter = 0
        self._global_step = 0

    def forward(self, x, update_stats=True):
        """If update_stats=False, skip bias/centroid updates (used during label generation)."""
        B, T, D = x.shape
        flat_x = x.reshape(B * T, D)

        bias_free_logits = self.router(x)
        biased_logits = bias_free_logits + self.expert_bias

        if self.training and update_stats:
            temp = max(0.1, 1.0 - self._global_step / 5000)
            gate_weights = F.softmax(bias_free_logits / temp, dim=-1)
        else:
            gate_weights = F.softmax(bias_free_logits, dim=-1)

        top_vals, top_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        flat_idx = top_idx.reshape(B * T, self.top_k)
        flat_vals = torch.gather(gate_weights, dim=-1, index=top_idx).reshape(B * T, self.top_k)

        capacity = int(math.ceil((B*T) / self.num_experts * MOE_CAPACITY_MULTIPLIER))
        out = torch.zeros_like(flat_x)

        if self.shared_expert:
            shared_out = self.shared(flat_x)
            out += self.shared_scale * shared_out

        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            val_k = flat_vals[:, k]
            for e in range(self.num_experts):
                mask = (idx_k == e)
                indices = mask.nonzero(as_tuple=True)[0]
                if indices.numel() == 0:
                    continue
                dropped = indices.numel() - capacity
                if dropped > 0:
                    scores = val_k[indices]
                    keep = torch.argsort(scores, descending=True)[:capacity]
                    self.overflow_counter += dropped
                    indices = indices[keep]
                    val_k_keep = val_k[indices]
                else:
                    val_k_keep = val_k[indices]

                expert_input = flat_x[indices]

                orig_tokens = expert_input.size(0)
                pad = (-orig_tokens) % 16

                if pad:
                     expert_input = F.pad(expert_input, (0, 0, 0, pad))

                expert_output = self.experts[e](expert_input)

                if pad:
                    expert_output = expert_output[:orig_tokens]

                out[indices] += expert_output * val_k_keep.unsqueeze(-1)

        if self.training and update_stats:
            self._update_routing_stats(flat_x, flat_idx)

        return out.reshape(B, T, D)

    @torch.no_grad()
    def _update_routing_stats(self, flat_x, flat_idx):
        """Update non-gradient routing state without retaining an activation graph."""
        counts = torch.bincount(flat_idx.flatten(), minlength=self.num_experts)
        target = (flat_x.size(0) * self.top_k) / self.num_experts
        self.expert_bias += (counts - target).sign() * MOE_BIAS_UPDATE_RATE
        self.expert_bias.clamp_(-5.0, 5.0)

        centroids = []
        for expert_idx in range(self.num_experts):
            indices = (flat_idx == expert_idx).any(dim=1).nonzero(as_tuple=True)[0]
            if indices.numel() > 0:
                centroids.append(flat_x[indices].mean(dim=0))
            else:
                centroids.append(torch.zeros(self.dim, device=flat_x.device, dtype=flat_x.dtype))
        self.last_centroids = torch.stack(centroids)

    @torch.no_grad()
    def update_routing_stats(self, x):
        """Update routing state for checkpointed forwards exactly once."""
        B, T, D = x.shape
        flat_x = x.detach().reshape(B * T, D)
        top_idx = torch.topk(self.router(x) + self.expert_bias, self.top_k, dim=-1).indices
        self._update_routing_stats(flat_x, top_idx.reshape(B * T, self.top_k))

# ─── Transformer Block ──────────────────────────────────────────
