import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ROPE_MAX_SEQ_LEN

from .config import *
class VectorisedGLA(nn.Module):
    def __init__(self, dim, heads, head_dim=GLA_HEAD_DIM):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.hidden_dim = heads * head_dim
        self.q_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.g_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.use_rope = USE_ROPE
        if self.use_rope:
            self.rope = RotaryEmbedding(head_dim, max_seq_len=ROPE_MAX_SEQ_LEN)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        H, HD = self.heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, HD)
        k = self.k_proj(x).view(B, T, H, HD)
        v = self.v_proj(x).view(B, T, H, HD)
        # chunk_gla expects logarithmic decay gates, not probabilities.
        g = F.logsigmoid(self.g_proj(x).view(B, T, H, HD))

        # ELU+1 feature map – applied before RoPE (per GLA best practices)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        if self.use_rope:
            cos, sin = self.rope(T, x.device)
            # Apply RoPE after feature map – keeps relative position in the inner product
            q = q.transpose(1, 2)  # (B, H, T, HD)
            k = k.transpose(1, 2)
            q, k = apply_rotary(q, k, cos, sin)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # Use Triton kernel – required
        # FLA 0.5.1 uses batch/sequence/head layout and returns (output, state).
        out, _ = chunk_gla(q, k, v, g, scale=None)
        out = out.reshape(B, T, self.hidden_dim)
        return self.out_proj(out), None

# ─── RMSNorm ────────────────────────────────────────────────────
