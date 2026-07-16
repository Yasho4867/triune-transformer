import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotary import *
from .attention import *
from .norms import *

class HybridAttention(nn.Module):
    def __init__(self, dim, heads, use_fp4=True):
        super().__init__()
        self.dim = dim
        self.heads = heads
        # Only GLA – no separate q/k/v projections; they live inside GLA.
        self.gla = VectorisedGLA(dim, heads, GLA_HEAD_DIM)

    def forward(self, x, cache=None):
        return self.gla(x, cache)

# ─── MoE FFN ────────────────────────────────────────────────────
