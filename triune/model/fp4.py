import torch
import torch.nn as nn

from .config import *



try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    te = None

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)
            
    def forward(self, x):
        return self.linear(x)
        
# ─── GLA ──────────────────────────────────────────────────────
try:
    from fla.ops.gla import chunk_gla
    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    chunk_gla = None
    raise ImportError("flash-linear-attention is required. Install with: pip install flash-linear-attention")

try:
    import transformer_engine.pytorch as te
    HAS_TE = True
except ImportError:
    HAS_TE = False
    te = None

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if HAS_TE:
            self.linear = te.Linear(in_features, out_features, bias=bias, params_dtype=torch.bfloat16)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias).to(torch.bfloat16)
            
    def forward(self, x):
        return self.linear(x)
