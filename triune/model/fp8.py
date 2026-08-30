"""Native FP8 (E4M3) Hardware Scaled GEMM Linear Layer for Triune Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_to_fp8_e4m3(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize a float tensor to float8_e4m3fn with a single maximum scale factor."""
    max_val = tensor.abs().max().clamp_min(1e-8)
    scale = 448.0 / max_val.float()  # 448 is max representable finite float8_e4m3fn
    tensor_scaled = (tensor.float() * scale).clamp(-448.0, 448.0)
    tensor_fp8 = tensor_scaled.to(torch.float8_e4m3fn)
    scale_inv = (1.0 / scale).to(tensor.dtype)
    return tensor_fp8, scale_inv


class FP8Linear(nn.Module):
    """Hardware-accelerated FP8 Linear layer utilizing native torch._scaled_mm."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype or torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype or torch.bfloat16))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if hardware FP8 scaled_mm is available
        if x.is_cuda and hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
            try:
                orig_shape = x.shape
                flat_x = x.reshape(-1, self.in_features)
                
                # Dynamic per-tensor quantization to FP8 E4M3
                x_fp8, scale_x = quantize_to_fp8_e4m3(flat_x)
                w_fp8, scale_w = quantize_to_fp8_e4m3(self.weight)
                
                # Execute native hardware FP8 GEMM on Tensor Cores
                out = torch._scaled_mm(
                    x_fp8,
                    w_fp8.t(),
                    scale_a=scale_x.float().unsqueeze(0),
                    scale_b=scale_w.float().unsqueeze(0),
                    out_dtype=x.dtype
                )
                
                if self.bias is not None:
                    out = out + self.bias
                return out.reshape(*orig_shape[:-1], self.out_features)
            except Exception:
                # Fallback to standard linear if hardware driver rejects specific batch shape
                return F.linear(x, self.weight, self.bias)
        
        return F.linear(x, self.weight, self.bias)
