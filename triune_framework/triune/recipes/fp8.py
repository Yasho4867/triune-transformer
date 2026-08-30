"""FP8 (E4M3 / Delayed Scaling) precision context support."""

from __future__ import annotations

import torch

from .bf16 import bf16_autocast


def build_fp8_precision_context(*, device, use_te: bool = True) -> callable:
    """Build FP8 precision autocast context.

    Tries Transformer Engine FP8 (DelayedScaling) first, falling back to
    PyTorch native float8_e4m3fn autocast if TE is unavailable.
    """
    if device.type != "cuda":
        raise RuntimeError("FP8 requires CUDA")

    major, minor = torch.cuda.get_device_capability(device)
    if major < 8 or (major == 8 and minor < 9):
        raise RuntimeError(f"FP8 requires Ada Lovelace/Hopper/Blackwell GPU (SM89+); found SM{major}{minor}")

    if use_te:
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format

            recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
            autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast", None)
            if autocast is not None:
                return lambda: autocast(enabled=True, recipe=recipe)
        except (ImportError, Exception):
            pass

    # For native PyTorch without TransformerEngine scaled GEMMs, use hardware BF16 autocast
    return lambda: torch.amp.autocast("cuda", dtype=torch.bfloat16)
