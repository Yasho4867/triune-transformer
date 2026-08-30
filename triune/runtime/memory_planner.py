"""VRAM Memory Planner & Automated Hardware Budgeting Engine.

Calculates memory footprints across parameters, optimizer states, activations,
and gradient accumulation buffers to fit models cleanly on consumer GPUs (e.g. RTX 5070 8GB/12GB VRAM).
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class MemoryEstimate:
    total_params: int
    param_memory_gb: float
    optimizer_memory_gb: float
    activation_memory_gb: float
    gradient_memory_gb: float
    total_vram_gb: float
    recommended_batch_size: int
    recommended_grad_accum: int
    recommended_checkpointing: bool
    recommended_precision: str


class MemoryPlanner:
    """Hardware-aware VRAM Memory Planner."""

    @staticmethod
    def estimate_vram(
        config: dict,
        target_vram_gb: float = 8.0,
        available_vram_gb: float | None = None,
    ) -> MemoryEstimate:
        """Estimate VRAM allocation breakdown and recommend optimal training hyperparameters."""
        if available_vram_gb is None and torch.cuda.is_available():
            available_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available_vram_gb = available_vram_gb or target_vram_gb

        vocab_size = config.get("vocab_size", 32000)
        hidden_dim = config.get("hidden_dim", 1536)
        num_layers = config.get("num_layers", 24)
        num_experts = config.get("num_experts", 8)
        seq_len = config.get("seq_len", 256)
        use_fp4 = config.get("use_fp4", False)

        # Estimate parameters count
        embed_params = vocab_size * hidden_dim
        layer_params = num_layers * (4 * (hidden_dim**2) + num_experts * (3 * (hidden_dim**2)))
        total_params = embed_params + layer_params

        # Bytes per parameter depending on precision
        bytes_per_param = 0.5 if use_fp4 else 2.0  # FP4 vs BF16/FP16
        param_memory_gb = (total_params * bytes_per_param) / (1024**3)

        # Optimizer memory: CentroidSteer / 8-bit AdamW ~ 2 bytes per param vs standard 8 bytes
        optimizer_bytes_per_param = 2.0
        optimizer_memory_gb = (total_params * optimizer_bytes_per_param) / (1024**3)

        # Calculate activation memory per sample (seq_len * hidden_dim * num_layers)
        activation_per_sample_mb = (seq_len * hidden_dim * num_layers * 2 * 4) / (1024**2)

        # Target safety threshold (85% of available VRAM)
        target_limit_gb = available_vram_gb * 0.85
        base_overhead_gb = param_memory_gb + optimizer_memory_gb + 0.5  # PyTorch CUDA context overhead

        remaining_vram_gb = max(0.5, target_limit_gb - base_overhead_gb)

        # Determine optimal batch size & grad accumulation
        if remaining_vram_gb < 1.0:
            recommended_batch_size = 1
            recommended_grad_accum = 16
            recommended_checkpointing = True
            recommended_precision = "fp8" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "bf16"
        elif remaining_vram_gb < 3.0:
            recommended_batch_size = 2
            recommended_grad_accum = 8
            recommended_checkpointing = True
            recommended_precision = "fp8"
        else:
            recommended_batch_size = 4
            recommended_grad_accum = 4
            recommended_checkpointing = False
            recommended_precision = "fp8"

        activation_memory_gb = (recommended_batch_size * activation_per_sample_mb) / 1024
        gradient_memory_gb = (total_params * 2.0) / (1024**3)

        total_vram_gb = param_memory_gb + optimizer_memory_gb + activation_memory_gb + gradient_memory_gb

        return MemoryEstimate(
            total_params=total_params,
            param_memory_gb=round(param_memory_gb, 3),
            optimizer_memory_gb=round(optimizer_memory_gb, 3),
            activation_memory_gb=round(activation_memory_gb, 3),
            gradient_memory_gb=round(gradient_memory_gb, 3),
            total_vram_gb=round(total_vram_gb, 3),
            recommended_batch_size=recommended_batch_size,
            recommended_grad_accum=recommended_grad_accum,
            recommended_checkpointing=recommended_checkpointing,
            recommended_precision=recommended_precision,
        )
