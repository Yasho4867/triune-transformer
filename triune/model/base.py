"""Universal Model Abstraction Base Class & Registry.

Provides support for Dense Transformers, MoEs, SSM/Mamba, and User-Defined Custom Architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Type
import torch
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Decorator to register a custom model class into the Triune Model Zoo."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


class TriuneModel(nn.Module, ABC):
    """Abstract Base Class for all Triune-compatible model architectures."""

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, force_depth: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass returning (logits, aux_router_loss)."""
        pass

    @abstractmethod
    def get_num_params(self) -> int:
        """Return total parameter count."""
        pass

