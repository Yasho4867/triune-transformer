"""Unified Model Zoo & Model Loading API.

Provides seamless loading for native Triune models ("triune-small", "triune-base", "triune-moe"),
Hugging Face models ("llama-3", "qwen-2.5"), and user-registered custom architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch

from .base import MODEL_REGISTRY
from .factory import build_model


def load_model(model_name_or_path: str | Path, **kwargs: Any) -> torch.nn.Module:
    """Unified Model Loader.

    Loads native Triune models, registered custom architectures, or local checkpoints.
    """
    from triune.configs import build_config

    model_name_str = str(model_name_or_path).lower()


    # Check local checkpoint path
    if Path(model_name_or_path).is_file():
        checkpoint = torch.load(model_name_or_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        config = build_config({**saved_config, **kwargs})
        model = build_model(config)
        state_dict = checkpoint["model_state"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        return model

    # Check registered model zoo
    if model_name_str in MODEL_REGISTRY:
        cls = MODEL_REGISTRY[model_name_str]
        return cls(**kwargs)

    # Built-in Triune presets
    if model_name_str in ("triune-small", "triune-base", "triune-moe", "triune-large"):
        if model_name_str == "triune-small":
            overrides = {"num_layers": 18, "hidden_dim": 1536, "num_heads": 12, "num_experts": 4}
        elif model_name_str == "triune-base":
            overrides = {"num_layers": 24, "hidden_dim": 1536, "num_heads": 12, "num_experts": 8}
        else:
            overrides = {"num_layers": 32, "hidden_dim": 1536, "num_heads": 12, "num_experts": 16}


        config = build_config({**overrides, **kwargs})
        return build_model(config)

    # Fallback to default build_model with kwargs as config overrides
    config = build_config(kwargs)
    return build_model(config)
