"""Model construction for callable applications."""

from __future__ import annotations

from .transformer import TriuneTransformer


def build_model(config: dict | None = None) -> TriuneTransformer:
    """Build the established Triune model from a per-run config dictionary."""
    if config is None:
        return TriuneTransformer()
    return TriuneTransformer(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        use_fp4=config.get("use_fp4", True),
    )
