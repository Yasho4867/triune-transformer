import torch

from .centroid import *

def build_optimizer(model, config):
    """
    Construct the optimizer for a model.

    Parameters
    ----------
    model : torch.nn.Module
    config : dict
    """

    if GALORE:
        print("✅ CentroidSteerOptimizer active")
        return CentroidSteerOptimizer(
            model,
            lr=config["lr"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
            rank=GALORE_RANK,
            update_gap=GALORE_UPDATE_GAP,
            steer_scale=config["steer_scale"],
            expert_lr=GALORE_LR,
            expert_betas=GALORE_BETAS,
            expert_wd=GALORE_WEIGHT_DECAY,
        )

    if HAS_8BIT and AdamW8bit is not None:
        return AdamW8bit(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=config["betas"],
        weight_decay=config["weight_decay"],
    )
