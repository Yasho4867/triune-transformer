from .centroid import *

if GALORE:
    optimizer = CentroidSteerOptimizer(
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
    print("✅ CentroidSteerOptimizer active")
else:
    if HAS_8BIT and AdamW8bit is not None:
        optimizer = AdamW8bit(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])

