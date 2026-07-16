import math

def lr_schedule(step):
    if step < config["warmup_steps"]:
        return config["lr"] * (step + 1) / config["warmup_steps"]
    progress = (step - config["warmup_steps"]) / (config["total_steps"] - config["warmup_steps"])
    progress = min(progress, 1.0)
    return config["min_lr"] + 0.5 * (config["lr"] - config["min_lr"]) * (1 + math.cos(math.pi * progress))

def set_optimizer_lr(lr):
    """Apply the schedule to either the custom optimizer or a stock torch optimizer."""
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

