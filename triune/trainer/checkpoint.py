import os
import torch

def save_latest(step, loss):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "best_eval_loss": best_eval_loss,
        "config": config,
        "depth_usage_ema": depth_usage_ema,
        "wandb_run_id": wandb.run.id if not args.no_wandb else None,
    }, os.path.join(config["checkpoint_dir"], "latest.pt"))

def save_best(step, loss):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "best_eval_loss": best_eval_loss,
        "config": config,
        "depth_usage_ema": depth_usage_ema,
        "wandb_run_id": wandb.run.id if not args.no_wandb else None,
    }, os.path.join(config["checkpoint_dir"], "best.pt"))

