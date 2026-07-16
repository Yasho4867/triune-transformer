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

# ─── Data loaders ──────────────────────────────────────────────
eval_dataloader = get_dataloader(is_holdout=True)
eval_iter = iter(eval_dataloader)
eval_batches = []
for _ in range(config["eval_batches"]):
    try:
        x, y = next(eval_iter)
    except StopIteration:
        print("⚠️ Eval stream exhausted, re-iterating")
        eval_iter = iter(eval_dataloader)
        x, y = next(eval_iter)
    eval_batches.append((x, y))

train_dataloader = get_dataloader(is_holdout=False)
data_iter = iter(train_dataloader)

