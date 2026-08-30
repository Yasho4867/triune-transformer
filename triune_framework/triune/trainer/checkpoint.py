"""Checkpoint persistence and restoration for :class:`triune.Trainer`."""

from __future__ import annotations

from pathlib import Path

import torch


def _checkpoint_payload(trainer, engine, step: int, loss: float) -> dict:
    return {
        "step": step,
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "loss": loss,
        "best_eval_loss": engine.best_eval_loss,
        "config": trainer.config,
        "depth_usage_ema": engine.depth_usage_ema,
        "wandb_run_id": trainer.logger.run_id,
    }


def save_latest(trainer, engine, step: int, loss: float) -> Path:
    path = Path(trainer.config["checkpoint_dir"]) / "latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(trainer, engine, step, loss), path)
    return path


def save_best(trainer, engine, step: int, loss: float) -> Path:
    path = Path(trainer.config["checkpoint_dir"]) / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(trainer, engine, step, loss), path)
    return path


def load_checkpoint(trainer, path: str | Path, *, load_optimizer: bool) -> dict:
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    trainer.model.load_state_dict(state_dict)
    if load_optimizer:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
