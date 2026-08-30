"""Command-line entry point for callable Triune training."""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triune.configs.config import build_config
from triune.data import build_dataloader, load_tokenizer
from triune.model import build_model
from triune.optim.factory import build_optimizer
from triune.recipes import build_precision_context
from triune.trainer import NullLogger, Trainer, WandbLogger


def _depth_distribution(value: str) -> list[float]:
    try:
        return [float(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Triune Transformer")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--resume_latest", type=Path)
    parser.add_argument("--checkpoint_dir", type=Path)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--eval_batches", type=int)
    parser.add_argument("--target_depth_dist", type=_depth_distribution)
    parser.add_argument("--usage_ema_decay", type=float)
    parser.add_argument("--bias_strength", type=float)
    parser.add_argument("--balance_coef", type=float)
    parser.add_argument("--exploration", choices=("linear", "cosine", "none"))
    parser.add_argument("--exploration_steps", type=int)
    parser.add_argument("--steer_scale", type=float)
    parser.add_argument("--use_fp4", "--use_nvfp4", dest="use_fp4", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_hf_login", action="store_true")
    parser.add_argument("--model_name", choices=("triune-small", "triune-base", "triune-moe"), default=None)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_experts", type=int)
    parser.add_argument("--tokenizer_path", type=Path, default=Path("triune_tokenizer.json"))
    parser.add_argument("--shuffle_buffer", type=int)
    parser.add_argument("--dataset_name")
    parser.add_argument("--dataset_config")
    parser.add_argument("--num_workers", type=int)
    return parser


def _request_hf_token() -> None:
    if "HF_TOKEN" in os.environ:
        return
    if not sys.stdin.isatty():
        print("⚠️ Non-interactive session: set HF_TOKEN to avoid dataset rate limits")
        return
    token = getpass.getpass("🔑 Enter your Hugging Face API token: ")
    if token.strip():
        os.environ["HF_TOKEN"] = token.strip()
        print("✅ HF_TOKEN set")
    else:
        print("⚠️ No HF token provided; dataset rate limits may apply")


def _checkpoint_run_id(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("wandb_run_id")
    except (OSError, RuntimeError, KeyError):
        return None


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    overrides = {
        key: getattr(args, key)
        for key in (
            "checkpoint_dir", "seq_len", "lr", "total_steps", "batch_size", "grad_accum_steps",
            "save_every", "log_every", "eval_every", "eval_batches", "target_depth_dist", "usage_ema_decay", "bias_strength",
            "balance_coef", "exploration", "exploration_steps", "steer_scale", "dataset_name",
            "dataset_config", "num_workers", "num_layers", "num_experts", "shuffle_buffer",
        )
    }
    if args.model_name == "triune-small":
        overrides.setdefault("num_layers", 18)
        overrides.setdefault("num_experts", 4)
    elif args.model_name == "triune-base":
        overrides.setdefault("num_layers", 24)
        overrides.setdefault("num_experts", 8)
    elif args.model_name == "triune-moe":
        overrides.setdefault("num_layers", 32)
        overrides.setdefault("num_experts", 16)

    overrides["checkpoint_dir"] = str(overrides["checkpoint_dir"]) if overrides["checkpoint_dir"] else None
    overrides["use_fp4"] = args.use_fp4
    config = build_config(overrides)
    Path(config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triune training")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    print(f"✅ GPU: {torch.cuda.get_device_name(device)}", flush=True)
    if not args.no_hf_login:
        _request_hf_token()

    resume_path = args.resume_latest or (Path(config["checkpoint_dir"]) / "best.pt" if args.resume_best else None)
    logger = NullLogger() if args.no_wandb else WandbLogger(
        project="triune-transformer", config=config, run_id=_checkpoint_run_id(resume_path)
    )
    try:
        tokenizer = load_tokenizer(args.tokenizer_path)
        config["vocab_size"] = tokenizer.get_vocab_size()
        sep_token_id = tokenizer.token_to_id("[SEP]")
        if sep_token_id is None:
            raise ValueError("Tokenizer must define a [SEP] special token")
        model = build_model(config).to(device).bfloat16()
        if args.grad_checkpoint is not False:
            model.gradient_checkpointing_enable()
            print("✅ Selective gradient checkpointing enabled", flush=True)
        print(f"Params: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
        optimizer = build_optimizer(model, config)
        train_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=False)
        eval_loader = build_dataloader(tokenizer, config, sep_token_id, is_holdout=True)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            config=config,
            device=device,
            precision_context=build_precision_context(use_fp4=config["use_fp4"], device=device),
            logger=logger,
        )
        if args.compile:
            trainer.compile()
            print("✅ torch.compile enabled", flush=True)
        if resume_path and not args.fresh:
            trainer.resume(resume_path, weights_only=args.resume_best)
            print(f"✅ Resumed from {resume_path}", flush=True)
        elif args.fresh:
            print("🆕 Fresh start", flush=True)
        print(f"🚀 Training from step {trainer.engine.step} to {config['total_steps']}", flush=True)
        return trainer.fit()
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
