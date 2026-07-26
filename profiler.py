"""Profile a Triune training forward/backward pass on demand."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from triune.configs import build_config
from triune.data import load_tokenizer
from triune.model import build_model


def profile_model(
    *,
    checkpoint_path: str | Path = "checkpoints_full/latest.pt",
    tokenizer_path: str | Path = "triune_tokenizer.json",
    output_path: str | Path = "profiler_output.txt",
    batch_size: int = 4,
    seq_len: int = 256,
    warmup_steps: int = 5,
    active_steps: int = 10,
) -> str:
    """Run a bounded profile and return the generated profiler table."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU profiling requires CUDA")
    device = torch.device("cuda")
    tokenizer = load_tokenizer(tokenizer_path)
    config = build_config({"vocab_size": tokenizer.get_vocab_size(), "seq_len": seq_len})
    checkpoint_path = Path(checkpoint_path)
    state_dict = None
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"]
        saved = checkpoint.get("config", {})
        config = build_config({
            "vocab_size": saved.get("vocab_size", tokenizer.get_vocab_size()),
            "hidden_dim": saved.get("hidden_dim"),
            "num_layers": saved.get("num_layers"),
            "use_fp4": saved.get("use_fp4", any(".ffn.experts." in key and ".linear.weight" in key for key in state_dict)),
            "seq_len": seq_len,
        })
        if any(key.startswith("_orig_mod.") for key in state_dict):
            state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model = build_model(config).to(device).bfloat16().train()
    if state_dict is not None:
        model.load_state_dict(state_dict)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    def training_step():
        model.zero_grad(set_to_none=True)
        inputs = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        labels = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs, force_depth=2)
            loss = loss_fn(logits.reshape(-1, config["vocab_size"]), labels.reshape(-1))
        loss.backward()

    for _ in range(warmup_steps):
        training_step()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
        for _ in range(active_steps):
            training_step()
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    Path(output_path).write_text(table, encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile Triune training steps")
    parser.add_argument("--checkpoint", default="checkpoints_full/latest.pt")
    parser.add_argument("--tokenizer_path", default="triune_tokenizer.json")
    parser.add_argument("--output", default="profiler_output.txt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--active_steps", type=int, default=10)
    args = parser.parse_args(argv)
    print(profile_model(
        checkpoint_path=args.checkpoint, tokenizer_path=args.tokenizer_path, output_path=args.output,
        batch_size=args.batch_size, seq_len=args.seq_len, warmup_steps=args.warmup_steps, active_steps=args.active_steps,
    ))


if __name__ == "__main__":
    main()
