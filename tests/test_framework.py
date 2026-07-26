"""Small CPU smoke tests for the callable training framework."""

from __future__ import annotations

import torch
import torch.nn as nn
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from triune.configs import build_config
from triune.trainer import NullLogger, Trainer


class _Tokenizer:
    def get_vocab_size(self):
        return 11

    @staticmethod
    def token_to_id(token):
        return {"[PAD]": 0, "[SEP]": 1}.get(token)

    @staticmethod
    def encode(_text):
        return type("Encoding", (), {"ids": [2, 3]})()

    @staticmethod
    def decode(_ids):
        return ""


class _TinyTriune(nn.Module):
    def __init__(self, vocab_size=11, hidden_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        self.router = nn.Linear(hidden_dim, 3)

    def forward(self, input_ids, force_depth=None):
        hidden = self.embedding(input_ids)
        return self.head(hidden), self.router(hidden.mean(dim=1))

    def forward_all_exits(self, input_ids, update_stats=False):
        logits, route_logits = self(input_ids)
        return logits, logits, logits, route_logits


class FrameworkTest(unittest.TestCase):
    def test_one_cpu_training_step_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run_training_and_resume(Path(directory))

    def _run_training_and_resume(self, checkpoint_dir: Path):
        config = build_config({
            "vocab_size": 11,
            "total_steps": 1,
            "warmup_steps": 0,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "seq_len": 3,
            "eval_batches": 1,
            "eval_every": 2,
            "log_every": 1,
            "save_every": 2,
            "checkpoint_dir": str(checkpoint_dir),
        })
        x = torch.tensor([[2, 3, 4], [5, 6, 7]])
        y = torch.tensor([[3, 4, 5], [6, 7, 8]])
        tokenizer = _Tokenizer()
        model = _TinyTriune()
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=config["lr"]),
            train_loader=[(x, y)],
            eval_loader=[(x, y)],
            tokenizer=tokenizer,
            config=config,
            device="cpu",
            logger=NullLogger(),
        )
        result = trainer.fit()
        self.assertEqual(result["step"], 1)

        restored = _TinyTriune()
        resumed = Trainer(
            model=restored,
            optimizer=torch.optim.AdamW(restored.parameters(), lr=config["lr"]),
            train_loader=[(x, y)],
            eval_loader=[(x, y)],
            tokenizer=tokenizer,
            config=config,
            device="cpu",
            logger=NullLogger(),
        )
        resumed.resume(checkpoint_dir / "latest.pt")
        self.assertEqual(resumed.engine.step, 1)


if __name__ == "__main__":
    unittest.main()
