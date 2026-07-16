"""
Triune dataset pipeline.

Temporary home for the dataset implementation extracted from
train_llm.py.

No functionality should change.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
from torch.utils.data import Dataset


class TokenStreamDataset(Dataset):
    """
    Dataset implementation migrated from train_llm.py.

    NOTE:
    This file intentionally mirrors the original implementation.
    Behaviour must remain identical.

    During later refactors this will be extended with
    streaming datasets, FineWeb support and memory mapping.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        seq_len: int,
    ):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y


def next_batch(
    dataset: TokenStreamDataset,
    batch_size: int,
    device: torch.device,
):
    """
    Compatibility wrapper.

    This function will later replace the implementation inside
    train_llm.py without changing behaviour.
    """

    indices = torch.randint(
        0,
        len(dataset),
        (batch_size,),
    )

    xs = []
    ys = []

    for i in indices.tolist():
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)

    return (
        torch.stack(xs).to(device, non_blocking=True),
        torch.stack(ys).to(device, non_blocking=True),
    )