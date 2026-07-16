import math

import torch
import torch.nn as nn
from torch.amp import autocast

@torch.no_grad()
def run_eval(force_depth=None):
    model.eval()
    losses = []
    for xb, yb in eval_batches:
        xb, yb = xb.to(device), yb.to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(xb, force_depth=force_depth)
            loss = nn.CrossEntropyLoss(ignore_index=pad_token_id)(
                logits.contiguous().view(-1, vocab_size),
                yb.contiguous().view(-1)
            )
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

SAMPLE_PROMPT = "The history of"

@torch.no_grad()
def run_sample():
    model.eval()
    ids = torch.tensor(tokenizer.encode(SAMPLE_PROMPT).ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = ids.size(1)
    for _ in range(40):
        with autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(ids, force_depth=2)
        nxt = logits[0, -1, :].argmax().item()
        if nxt in (pad_token_id, sep_token_id):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return tokenizer.decode(ids[0, prompt_len:].tolist())

