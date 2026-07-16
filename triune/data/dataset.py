import random

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, DownloadConfig

from triune.configs.config import *

class TokenStreamDataset(IterableDataset):
    def __init__(self, tokenizer, seq_len, max_tokens, sep_token_id,
                 shuffle_buffer=SHUFFLE_BUFFER, is_holdout=False, offset=0):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.sep_token_id = sep_token_id
        self.shuffle_buffer = shuffle_buffer
        self.is_holdout = is_holdout
        self.offset = offset

    def __iter__(self):
        buffer = []
        token_count = 0
        dl_config = DownloadConfig(max_retries=10, resume_download=True)
        hf_stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            download_config=dl_config
        )
        skip_tokens = self.offset
        if skip_tokens > 0:
            for sample in hf_stream:
                text = sample.get("text", "")
                if not text.strip():
                    continue
                ids = self.tokenizer.encode(text).ids
                if skip_tokens > len(ids):
                    skip_tokens -= len(ids)
                    continue
                else:
                    ids = ids[skip_tokens:]
                    skip_tokens = 0
                    buffer.extend(ids)
                    break

        chunk_buffer = []
        for sample in hf_stream:
            text = sample.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text).ids
            buffer.extend(ids)
            buffer.append(self.sep_token_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                chunk_buffer.append((torch.tensor(chunk[:-1], dtype=torch.long),
                                     torch.tensor(chunk[1:], dtype=torch.long)))
                token_count += self.seq_len
                if len(chunk_buffer) >= self.shuffle_buffer:
                    random.shuffle(chunk_buffer)
                    for x, y in chunk_buffer:
                        yield x, y
                    chunk_buffer.clear()
                if self.max_tokens is not None and token_count >= self.max_tokens:
                    # Evaluation commonly stops before the shuffle buffer fills.
                    # Flush its completed sequences instead of returning an empty
                    # iterable to the DataLoader.
                    random.shuffle(chunk_buffer)
                    for x, y in chunk_buffer:
                        yield x, y
                    return
        random.shuffle(chunk_buffer)
        for x, y in chunk_buffer:
            yield x, y

def get_dataloader(is_holdout=False):
    if is_holdout:
        max_tokens = args.eval_batches * args.batch_size * args.seq_len
        offset = 0
    else:
        max_tokens = None
        offset = args.eval_batches * args.batch_size * args.seq_len
    ds = TokenStreamDataset(tokenizer, seq_len=config["seq_len"],
                            max_tokens=max_tokens,
                            sep_token_id=sep_token_id,
                            shuffle_buffer=SHUFFLE_BUFFER,
                            is_holdout=is_holdout,
                            offset=offset)
    return DataLoader(ds, batch_size=config["batch_size"], num_workers=0, drop_last=True)

