import torch

def next_batch():
    global data_iter
    try:
        return next(data_iter)
    except StopIteration:
        print("⚠️ Training stream exhausted – restarting")
        data_iter = iter(get_dataloader(is_holdout=False))
        return next(data_iter)

