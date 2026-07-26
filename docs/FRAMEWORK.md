# Triune framework guide

## Architecture and ownership

`config.py` is the canonical source of model and research defaults. It keeps
the established research direction intact: GLA, depth exits, MoE, centroid
steering, custom routing objectives, and GaLore are unchanged.
`triune.configs.build_config()` creates one validated config dictionary per run.

| Area | Public API | Responsibility |
| --- | --- | --- |
| Model | `triune.model.build_model(config)` | Builds the Triune architecture. |
| Data | `triune.data.load_tokenizer`, `build_dataloader` | BPE loading and streaming train/eval data. |
| Optimizer | `triune.optim.factory.build_optimizer` | Centroid-steered GaLore / fallback optimizer. |
| Precision | `triune.recipes.build_precision_context` | BF16 or Transformer Engine NVFP4 context. |
| Training | `triune.trainer.Trainer` | Train, evaluate, log, checkpoint, resume. |
| Inference | `triune.inference` | Checkpoint loading and text generation. |

Evaluation batches are materialized once from a bounded stream. Training starts
after that reserved stream region, so it cannot accidentally train on the
evaluation slice.

## Environment

The project needs PyTorch with CUDA, `datasets`, `tokenizers`,
`flash-linear-attention`, and optionally `bitsandbytes` and `wandb`.

For the current custom Transformer Engine build:

```bash
cd /mnt/c/Users/yashb_f1ls/OneDrive/Documents/TriuneTransformer
export PYTHONPATH=/home/yasho4867/TransformerEngine_Native/TransformerEngine:$PYTHONPATH
```

NVFP4 requires Transformer Engine with `NVFP4BlockScaling` support and a
Blackwell-class GPU (SM100+). The training command checks both before starting.

## Commands

### Train

```bash
# Fresh NVFP4 run
python scripts/train.py --fresh --use_fp4

# Fresh BF16 run
python scripts/train.py --fresh

# Typical overrides
python scripts/train.py --fresh --use_fp4 \
  --batch_size 8 --grad_accum_steps 4 --seq_len 256 \
  --total_steps 50000 --lr 1e-4

# Disable external logging and HF token prompt
python scripts/train.py --fresh --no_wandb --no_hf_login
```

### Resume

```bash
# Restore model, optimizer, step, router EMA, and W&B run identity
python scripts/train.py --resume_latest checkpoints_full/latest.pt --use_fp4

# Load best weights only; optimizer and LR schedule restart
python scripts/train.py --resume_best --use_fp4
```

### Chat and utilities

```bash
python scripts/chat.py --checkpoint checkpoints_full/best.pt
python scripts/tokenizer.py --output triune_tokenizer.json
python scripts/gpucheck.py
python scripts/debug.py
python scripts/profile_model.py --checkpoint checkpoints_full/latest.pt
python tests/test_framework.py
```

In chat, prefix the prompt with `reflex `, `limbic `, `cortex `, or `auto ` to
choose a depth route.

## Programmatic use

```python
import torch

from triune.configs import build_config
from triune.data import build_dataloader, load_tokenizer
from triune.model import build_model
from triune.optim.factory import build_optimizer
from triune.recipes import build_precision_context
from triune.trainer import NullLogger, Trainer

config = build_config({"use_fp4": True})
tokenizer = load_tokenizer("triune_tokenizer.json")
config["vocab_size"] = tokenizer.get_vocab_size()
device = torch.device("cuda")
sep_id = tokenizer.token_to_id("[SEP]")

model = build_model(config).to(device).bfloat16()
trainer = Trainer(
    model=model,
    optimizer=build_optimizer(model, config),
    train_loader=build_dataloader(tokenizer, config, sep_id, is_holdout=False),
    eval_loader=build_dataloader(tokenizer, config, sep_id, is_holdout=True),
    tokenizer=tokenizer,
    config=config,
    device=device,
    precision_context=build_precision_context(use_fp4=config["use_fp4"], device=device),
    logger=NullLogger(),
)
trainer.fit()
```

## Checkpoints

The default checkpoint directory is `checkpoints_full`.

- `latest.pt` is written periodically, at normal completion, and before an
  unhandled exception is re-raised.
- `best.pt` is written when Cortex evaluation loss improves.

Checkpoints contain model parameters, optimizer state, config, depth-usage EMA,
best evaluation loss, training step, and W&B run ID. Compiled-forward keys are
normalized during restoration.
