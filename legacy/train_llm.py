import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from model import TriuneTransformer, MoE_FFN, FP4Linear
from tokenizers import Tokenizer
from torch.amp import autocast
import os, math, argparse, random, time, getpass, signal, sys
from config import *

from triune.data.dataset import *
from triune.data.dataloader import *

from triune.optim.centroid import *
from triune.optim.factory import *

from triune.trainer.scheduler import *
from triune.trainer.checkpoint import *
from triune.trainer.evaluation import *
from triune.trainer.callbacks import *


# ─── NVFP4 Recipe ──────────────────────────────────────────────
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, NVFP4BlockScaling
    HAS_TE = True
except ImportError:
    HAS_TE = False
    te = None
    NVFP4BlockScaling = None

# ─── bitsandbytes for 8‑bit optimizer ────────────────────────
try:
    from bitsandbytes.optim import AdamW8bit
    HAS_8BIT = True
except ImportError:
    HAS_8BIT = False
    AdamW8bit = None

try:
    from datasets import load_dataset, DownloadConfig
except ImportError:
    raise ImportError("datasets library not installed")

# ─── Performance tweaks ──────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--resume_best", action="store_true")
parser.add_argument("--resume_latest", type=str, default=None)
parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
parser.add_argument("--fresh", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
parser.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS)
parser.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=GRAD_CHECKPOINT)
parser.add_argument("--eval_every", type=int, default=EVAL_EVERY)
parser.add_argument("--eval_batches", type=int, default=EVAL_BATCHES)
parser.add_argument("--target_depth_dist", type=str, default=",".join(map(str, TARGET_DEPTH_DIST)))
parser.add_argument("--usage_ema_decay", type=float, default=0.98)
parser.add_argument("--bias_strength", type=float, default=DEPTH_BIAS_STRENGTH)
parser.add_argument("--balance_coef", type=float, default=DEPTH_BALANCE_COEF)
parser.add_argument("--exploration", type=str, default="linear", choices=["linear", "cosine", "none"])
parser.add_argument("--exploration_steps", type=int, default=5000)
parser.add_argument("--no_wandb", action="store_true")
parser.add_argument("--no_hf_login", action="store_true")
parser.add_argument("--use_fp4", "--use_nvfp4", dest="use_fp4", action="store_true",
                    help="Train Transformer Engine MoE linears with the NVFP4 E2M1 recipe")
parser.add_argument("--steer_scale", type=float, default=STEER_SCALE)
args = parser.parse_args()

# ─── Check seq_len vs ROPE_MAX_SEQ_LEN ────────────────────
if args.seq_len > ROPE_MAX_SEQ_LEN:
    raise ValueError(f"--seq_len ({args.seq_len}) exceeds ROPE_MAX_SEQ_LEN ({ROPE_MAX_SEQ_LEN})")

config = {
    "vocab_size": VOCAB_SIZE,
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "batch_size": args.batch_size,
    "grad_accum_steps": args.grad_accum_steps,
    "seq_len": args.seq_len,
    "total_steps": args.total_steps,
    "save_every": SAVE_EVERY,
    "log_every": LOG_EVERY,
    "eval_every": args.eval_every,
    "eval_batches": args.eval_batches,
    "target_depth_dist": [float(x) for x in args.target_depth_dist.split(",")],
    "usage_ema_decay": args.usage_ema_decay,
    "bias_strength": args.bias_strength,
    "balance_coef": args.balance_coef,
    "lr": args.lr,
    "min_lr": MIN_LR,
    "warmup_steps": WARMUP_STEPS,
    "weight_decay": WEIGHT_DECAY,
    "betas": BETAS,
    "checkpoint_dir": args.checkpoint_dir,
    "exploration": args.exploration,
    "exploration_steps": args.exploration_steps,
    "use_fp4": args.use_fp4,
    "steer_scale": args.steer_scale,
}
if len(config["target_depth_dist"]) != 3 or any(p < 0 for p in config["target_depth_dist"]):
    raise ValueError("--target_depth_dist must contain three non-negative comma-separated values")
if not math.isclose(sum(config["target_depth_dist"]), 1.0, rel_tol=0.0, abs_tol=1e-6):
    raise ValueError("--target_depth_dist must sum to 1.0")
os.makedirs(config["checkpoint_dir"], exist_ok=True)

if not args.no_wandb:
    import wandb
    run_id = None
    if args.resume_latest or args.resume_best:
        # Attempt to read run_id from checkpoint
        resume_path = args.resume_latest or os.path.join(args.checkpoint_dir, "best.pt")
        if os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                run_id = ckpt.get("wandb_run_id", None)
            except:
                pass
    wandb.init(project="triune-transformer", config=config, id=run_id, resume="allow" if run_id else None)
else:
    class DummyLogger:
        def log(self, *args, **kwargs): pass
    wandb = DummyLogger()

if not args.no_hf_login and "HF_TOKEN" not in os.environ:
    if sys.stdin.isatty():
        token = getpass.getpass("🔑 Enter your Hugging Face API token: ")
        if token.strip():
            os.environ["HF_TOKEN"] = token
            print("✅ HF_TOKEN set")
        else:
            print("⚠️ No token – rate limits may apply")
    else:
        print("⚠️ Non‑interactive session: set HF_TOKEN environment variable to avoid hanging")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.use_fp4 and (not HAS_TE or NVFP4BlockScaling is None):
    raise RuntimeError(
        "--use_fp4 requires Transformer Engine with NVFP4BlockScaling support (2.17.0 or newer)."
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

if args.use_fp4:
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 10:
        raise RuntimeError(
            f"--use_fp4 requires a Blackwell-class GPU (SM100+); found compute capability {capability[0]}.{capability[1]}."
        )

tokenizer = Tokenizer.from_file("triune_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
config["vocab_size"] = vocab_size
pad_token_id = tokenizer.token_to_id("[PAD]")
sep_token_id = tokenizer.token_to_id("[SEP]")
if pad_token_id is None or sep_token_id is None:
    raise ValueError("Tokenizer must define [PAD] and [SEP] special tokens")

# ─── Data ──────────────────────────────────────────────────────
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

# ─── Model ─────────────────────────────────────────────────────
model = TriuneTransformer(vocab_size=vocab_size, use_fp4=args.use_fp4)
model = model.to(device).bfloat16()

if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
    print("✅ Selective gradient checkpointing enabled")

total_params = sum(p.numel() for p in model.parameters())
print(f"Params: {total_params:,}")

# ─── CentroidSteerOptimizer ────────────────────────────────────
class CentroidSteerOptimizer(torch.optim.Optimizer):
    def __init__(self, model, lr, betas, weight_decay,
                 rank=GALORE_RANK, update_gap=GALORE_UPDATE_GAP,
                 steer_scale=0.1,
                 expert_lr=GALORE_LR, expert_betas=GALORE_BETAS, expert_wd=GALORE_WEIGHT_DECAY):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([{'params': [dummy_param]}], defaults)
        self.rank = rank
        self.update_gap = update_gap
        self.step_count = 0
        self.steer_scale = steer_scale
        self.expert_lr = expert_lr
        self.expert_betas = expert_betas
        self.expert_wd = expert_wd

        self.non_expert_params = []
        self.layer_groups = []

        group_idx = 0
        num_groups = 0
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2:
                                num_groups += 1

        group_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, MoE_FFN):
                for expert_idx, expert in enumerate(module.experts):
                    for subname, submod in expert.named_modules():
                        if hasattr(submod, 'weight') and isinstance(submod.weight, torch.nn.Parameter):
                            p = submod.weight
                            if p.requires_grad and p.dim() >= 2:
                                stagger = -(group_idx * (update_gap // max(1, num_groups)))
                                self.layer_groups.append({
                                    'module': module,
                                    'expert_idx': expert_idx,
                                    'param': p,
                                    'projection': None,
                                    'proj_step': stagger,
                                    'state': {'momentum': None, 'variance': None, 'step': 0}
                                })
                                group_idx += 1

        expert_param_ids = {id(g['param']) for g in self.layer_groups}
        for name, param in model.named_parameters():
            if id(param) not in expert_param_ids:
                self.non_expert_params.append(param)

        if HAS_8BIT and AdamW8bit is not None:
            self.base_optimizer = AdamW8bit(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            self.base_optimizer = torch.optim.AdamW(self.non_expert_params, lr=lr, betas=betas, weight_decay=weight_decay)

    def zero_grad(self):
        self.base_optimizer.zero_grad()
        for group in self.layer_groups:
            p = group['param']
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def set_lr(self, lr):
        for pg in self.base_optimizer.param_groups:
            pg['lr'] = lr

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        self.base_optimizer.step()

        expert_lr = self.expert_lr
        expert_beta1, expert_beta2 = self.expert_betas
        expert_wd = self.expert_wd

        for group in self.layer_groups:
            module = group['module']
            expert_idx = group['expert_idx']
            p = group['param']
            state = group['state']

            if p.grad is None:
                continue

            grad = p.grad.data
            m, n = grad.shape
            grad_fp32 = grad.float()

            if (self.step_count - group['proj_step']) >= self.update_gap or group['projection'] is None:
                U, S, V = torch.svd_lowrank(grad_fp32, q=min(self.rank + 10, m, n), niter=4)
                rank = min(self.rank, m, n)
                P = U[:, :rank] @ torch.diag(S[:rank])
                group['projection'] = P.to(grad.dtype)
                group['proj_step'] = self.step_count
                state['momentum'] = None
                state['variance'] = None
                state['step'] = 0

            P = group['projection']
            zero_col = torch.zeros(m, 1, dtype=grad.dtype, device=grad.device)

            # ─── Centroid steering ────────────────────────────
            centroids = module.last_centroids
            steer_applied = False
            if centroids is not None and expert_idx < centroids.size(0) and self.steer_scale > 0:
                c = centroids[expert_idx]
                if c.size(0) != m:
                    expert = module.experts[expert_idx]
                    first_linear = expert[0]
                    if isinstance(first_linear, FP4Linear):
                        w = first_linear.linear.weight
                        b = first_linear.linear.bias
                    else:
                        w = first_linear.weight
                        b = first_linear.bias
                    c_projected = F.linear(c.unsqueeze(0), w, b).squeeze(0)
                else:
                    c_projected = c

                c_norm = c_projected.norm()
                if c_norm > 1e-8:
                    c_hat = c_projected / c_norm
                    c_proj = P @ (P.T @ c_hat)
                    c_res = c_hat - c_proj
                    c_res_norm = c_res.norm()
                    if c_res_norm > 1e-8:
                        c_orth = c_res / c_res_norm
                        P_aug = torch.cat([P, self.steer_scale * c_orth.unsqueeze(1)], dim=1)
                        steer_applied = True

            if not steer_applied:
                P_aug = torch.cat([P, zero_col], dim=1)

            g_lr = P_aug.T @ grad

            state['step'] += 1
            if state['momentum'] is None:
                state['momentum'] = g_lr.clone()
                state['variance'] = g_lr.pow(2).clone()
            else:
                state['momentum'] = expert_beta1 * state['momentum'] + (1 - expert_beta1) * g_lr
                state['variance'] = expert_beta2 * state['variance'] + (1 - expert_beta2) * g_lr.pow(2)

            step = state['step']
            m_hat = state['momentum'] / (1 - expert_beta1 ** step)
            v_hat = state['variance'] / (1 - expert_beta2 ** step)

            delta_lr = m_hat / (v_hat.sqrt() + 1e-8)
            delta_full = P_aug @ delta_lr

            if expert_wd != 0:
                p.data -= expert_lr * expert_wd * p.data
            p.data -= expert_lr * delta_full.reshape(p.shape)

    def state_dict(self):
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'layer_groups': [
                {
                    'projection': g['projection'],
                    'proj_step': g['proj_step'],
                    'state': g['state']
                }
                for g in self.layer_groups
            ]
        }

    def load_state_dict(self, state_dict):
        # Integrity checks
        assert len(state_dict['layer_groups']) == len(self.layer_groups), \
            f"Layer group count mismatch: saved {len(state_dict['layer_groups'])}, current {len(self.layer_groups)}"
        for saved, cur in zip(state_dict['layer_groups'], self.layer_groups):
            # Shape check on projection if saved
            if saved['projection'] is not None:
                rows, cols = cur['param'].shape
                expected_shape = (rows, min(self.rank, rows, cols))
                assert tuple(saved['projection'].shape) == expected_shape, \
                    f"Projection shape mismatch: saved {saved['projection'].shape}, expected {expected_shape}"
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        for g, sd in zip(self.layer_groups, state_dict['layer_groups']):
            g['projection'] = sd['projection']
            g['proj_step'] = sd['proj_step']
            g['state'].update(sd['state'])

if GALORE:
    optimizer = CentroidSteerOptimizer(
        model,
        lr=config["lr"],
        betas=config["betas"],
        weight_decay=config["weight_decay"],
        rank=GALORE_RANK,
        update_gap=GALORE_UPDATE_GAP,
        steer_scale=config["steer_scale"],
        expert_lr=GALORE_LR,
        expert_betas=GALORE_BETAS,
        expert_wd=GALORE_WEIGHT_DECAY,
    )
    print("✅ CentroidSteerOptimizer active")
else:
    if HAS_8BIT and AdamW8bit is not None:
        optimizer = AdamW8bit(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=config["betas"], weight_decay=config["weight_decay"])

# ─── LR schedule ──────────────────────────────────────────────

# moved to triune.trainer.scheduler

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

def next_batch():
    global data_iter
    try:
        return next(data_iter)
    except StopIteration:
        print("⚠️ Training stream exhausted – restarting")
        data_iter = iter(get_dataloader(is_holdout=False))
        return next(data_iter)

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

loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
router_loss_fn = nn.CrossEntropyLoss()

model.train()
step = start_step
grad_accum = config["grad_accum_steps"]
print(f"\n🚀 Training from step {step} to {config['total_steps']}")

nvfp4_recipe = None
if args.use_fp4:
    # NVFP4 has its own two-level block-scaling scheme. Delayed FP8 amax
    # settings do not apply to this recipe.
    nvfp4_recipe = NVFP4BlockScaling(
        fp4_format=Format.E2M1,
        disable_stochastic_rounding = True,
        disable_rht= True,


        )
    print("✅ NVFP4 recipe enabled")

z_loss_coef = 0.001

def model_autocast():
    """Return the configured precision context for the training forward pass."""
    if nvfp4_recipe is not None:
        te_autocast = getattr(te, "autocast", None) or getattr(te, "fp8_autocast", None)
        if te_autocast is None:
            raise RuntimeError("Installed Transformer Engine exposes no autocast context manager")
        return te_autocast(enabled=True, recipe=nvfp4_recipe)
    return autocast('cuda', dtype=torch.bfloat16)

def sigint_handler(sig, frame):
    print("\n⚠️ KeyboardInterrupt – saving checkpoint...")
    save_latest(step, 0.0)
    sys.exit(0)

def sigterm_handler(sig, frame):
    print("\n⚠️ SIGTERM received – saving checkpoint...")
    save_latest(step, 0.0)
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigterm_handler)

try:
    while step < config["total_steps"]:
        lr = lr_schedule(step)
        set_optimizer_lr(lr)

        # Sync global step to MoE layers
        for module in model.modules():
            if isinstance(module, MoE_FFN):
                module._global_step = step

        optimizer.zero_grad()
        acc_loss = 0.0
        acc_lm = 0.0
        acc_router = 0.0
        acc_balance = 0.0
        acc_overflow = 0
        last_route = None

        for micro in range(grad_accum):
            x, y = next_batch()
            x, y = x.to(device), y.to(device)

            # ─── Router labels per sample ─────────────────────
            with torch.no_grad():
                # Ensure MoE layers don't update stats during label generation
                reflex, limbic, cortex, _ = model.forward_all_exits(x, update_stats=False)
                y_flat = y.contiguous().view(-1)
                loss_reflex = loss_fn(reflex.contiguous().view(-1, vocab_size), y_flat)
                loss_limbic = loss_fn(limbic.contiguous().view(-1, vocab_size), y_flat)
                loss_cortex = loss_fn(cortex.contiguous().view(-1, vocab_size), y_flat)
                valid_mask = (y_flat != pad_token_id)
                valid_per_sample = valid_mask.view(x.shape[0], -1).sum(dim=1).clamp_min(1)
                loss_reflex = (loss_reflex * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                loss_limbic = (loss_limbic * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                loss_cortex = (loss_cortex * valid_mask.float()).view(x.shape[0], -1).sum(dim=1) / valid_per_sample
                losses = torch.stack([loss_reflex, loss_limbic, loss_cortex], dim=1)
                usage_gap = target_depth_dist - depth_usage_ema
                adjusted = losses - config["bias_strength"] * usage_gap.unsqueeze(0)
                best_depth = adjusted.argmin(dim=1)
                depth_labels = best_depth
                one_hot = F.one_hot(best_depth, num_classes=3).float().mean(dim=0)
                depth_usage_ema.mul_(config["usage_ema_decay"]).add_(one_hot * (1 - config["usage_ema_decay"]))

            # Exploration
            expl_type = config["exploration"]
            expl_steps = config["exploration_steps"]
            if expl_type == "none":
                exploration_rate = 0.0
            elif expl_type == "linear":
                exploration_rate = max(0.0, 1.0 - step / expl_steps)
            elif expl_type == "cosine":
                if step < expl_steps:
                    exploration_rate = 0.5 * (1 + math.cos(math.pi * step / expl_steps))
                else:
                    exploration_rate = 0.0
            else:
                exploration_rate = 0.0

            if random.random() < exploration_rate:
                chosen_depth = random.choice([0, 1, 2])
            else:
                chosen_depth = None

            # ─── Forward pass ──────────────────────────────────
            with model_autocast():
                logits, route_logits = model(x, force_depth=chosen_depth)

            y_flat = y.contiguous().view(-1)
            lm_loss = loss_fn(logits.contiguous().view(-1, vocab_size), y_flat).mean()

            rloss = router_loss_fn(route_logits, depth_labels)
            z_loss = torch.logsumexp(route_logits, dim=-1).pow(2).mean()
            probs = torch.softmax(route_logits, dim=-1)
            mean_prob = probs.mean(dim=0)
            balance_loss = (mean_prob - target_depth_dist).pow(2).mean()

            micro_loss = lm_loss + 0.5 * rloss + z_loss_coef * z_loss + config["balance_coef"] * balance_loss
            acc_router += rloss.item()
            acc_balance += balance_loss.item()

            (micro_loss / grad_accum).backward()
            acc_loss += micro_loss.item()
            acc_lm += lm_loss.item()
            last_route = route_logits

            # Accumulate overflow
            for module in model.modules():
                if isinstance(module, MoE_FFN):
                    acc_overflow += module.overflow_counter
                    module.overflow_counter = 0

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = acc_loss / grad_accum
        lm_val = acc_lm / grad_accum
        router_val = acc_router / grad_accum
        balance_val = acc_balance / grad_accum

        if step % config["eval_every"] == 0 and step > 0:
            eval_loss = run_eval(force_depth=2)       # Cortex
            dynamic_eval_loss = run_eval(force_depth=None)  # Router's own choice
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                save_best(step, eval_loss)
            sample = run_sample()
            print(f"   📊 Eval (Cortex): {eval_loss:.4f} | Dynamic: {dynamic_eval_loss:.4f} | Perplexity: {math.exp(min(eval_loss,20)):.2f}")
            print(f"   📝 Sample: {SAMPLE_PROMPT}{sample}\n")
            if not args.no_wandb:
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/dynamic_loss": dynamic_eval_loss,
                    "eval/perplexity": math.exp(min(eval_loss, 20)),
                    "best_eval_loss": best_eval_loss,
                }, step=step)

        if step % config["save_every"] == 0 and step > 0:
            save_latest(step, loss_val)

        if step % config["log_every"] == 0:
            depth_map = {0: "Reflex", 1: "Limbic", 2: "Cortex"}
            chosen = last_route.argmax(dim=-1)[0].item()
            label = depth_labels[0].item()
            usage_str = f"R:{depth_usage_ema[0]:.2f} L:{depth_usage_ema[1]:.2f} C:{depth_usage_ema[2]:.2f}"
            print(f"Step {step:6d}/{config['total_steps']} | Loss: {loss_val:.4f} (LM: {lm_val:.4f}, Router: {router_val:.4f}, Bal: {balance_val:.4f}) "
                  f"| Router: {depth_map[chosen]} (target: {depth_map[label]}) "
                  f"| Usage: {usage_str} | LR: {lr:.2e} | VRAM: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB | Best Eval: {best_eval_loss:.4f} | Overflow: {acc_overflow}")

        if not args.no_wandb and step % config["log_every"] == 0:
            wandb.log({
                "train/loss": loss_val,
                "train/lm_loss": lm_val,
                "train/router_loss": router_val,
                "train/balance_loss": balance_val,
                "train/lr": lr,
                "usage/reflex": depth_usage_ema[0].item(),
                "usage/limbic": depth_usage_ema[1].item(),
                "usage/cortex": depth_usage_ema[2].item(),
                "vram": torch.cuda.memory_allocated(device)/1024**3,
                "best_eval_loss": best_eval_loss,
                "exploration_rate": exploration_rate,
                "overflow": acc_overflow,
            }, step=step)

        step += 1

except KeyboardInterrupt:
    print("\n⚠️ KeyboardInterrupt – saving checkpoint...")
    save_latest(step, 0.0)
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Unhandled exception: {e}")
    save_latest(step, 0.0)
    raise

save_latest(step-1, 0.0)
print(f"\n✅ Training complete! Best eval loss: {best_eval_loss:.4f}")
