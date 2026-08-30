# Dynamic & Modular Architecture Guide

The Triune framework is engineered with a **fully dynamic, modular, and configurable architecture**. Every component—from the layer depth and head dimensions to the MoE expert counts, early-exit layers, Gumbel router parameters, and GaLore optimizer rank—can be customized dynamically via Python API or the unified CLI.

---

## 1. Dynamic Model Definition (Python API)

You can define any custom Triune architecture on the fly without modifying source code:

```python
import torch
from triune.model import TriuneTransformer

# Define a custom dynamic Triune architecture
model = TriuneTransformer(
    vocab_size=32000,           # Vocabulary size
    hidden_dim=1536,            # Must equal num_heads * head_dim
    num_layers=18,              # Dynamic total layer count
    num_heads=12,               # Attention heads
    head_dim=128,               # Gated Linear Attention head dimension
    num_experts=4,              # Dynamic MoE experts per layer (e.g. 2, 4, 8, 16)
    router_prefix_layers=3,     # Layers before early-exit routing decision
    reflex_exit_layer=6,        # Reflex exit layer index
    limbic_exit_layer=14,       # Limbic exit layer index (must be < num_layers)
    use_fp4=False               # Set True for FP4/NVFP4 quantization
)

print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Architectural Constraints & Invariants:
1. **Hidden Dimension Invariant**:
   $$\text{hidden\_dim} = \text{num\_heads} \times \text{head\_dim}$$
2. **Layer Ordering Constraint**:
   $$\text{router\_prefix\_layers} < \text{reflex\_exit\_layer} < \text{limbic\_exit\_layer} < \text{num\_layers}$$
3. **MoE Activation**:
   - Layers $0 \le i \le \text{reflex\_exit\_layer}$: Dense feed-forward networks.
   - Layers $i > \text{reflex\_exit\_layer}$: Dynamic Mixture-of-Experts with `num_experts` and a shared expert.

---

## 2. Dynamic Hardware & VRAM Memory Planning

Triune includes an automated **Hardware Memory Planner** (`MemoryPlanner`) that inspects your GPU before allocation and calculates the exact VRAM footprint for any configuration.

### VRAM Calculation Breakdown:
1. **Model Weights Footprint**:
   $$M_{\text{weights}} = \frac{N_{\text{params}} \times \text{bytes\_per\_param}}{1024^3}\text{ GiB}$$
   - `float32`: 4.0 bytes/param
   - `bfloat16` / `fp16`: 2.0 bytes/param
   - `fp8`: 1.0 byte/param
   - `fp4` / `nvfp4`: 0.5 bytes/param

2. **GaLore Centroid Optimizer Memory**:
   - Standard AdamW: $8.0\text{ bytes/param}$ ($16.0\text{ GB}$ for 2B params)
   - **Triune CentroidSteerOptimizer (GaLore)**: Projects gradients into rank-$r$ subspaces, tracking state only for active rank projections $\approx 2.0\text{ bytes/param}$ ($4.0\text{ GB}$ for 2B params, a **$4\times$ to $6\times$ reduction**).

3. **Activation Cache with Selective Checkpointing**:
   - With gradient checkpointing: Activations scale with sequence length and batch size:
     $$M_{\text{act}} \approx \frac{B \times T \times D \times 2}{1024^2}\text{ MiB}$$

### Hardware Budget Recommendations:

| Target Hardware | VRAM | Recommended Preset | Dynamic Config Overrides |
| :--- | :--- | :--- | :--- |
| **RTX 4060 / 5070 Laptop** | **8 GB** | `triune-small` | `--num_layers 18 --num_experts 4 --batch_size 2 --grad_accum 8` |
| **RTX 4070 / 5070 Ti / 3080** | **12 GB** | `triune-small` / `fp8-base` | `--num_layers 20 --num_experts 6 --batch_size 4 --grad_accum 4` |
| **RTX 4080 / 4090** | **16 - 24 GB** | `triune-base` | `--num_layers 24 --num_experts 8 --batch_size 4 --grad_accum 4` |
| **A100 / H100 / H200** | **80 - 141 GB** | `triune-moe` | `--num_layers 32 --num_experts 16 --batch_size 16 --grad_accum 2` |

---

## 3. Dynamic Gumbel-Softmax Routing

The router uses a **Straight-Through (ST) Gumbel-Softmax** module that provides discrete execution paths while allowing continuous backpropagation:

```python
# Forward pass through model with dynamic temperature
logits, route_logits = model(x, temperature=0.8)

# Access the native load-balancing regularization loss
balance_loss = model.last_balance_loss
```

### Customizing Router Parameters:
```python
from triune.model.router import GumbelSoftmaxRouter

router = GumbelSoftmaxRouter(
    hidden_dim=1536,
    target_depth_dist=(0.40, 0.35, 0.25),  # 40% Reflex, 35% Limbic, 25% Cortex
    balance_coef=0.30                      # Load-balancing penalty multiplier
)
```

---

## 4. Full Training CLI with Dynamic Flags

You can override every hyperparameter directly from the command line:

```bash
python scripts/train.py \
    --model_name triune-small \
    --num_layers 18 \
    --num_experts 4 \
    --seq_len 256 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --total_steps 50000 \
    --target_depth_dist 0.34,0.33,0.33 \
    --steer_scale 0.20 \
    --shuffle_buffer 64 \
    --save_every 1000 \
    --eval_every 500
```

### Dynamic CLI Options Reference:
- `--model_name`: Base preset (`triune-small`, `triune-base`, `triune-moe`).
- `--num_layers`: Total layer depth (must exceed `--limbic_exit_layer`).
- `--num_experts`: Number of MoE experts per layer (default: 8).
- `--dataset_name`: Hugging Face dataset (e.g. `roneneldan/TinyStories`, `HuggingFaceFW/fineweb-edu`) or local file path (`data.jsonl`, `data.txt`, `data.parquet`).
- `--target_depth_dist`: Target comma-separated probabilities for early exits (e.g. `0.50,0.30,0.20`).
- `--steer_scale`: Subspace steering coefficient for centroid-augmented GaLore optimizer.
- `--shuffle_buffer`: Size of streaming token shuffle buffer (smaller values start streaming instantly).
- `--no_wandb`: Disables W&B logging for offline/local training.
