# Triune Framework Engine (`triune-framework`)

Standalone Python AI Framework Engine for model pre-training, fine-tuning, memory planning, and inference.

## Installation

```bash
pip install triune-framework
```

## Programmatic Usage

```python
import triune

# 1. Estimate VRAM for RTX 5070
config = triune.build_config({})
plan = triune.MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)

# 2. Load Model via Model Zoo
model = triune.load_model("triune-base")
```
