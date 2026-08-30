# Triune Transformer

Triune is an experimental language-model training project based on its existing
GLA attention, adaptive-depth routing, Reflex/Limbic/Cortex exits,
Mixture-of-Experts FFNs, centroid-steered GaLore optimizer, and optional
Transformer Engine NVFP4 execution.

The codebase is a callable Python framework: importing a module does not parse
arguments, download data, build a 2.5B model, or start training.

## Quick start

```bash
cd /mnt/c/Users/yashb_f1ls/OneDrive/Documents/TriuneTransformer
export PYTHONPATH=/home/yasho4867/TransformerEngine_Native/TransformerEngine:$PYTHONPATH
python scripts/train.py --fresh --use_fp4
```

## Documentation

- 📘 [Dynamic & Modular Architecture Guide](docs/DYNAMIC_MODULAR_GUIDE.md) - How to customize layers, MoE experts, head dimensions, VRAM budgeting, and CLI options.
- 📐 [Mathematical Foundations & Proofs](docs/MATH_FOUNDATIONS.md) - Formal LaTeX proofs for GLA covariance, dual-sided GaLore subspace projections, and Gumbel-Softmax ST routing.
- 🛠️ [Operational Framework Guide](docs/FRAMEWORK.md) - Training APIs, memory planner, checkpointing, and Studio API endpoints.

## Verify the Framework

```bash
# Run framework smoke tests (CPU/CUDA)
python tests/test_framework.py
```
