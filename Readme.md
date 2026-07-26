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

See [the framework guide](docs/FRAMEWORK.md) for all commands, checkpoint
behavior, Python APIs, configuration, and NVFP4 requirements.

## Layout

```text
triune/             Reusable model, data, optimizer, recipes, trainer, inference
scripts/            Thin command-line entry points
tests/              Dependency-free framework smoke test
config.py           Canonical research and model defaults
docs/FRAMEWORK.md   Operational guide
```

## Verify the framework

```bash
python tests/test_framework.py
```

This runs a CPU-sized one-step training and checkpoint-resume test; it does
not allocate the production model.
