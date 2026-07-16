#!/usr/bin/env bash
set -e

echo "=== Upgrading pip ==="
python -m pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch 2.10 + CUDA 13.0 ==="
python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --extra-index-url https://download.pytorch.org/whl/cu130

echo "=== Installing core packages ==="
python -m pip install \
  transformers==5.5.0 \
  datasets \
  accelerate \
  tokenizers \
  huggingface_hub \
  wandb \
  tqdm \
  numpy \
  safetensors \
  packaging \
  sentencepiece \
  einops

echo "=== Installing Triton ==="
python -m pip install triton

echo "=== Installing Transformer Engine ==="
python -m pip install --no-build-isolation "transformer-engine[pytorch]"

echo "=== Verifying installation ==="

python - <<'PY'
import torch

print("="*50)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import triton
print("Triton:", triton.__version__)

import transformer_engine.pytorch as te
print("Transformer Engine: OK")

print("="*50)
PY