# TriuneTransformer 2.0 – Full Spec

## Hardware
- GPU: NVIDIA RTX 5070 Laptop (8 GB VRAM)
- CPU: Ryzen 9 8940HX
- RAM: 16 GB

## Software Stack
- PyTorch 2.12 + CUDA 12.8
- Optional: flash‑attn, transformer‑engine, unsloth, galore‑torch

## Features Enabled
| Feature | Status | How to toggle |
|---------|--------|---------------|
| Hybrid Jet‑Nemotron architecture | ✅ | `USE_HYBRID = True` in config.py |
| JetBlock linear attention | ✅ | Set `HYBRID_RATIO` |
| Q‑GaLore optimizer | ✅ | `USE_Q_GALORE = True` |
| FP8 training (TE) | ✅ | `USE_FP8 = True` (requires TE) |
| FlashAttention‑3 | ✅ | `USE_FLASH_ATTN = True` (requires flash‑attn) |
| Unsloth kernels | ⚠️ | `USE_UNSLOTH = True` (experimental on Windows) |
| DataFlex dynamic mixing | ⚠️ | `USE_DATAFLEX = True` (basic implementation) |
| Grouter routing | ⚠️ | Placeholder – to be refined |

## How to Run
1. Install dependencies:
   ```bash
   pip install torch datasets tokenizers bitsandbytes
   pip install flash-attn transformer-engine unsloth galore-torch  # optional