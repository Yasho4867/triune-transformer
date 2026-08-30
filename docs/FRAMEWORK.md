# Triune Ecosystem Framework Guide

## Architecture Overview

Triune is an **All-in-One AI Research Suite** featuring a standalone core Python framework (`triune`), unified CLI, embedded WebUI API server (`triune.api`), VRAM Memory Planner (`triune.runtime.MemoryPlanner`), sandboxed code execution (`triune.runtime.PythonSandbox`), and visual node plugin protocol (`triune.plugins`).

```
triune/
├── __init__.py          # Main Exports (load_model, register_model, Trainer, MemoryPlanner)
├── model/               # Universal Model Base, Native MoE/Limbic, Zoo Adapters
├── trainer/             # High-level Trainer, Engine, Checkpointer
├── runtime/             # VRAM Memory Planner, Python Sandbox, Telemetry
├── optim/               # CentroidSteerOptimizer, GaLore, 8-bit AdamW
├── data/                # Tokenizers, Streaming Datasets, Parquet/JSONL Loaders
├── inference/           # Generation, KV-Cache Management, Samplers
├── recipes/             # BF16, FP8 E4M3/Hybrid, NVFP4 Precision Recipes
├── agents/              # Lightweight Tool-Use & Multi-Agent Engine
├── callbacks/           # Custom Training, Logging, and Telemetry Hooks
├── export/              # SafeTensors, GGUF, and ONNX Exporters
├── api/                 # Embedded FastAPI & WebSockets (OpenAI Spec & Telemetry)
└── plugins/             # Node-based Visual Pipeline Schema & Custom Node Registry
```

## Public API & Component Map

| Component | Public API | Responsibility |
| --- | --- | --- |
| Model Zoo | `triune.load_model("triune-base")`, `register_model` | Loads native, HF, or user custom models. |
| VRAM Planner | `triune.MemoryPlanner.estimate_vram(config)` | Auto-budgets VRAM for RTX 5070 / target GPUs. |
| Sandboxed Code | `triune.PythonSandbox` | Safely executes custom loss ops, nodes, & agent tools. |
| Custom Nodes | `triune.register_node(name)` | Registers custom visual nodes into Studio & CLI. |
| Agent Engine | `triune.Agent`, `MultiAgentOrchestrator` | Multi-agent execution and tool calling. |
| Exporters | `triune.export_safetensors`, `export_gguf`, `export_onnx` | One-command model weight exports. |
| Precision | `triune.build_fp8_precision_context`, `bf16_autocast` | FP8 (E4M3/HYBRID), BF16, or NVFP4 context. |
| Embedded API | `triune.api.run_server(host, port)` | Powers Triune Studio (OpenAI spec `/v1/chat/completions`). |

## CLI Commands

```bash
# Launch embedded API server for Triune Studio
triune serve --port 8000

# Estimate VRAM budget for RTX 5070 Laptop GPU (8GB / 12GB)
triune plan-memory --vram-gb 8.0

# Terminal interactive chat
triune chat --model triune-base

# Execute framework test suite
wsl /home/yasho4867/venvs/triune/bin/python tests/test_framework.py
```

## Programmatic Usage Example

```python
import torch
import triune

# 1. Estimate VRAM Memory Plan for Laptop GPU
config = triune.build_config({})
plan = triune.MemoryPlanner.estimate_vram(config, target_vram_gb=8.0)
print("Recommended Batch Size:", plan.recommended_batch_size)

# 2. Load Model via Model Zoo API
model = triune.load_model("triune-base").cuda()

# 3. Register Custom Node for Studio & CLI
@triune.register_node("Custom Preprocessor", category="data")
def custom_preprocessor(text: str) -> str:
    return text.strip().lower()

# 4. Safe Code Execution inside Sandbox
sandbox = triune.PythonSandbox()
result = sandbox.execute_code("y = x * 2", locals_dict={"x": 10})
print(result["y"])  # 20
```
