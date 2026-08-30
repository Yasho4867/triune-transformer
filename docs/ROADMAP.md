# Triune Ecosystem Roadmap

## Phase 1 – Framework Engine (Current Focus)
- ✅ Refactor core `triune` package into 11 modular components (`model`, `trainer`, `runtime`, `optim`, `data`, `inference`, `recipes`, `agents`, `callbacks`, `export`, `api`, `plugins`).
- ✅ Implement VRAM Memory Planner (`triune.runtime.MemoryPlanner`) for RTX 5070 Laptop GPUs.
- ✅ Implement `PythonSandbox` for secure code, custom node, and agent execution.
- ✅ Implement FP8 (E4M3/HYBRID) precision context recipe with PyTorch native fallback.
- ✅ Implement `load_model` API and Model Zoo adapters.
- ✅ Implement Embedded FastAPI & WebSocket telemetry server (`triune.api`) powering Studio.
- ✅ Implement Node-based plugin schema & custom node registry (`triune.plugins`).
- ✅ Implement model exporters (`export_safetensors`, `export_gguf`, `export_onnx`).

## Phase 2 – Native Triune Models & Benchmarks
- 🔲 Pre-train native **Triune 2.5B MoE** baseline checkpoint on FineWeb-Edu.
- 🔲 Benchmark inference throughput against open models (Llama 3, Qwen 2.5, Smollm) with `Reflex` and `Limbic` dynamic depth exit heads.
- 🔲 Publish checkpoints to Hugging Face Hub under unified `load_model("triune-base")` API.

## Phase 3 – Triune Studio (Visual GUI / IDE)
- 🔲 Ship self-contained Triune Studio desktop installer.
- 🔲 Interactive Chat & Playground with multi-model side-by-side comparison.
- 🔲 Real-time Training Dashboard (Loss curves, router exit head distributions, VRAM memory timeline).
- 🔲 Node-based Visual Graph Builder for custom data pipelines, model building, and agentic workflows.
