# 🔥 MOJO_STUFF

Experiments in **Mojo** and the **Modular (MAX)** stack on Apple Silicon — a
learning curriculum, a from-scratch agent, and notes on porting to HPC hardware.
Each subdirectory is self-contained and has its own README with details.

## 📚 Subdirectories

### `MOJO_CURRICULUM/`
A hands-on curriculum for GPU programming and the Modular stack on an M4 Max,
built as tutorial-paced, heavily-commented example programs. It follows one
operation (matrix multiply) across three tiers — **writing kernels by hand in
Mojo**, **serving models with MAX**, and **training with PyTorch** — to show who
writes the kernel and when that should be you. Start with `CURRICULUM.md`.

### `MOJO_HARNESS/`
A tiny LLM coding agent — a minimal Claude-Code — written **natively in Mojo**,
no Python and no third-party libraries. It streams a chat request to a local
OpenAI-compatible endpoint and runs shell tools the model asks for. Networking,
JSON, HTTP, and SSE streaming are all hand-written Mojo; the point is to show off
the language. macOS/Apple Silicon.

### `MOJO_POLARIS_AURORA/`
Working directory for getting the Mojo/MAX stack running on Argonne's HPC
machines — `POLARIS/` (NVIDIA A100, CUDA backend) and `AURORA/` (Intel Max GPU,
CPU-only). Prototype on the Mac, then push to the target boxes; the canonical
teaching source is `MOJO_CURRICULUM/`. Note its README's path warning if you
resume that work.

## 🚀 Running

Each subdirectory uses [uv](https://docs.astral.sh/uv/), which fetches the Mojo
toolchain automatically — you don't need to install Mojo separately. From a
subdirectory, run programs with `uv run mojo run <file>.mojo`. See each README
for specifics.
