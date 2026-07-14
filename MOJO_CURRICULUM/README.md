# 🧭 MOJO_CURRICULUM — GPU compute on Apple Silicon, three ways

A hands-on curriculum for learning GPU programming and the Modular stack (Mojo + MAX)
on an M4 Max, built as tutorial-paced, heavily-commented example programs. Start with
**CURRICULUM.md** for the per-file walkthrough; this README is the map and the story
that ties the pieces together.

## 🧵 One operation, three tiers

The whole directory circles a single question: **when the GPU does matrix math, who
writes the kernel — and when should that be you?** The same matrix multiply is the
thread running through every tier:

| Tier | Tool | You provide | Who writes the kernel | Files |
|---|---|---|---|---|
| Write the kernel | **Mojo** | the GPU code itself | **you** | `00`–`03` |
| Run a model | **MAX** | a model + a prompt | Modular (prebuilt) | `max_generate.sh`, `max_serve_litellm.sh` |
| Train a model | **PyTorch** | a model + data | the framework | `train_torch_mlp.py`, `bench_torch.py` |

Read top to bottom, it's a progression in *how much the machinery does for you*:

- **Mojo** — you write the matmul yourself and optimize it by hand (naive → tiled →
  register-blocked, in `03`). This is the transferable HPC skill and the substance of
  the course.
- **MAX** — you write nothing. `max serve` / `max generate` run Modular's optimized
  kernels to do *inference* on a prebuilt model. The deployment tool.
- **PyTorch** — you define a model and call `.backward()`; the framework's kernels (the
  same matmul, hand-tuned by others — measured in `bench_torch.py`) *train* it. The
  training tool MAX doesn't provide (see the note below).

## 🔀 Where the tiers meet

They aren't three separate worlds. The bridge is **custom MAX ops written in Mojo**:
when a shipped MAX kernel doesn't fit your model, you write that kernel in Mojo and MAX
runs it. That is *why Mojo is a language and not just a demo* — the hand-written-kernel
skill from `00`–`03` is exactly what extends the inference engine of the MAX tier. The
matmul you optimize by hand in `03`, the framework calls in `bench_torch.py`, and the
engine runs when serving Qwen in `max_serve_litellm.sh` is, underneath, the same
operation. The three tiers are three answers to "who writes it, and when is that you?"

**This bridge is runnable — see `05_custom_max_op.py`.** The `relu` from `04`/`04b` is
registered as a custom MAX op (`custom_op_kernels/relu.mojo`, `@compiler.register`) and
executed as a node in a MAX graph, verified on the Metal GPU. That's the capstone: MAX
running *your* Mojo kernel — the endgame of the whole "move to the Mojo ecosystem?" question.

**Nabla makes that bridge runnable.** It drops custom Mojo kernels into an autodiff
engine built on Mojo + MAX, so hand-written kernels and training become one workflow —
`train_nabla_mlp.py` (in its own venv via `setup_nabla_venv.sh`). That's the Mojo-native
training path this project is evaluating for a possible full switch to the ecosystem.

The **framework-free** version of the same idea is `04_train_mlp.mojo` — training written
entirely by hand in Mojo (no autodiff, no dependency), verified to 6 figures against a
NumPy reference. Together they bracket the question: `04` is what training *costs* without
a framework, Nabla is what it *feels like* with one.

> **⚠️ On training in MAX.** MAX itself is inference-focused; native autodiff/training is
> **not** a first-class MAX feature on *any* platform as of July 2026 — not a mac-silicon
> gap (Chris Lattner: *"We're not focused on solving training yet. Maybe we'll get
> there."*). The ecosystem's answer is **Nabla**, a Mojo-native autodiff framework built
> *on* Mojo + MAX. It's **alpha** and pins to Modular nightly, so it lives in its own
> venv and PyTorch stays the stable training tier here. Track the Modular forum/changelog
> for MAX-native training landing.

## 📂 What's here

**Mojo curriculum (numbered — full walkthrough in `CURRICULUM.md`):**
`00_simd_type.mojo` … `04b_train_mlp_gpu.mojo` — SIMD register → CPU array → GPU threads →
matmul → a by-hand training loop (CPU `04`, then GPU `04b` reusing the `03` kernel), one
idea per step. Capped by `05_custom_max_op.py` (+ `custom_op_kernels/relu.mojo`) — your Mojo
`relu` op running *inside* MAX.

**MAX (running models on the Apple GPU, no Mojo):**
`max_generate.sh` (one-shot generation) · `max_serve_litellm.sh` (serving endpoint +
`litellm` client).

**Training — three ways:** by hand in pure Mojo (`04_train_mlp.mojo`, verified against
`train_mlp_reference.py`), in **PyTorch** (`train_torch_mlp.py`, stable), and in **Nabla**
(`train_nabla_mlp.py` + `setup_nabla_venv.sh`, Mojo-native but alpha). Plus `bench_torch.py`
(matmul CPU-vs-MPS baseline for `03*`).

**Notes / background:**
`CURRICULUM.md` (the per-file "why") · `MLIR_MOJO_MAX.md` (how MLIR, Mojo, and MAX
relate) · `MAX_VS_MOJO_GETTING_STARTED.md` (teaching notes: where to run it) ·
`RMB_SIMD_NOTES.txt` (SIMD mental model for 00/01) · `RESULTS01.md` (recorded results).

## 🏃 Running

```bash
uv run mojo 00_simd_type.mojo     # Mojo files (00–04b); GPU files (02+, 04b) need Metal
uv run mojo 04b_train_mlp_gpu.mojo  # Mojo: MLP training on the GPU (reuses the 03 kernel)
uv run python 05_custom_max_op.py   # capstone: custom Mojo relu op running inside MAX (Metal)
./max_generate.sh                 # MAX: one-shot generation
./max_serve_litellm.sh            # MAX: serving endpoint + litellm client
python train_torch_mlp.py         # PyTorch: run from your torch env, NOT MOJO_CURRICULUM/.venv
python bench_torch.py             # PyTorch: matmul CPU vs MPS
./setup_nabla_venv.sh             # Nabla: one-time isolated nightly venv (.venv-nabla)
.venv-nabla/bin/python train_nabla_mlp.py   # Nabla: same MLP loop, Mojo-native (alpha)
```
