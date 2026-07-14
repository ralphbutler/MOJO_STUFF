# 📚 Mojo GPU Curriculum

A numbered sequence of small Mojo programs that build from a single SIMD register
up to GPU kernels. Each file is self-contained and heavily commented; this doc
holds the **why** — the ideas you need to make sense of the code, not a restatement
of it.

**The arc:** one register → one core over an array → thousands of GPU threads → a full
training loop built by hand from those kernels (`04` on CPU, `04b` on the GPU) → your own
kernel running *inside MAX* as a custom op (`05`). The recurring experiment is *vector add*
then *matmul*, kept deliberately trivial so each step changes exactly one thing.

Once the numbered Mojo kernels make sense, the doc widens to the question they set
up — *when would you write a kernel at all, vs. reach for a tool that already has
them?* That's the **Beyond Mojo** section below, covering **MAX** (running models)
and **PyTorch** (training + a framework baseline).

## 🏃 Running

```bash
uv run mojo 00_simd_type.mojo      # any file; swap the name
```

GPU files (`02+`) require an accelerator; on this Mac that's Apple Metal.

## 🔢 The Programs

### `00_simd_type.mojo` — the SIMD type alone
Pokes at `SIMD[dtype, W]` with no arrays in sight, so the type feels concrete
before it's used at scale.

- **`SIMD[dtype, W]` is a fixed-width register vector**, not a dynamic list. `W`
  is a compile-time power of 2.
- **Construction is 1-or-N**: one value *splats* (broadcasts) to every lane;
  otherwise you must supply exactly `W` values. Two values for a 4-wide vector is
  a compile error.
- **One op hits all lanes at once** — that's the whole point of SIMD. Plus
  reductions (`reduce_add`), masks + `select` (branchless per-lane choice), and
  casts.
- **Tie-in:** a `Scalar[dtype]` *is* `SIMD[dtype, 1]`. A single number is just the
  degenerate vector.

**Run:** `uv run mojo 00_simd_type.mojo` → prints each operation's result; no
PASS/FAIL, just eyeball that the lane math matches the comments.

### `01_vecadd_unsafe.mojo` — SIMD over a real array (unsafe/raw)
One CPU core walks a million-element array, adding `W` floats per step.

- **`W = simd_width_of[dtype]()`** = register width ÷ element size, computed at
  compile time. On this Mac (128-bit NEON): `float32` → `W=4`, `float64` → `W=2`.
  The code adapts to the hardware instead of hardcoding a lane count.
- **`comptime`** = fixed at compile time. `W` must be `comptime` because it's a
  *type parameter* (`SIMD[dtype, W]`), and those are resolved when the code is
  built.
- **`alloc[T]` → `UnsafePointer`**: the compiler no longer tracks this memory.
  No bounds checks, no auto-free — hence the manual `.free()` and the
  `while i + W <= N` guard. The unsafety is *deliberate and visible*.
- **Why raw here:** raw pointers with `load`/`store` map straight to the hardware
  with zero overhead — ideal for a benchmark, and it's the exact model GPU
  kernels use (see `02`).

**Run:** `uv run mojo 01_vecadd_unsafe.mojo` → expect `RESULT: PASS`, `mismatches: 0`,
and a `CPU time` in ms. That time is the baseline `02` compares against.

### `01_vecadd_safe.mojo` — the same thing, idiomatic
The version you'd actually ship for CPU app code. Same math, safer memory.

- **`List[T]` owns the buffer**: bounds-checked access, freed automatically at end
  of scope. No `alloc`, no `.free()`.
- **`.unsafe_ptr()` is a localized escape hatch**: you still need a raw pointer for
  SIMD `load`/`store`, but only inside the hot loop, and the `List` still owns and
  frees the memory. Safe everywhere else.
- **The lesson of the pair (01_unsafe vs 01_safe):** Mojo is *safe by default, unsafe by
  explicit opt-in* (like Rust, unlike C). You never get raw memory by accident —
  you type `Unsafe`. Idiomatic high-performance code = safe container owns the
  memory, raw pointer only at the compute core.

**Run:** `uv run mojo 01_vecadd_safe.mojo` → same `RESULT: PASS` and a `CPU time`
within noise of `01_unsafe`. The safety costs nothing here — that's the point.

### `02_vecadd_gpu.mojo` — the same vector add, on the GPU
Parallelism moves from "4 lanes per instruction" to "thousands of threads at once."

- **One thread per element**, not one core over the array. `global_idx.x` gives
  each thread its element index; the launch rounds up to whole blocks, so the last
  block overhangs `N` — hence `if tid < size`.
- **The kernel is a plain function** — no CUDA-style decorators. `DeviceContext`
  allocates device buffers and launches it via `enqueue_function`.
- **`TileTensor` + `layout`** describe the buffer shape; `map_to_host()` moves data
  host↔device for fill and verify.
- **The point of the experiment:** the GPU is *not* expected to win here. Vector
  add is memory-bound (≈1 add per 2 loads, ~zero arithmetic intensity), so both
  backends are limited by memory bandwidth and the GPU also pays transfer cost.
  The GPU wins when each value is *reused* many times — which is matmul, coming
  in `03`.

**Run:** `uv run mojo 02_vecadd_gpu.mojo` → expect `RESULT: PASS` and a `GPU time`
that is *not* faster than `01_unsafe`'s CPU time. That "loss" is the intended result;
the closing note explains why, and `03` is where it flips.

### `03` — matmul: where the GPU finally wins
Matrix multiply is the payoff. Unlike vector add, each input value is reused `O(N)`
times, so there's real math to hide memory latency behind — high arithmetic
intensity. This is a *sub-sequence* (`03a` → `03e`) showing the same matmul
optimized in stages; the lever throughout is **arithmetic intensity** (math done
per memory access and per barrier). `03a`–`03d` are GPU; `03e` is the CPU counterpart.

- **`03a_matmul_naive.mojo` — naive.** One thread per output element `C[row,col]`. The
  simplest correct GPU matmul, no shared-memory machinery. Re-reads every A row and
  B column from slow global memory → memory-bandwidth-bound. The honest starting
  point.
- **`03b_matmul_tiled.mojo` — shared-memory tiling.** Each block cooperatively loads
  a `TILE×TILE` block of A and B into fast on-chip **shared memory** once, then every
  thread reuses it `TILE` times before the next load. Introduces `barrier()` to sync
  the cooperative load. Moves the kernel from bandwidth-bound toward compute-bound —
  this is the optimization Apple's MPS uses.
- **`03c_matmul_coarse.mojo` — register blocking / thread coarsening.** Each thread
  computes an `8×8` micro-tile of C, holding 64 partial sums in **registers**, doing
  ~512 MACs per barrier (~32× more work per sync than simple tiling). The last big
  lever before you're near hardware peak. Requires `N % BM == 0`, `N % BK == 0` (no
  ragged edges, for speed).
- **`03d_matmul_check.mojo` — correctness.** The benchmark's all-ones × all-twos fill
  makes every `C = 2N`, which *hides* index bugs. This uses small, non-symmetric
  inputs and checks the GPU result against a CPU triple-loop, element by element —
  so row/col or block/tile mix-ups actually show up.
- **`03e_matmul_cpu.mojo` — the CPU counterpart (no GPU needed).** Same `C = A @ B`,
  but for machines with no Mojo GPU backend (e.g. Aurora's Intel Max GPUs). The CPU's
  two levers replace shared memory + registers: **wide SIMD** (`W = simd_width_of[f32]`
  — 4 on NEON, 16 on AVX-512) and **many cores** (`parallelize` over rows). Runs the
  same three-stage ladder — naive → SIMD → parallel — reporting GFLOP/s and checking
  each stage against the naive reference. The naive→SIMD jump is the vector width; the
  SIMD→parallel jump is the core count.

**Run** (naive → tiled → coarse; edit `N` at the top to sweep sizes):
```
uv run mojo 03a_matmul_naive.mojo
uv run mojo 03b_matmul_tiled.mojo
uv run mojo 03c_matmul_coarse.mojo
uv run mojo 03d_matmul_check.mojo   # PASS/FAIL, run after changing the coarse kernel
uv run mojo 03e_matmul_cpu.mojo     # CPU ladder; GFLOP/s + PASS/FAIL, no GPU required
```
Expect each GPU stage faster than the last at `N=2048`, and — unlike `02` — the GPU
comfortably beating a CPU. Watch the ms/pass drop as arithmetic intensity climbs.
`03e` stands alone as the CPU story for GPU-less hardware.

### `04_train_mlp.mojo` — training a net in pure Mojo (pure-Mojo capstone)
The whole curriculum assembled: a 2-layer MLP (`2 → H → 1`) trained by hand to fit the
radius `r = √(x₀²+x₁²)`. No framework, no autograd — since MAX doesn't train and we're
not using Nabla here, **you are the autograd**: every gradient is derived on paper and
coded by hand.

- **Forward is `03`'s matmul** — `X@W1 → relu → a1@W2`. The kernel you optimized becomes
  a layer.
- **Backward is more matmul** — the gradients are transposed matmuls (`a1ᵀ@dz2`,
  `Xᵀ@dz1`, `dz2@W2ᵀ`) plus an elementwise relu-gradient. The same operation, run
  backward.
- **Kept naive** (triple loops, `N=256`): `03` already covered making matmul fast; here
  the lesson is the *training math*, not GFLOP/s. GPU-izing it = swapping in the `03`
  kernels.
- **Why it matters for the ecosystem question:** training with zero framework fragility —
  the honest counterpart to the (alpha, nightly-pinned) Nabla path. It shows exactly what
  MAX's missing training API costs you: you write the gradients yourself.

**Correctness — `train_mlp_reference.py`:** the identical setup in NumPy, seeded from the
same LCG so both start from bit-identical data and weights. Verified — the two loss curves
match to 6 figures (`2.178 → 0.000561` over 300 epochs); that agreement is the proof the
by-hand Mojo backprop is right.

**Run:** `uv run mojo 04_train_mlp.mojo` (and `python train_mlp_reference.py` for the NumPy
ground truth) → expect matching loss trajectories.

### `04b_train_mlp_gpu.mojo` — the same training, on the GPU
`04` moved onto the accelerator: every forward/backward step is a GPU kernel, and the
matmuls **reuse `03`'s one-thread-per-output design** — generalized from square `N×N` to
the rectangular and transposed shapes backprop needs (`A@B`, `Aᵀ@B`, `A@Bᵀ`).

- **Verified:** same LCG and math as `04`, so it prints the identical loss curve
  (`2.178 → 0.000561`, bit-for-bit here). Matching numbers = the GPU kernels — including
  the transposed-matmul gradients — are correct.
- **~15 kernel launches per epoch:** 2 matmuls forward; 3 matmuls + elementwise +
  column-sum reductions backward; 4 SGD updates.
- **The `02` lesson, measured.** Both files print `train time` (ms/epoch, warmup
  excluded). At the tiny default `N=256, H=16` the GPU is ~**30× slower** (0.7 vs 0.024
  ms/epoch) — ~15 kernel launches per epoch, almost pure overhead. Bump to `N=8192,
  H=1024` and it flips: GPU ~**15× faster** (7.8 vs 115 ms/epoch on the M4 Max), identical
  loss on both. That crossover is the whole arithmetic-intensity / parallelism story in one
  experiment. Structural takeaway: training — forward *and* backward — is just matmuls, and
  `03`'s kernel runs all of it.

**Run:** `uv run mojo 04b_train_mlp_gpu.mojo` → same loss trajectory as `04` and the NumPy
reference.

### `05_custom_max_op.py` (+ `custom_op_kernels/relu.mojo`) — the capstone: your Mojo op inside MAX
Where the whole journey converges. The `relu` you hand-wrote in `04`/`04b` is registered as
a **custom MAX operation** (`@compiler.register("relu")` in `custom_op_kernels/relu.mojo`)
and run as a node in a MAX graph. This is the concrete answer to *why Mojo is a language,
not just a demo*: when a shipped op doesn't fit, you write the kernel in Mojo and MAX
compiles and runs it — here on the Metal GPU.

- **Two halves.** `custom_op_kernels/relu.mojo` is the Mojo side — one `@compiler.register`
  struct whose `execute` method applies an elementwise rule (`max(x, 0)`) via `foreach`.
  `05_custom_max_op.py` is the driver — it builds a one-op graph with
  `ops.custom(name="relu", …)`, points MAX at the kernel dir with `custom_extensions=[…]`,
  compiles via `InferenceSession`, and executes.
- **Verified:** mixed-sign input → output is exactly `max(x, 0)` → `RESULT: PASS`, on
  `device: GPU (Metal)`.
- **The arc closed:** in `00`–`04b` you *wrote* kernels; in the MAX tier (below) you *ran*
  prebuilt models; `05` is where they meet — MAX running *your* Mojo kernel. That's the
  endgame of the "should we move to the Mojo ecosystem?" question: yes — you can extend the
  engine itself, in Mojo.

**Run:** `uv run python 05_custom_max_op.py` → prints input, `relu(x)`, expected, `PASS`.

## 🚀 Beyond Mojo: MAX and PyTorch

Once the Mojo kernels make sense, the real question is *when you'd write a kernel at
all vs. reach for a tool that already has them.* Three tiers, lowest to highest:

| Tier | Tool | You provide | It provides | Files |
|---|---|---|---|---|
| Write the kernel | **Mojo** | the GPU code | a language that compiles to Metal / CUDA / … | `00`–`04b` |
| Run a model | **MAX** | a model name + a prompt | prebuilt, optimized inference | `max_generate.sh`, `max_serve_litellm.sh` |
| Train a model | **PyTorch** | a model + data | autograd, optimizers, MPS kernels | `train_torch_mlp.py`, `bench_torch.py` |

The dividing lines *are* the "when Mojo vs when MAX" answer:

- **MAX is inference/deployment, not training — at least today.** The installed macOS
  build exposes graph building, `nn` inference layers, and an inference engine, but no
  autograd and no `.backward()`; and per Modular (mid-2026) training isn't first-class
  in MAX on *any* platform yet — Lattner: "not focused on solving training yet, maybe
  we'll get there." Hand MAX a trained model and it serves it fast; to *learn* weights
  you drop to PyTorch — or to **Nabla**, the Mojo-native autodiff layer built on MAX
  (see below). (Watch the Modular changelog — this is roadmap, not a law.)
- **Mojo is for when no shipped kernel fits** — a custom op, a new architecture, an
  accelerator the frameworks don't cover. Otherwise the framework's kernel (see
  `bench_torch.py`) is already near hardware peak and took you one line.

### 🚀 MAX — running prebuilt models (Metal, no Mojo)

Same engine, two front doors. Neither involves Mojo: MAX runs its own optimized
kernels on the Apple GPU; you pick a model and a prompt. Model defaults to
Qwen2.5-0.5B; the first run compiles it (~20–60s), then it's fast.

- **`max_generate.sh` — one-shot generation.** `max generate` loads the model, runs
  one prompt, prints the completion, exits. The *batch / offline* door: quick checks,
  scripted runs.
- **`max_serve_litellm.sh` — a serving endpoint.** `max serve` keeps the model
  resident behind an OpenAI-compatible HTTP endpoint; the script calls it with
  `litellm`, then shuts it down. The *production serving* door — the "one command,
  faster than vLLM" story, and the natural Day-1 hook for a class.

**Run:** `./max_generate.sh` or `./max_serve_litellm.sh` (each `cd`s to its own dir;
server logs land in `max_serve.log`).

### 🔬 PyTorch — training, and the framework yardstick

The tier MAX doesn't cover. Both run from an env that has `torch` (your base env),
**not** `MOJO_CURRICULUM/.venv` (which holds MAX). Both use MPS — the same Apple GPU the
Mojo kernels target, reached through a framework instead of hand-written code.

- **`train_torch_mlp.py` — the training demo.** A tiny MLP learns to separate two
  concentric rings of points. Its four-line loop (`zero_grad` → forward+loss →
  `backward` → `step`) is exactly what MAX has no API for. Nonlinear on purpose:
  delete the `ReLU` and accuracy collapses to ~50% — which is *why* the hidden layer
  exists. **Run:** `python train_torch_mlp.py` → expect final accuracy ~99%.
- **`bench_torch.py` — the matmul baseline for `03`.** Times PyTorch matmul on CPU and
  MPS at the same `N=2048` as the Mojo matmul files. This is the "someone already
  optimized this kernel" number your hand-written `03*` kernels are chasing — and the
  honest reason to reach for a framework unless you have a reason not to. **Run:**
  `python bench_torch.py` (or `python bench_torch.py 4096 50` to sweep sizes).

### 🔥 Nabla — training *inside* the Mojo ecosystem (alpha)

The most consequential stop on the journey, because it answers the question the whole
"could we move to Mojo?" project is asking. We hand-rolled training in `04`/`04b` to prove
it's *possible* in pure Mojo — but nobody wants to derive gradients by hand for a real
model. **Nabla** is the ecosystem's answer: a Mojo-native autodiff framework
(`grad`/`vmap`/`jit`, PyTorch-style modules) built *on* Mojo + MAX. It is precisely the
"custom Mojo kernels + automatic differentiation" convergence — what `04b` does by hand,
done for you, and what Chris Lattner points to when asked about training in the stack.

- **`train_nabla_mlp.py`** — the same radius-regression MLP as `04`, but Nabla's autograd
  runs the backward pass. This is what training *feels like* with a framework, entirely
  within the Mojo/MAX world — no PyTorch, no leaving the ecosystem.
- **Why it belongs next to our roll-your-own:** `04`/`04b` are what training *costs*
  without a framework; Nabla is what it *feels like* with one. Together they bracket the
  switch question. We keep the by-hand versions because they teach the mechanics and never
  break; Nabla shows where the ecosystem is *heading*.
- **The reality check — itself part of the lesson.** Nabla is **alpha** and pins to
  Modular **nightly**, so it gets its own venv (`setup_nabla_venv.sh` → `.venv-nabla`,
  isolated from the stable MAX in `.venv`). Getting it to run took version-pinning: its
  latest release ships an *unpinned* `modular` dependency, so a bare install pulls a
  too-new nightly and `backward()` dies with `ModuleOp has no attribute 'operation'`. The
  setup script pins `modular` back to Nabla's build era. That fragility is real data for
  "is the ecosystem ready?" — the *idea* is compelling, the *packaging* is young.

**Run:** `./setup_nabla_venv.sh` (once), then `.venv-nabla/bin/python train_nabla_mlp.py`
→ loss falls toward 0 on the same task as `04`.

## 🧭 Recurring Ideas

| Idea | Where it shows up |
|---|---|
| `comptime` = resolved at build time (needed for type params) | `W`, `N`, `dtype` everywhere |
| SIMD width adapts to hardware | `simd_width_of` in `01` |
| Safe by default, `Unsafe`/`.unsafe_ptr()` = explicit opt-in | `01_unsafe` vs `01_safe` |
| Same problem, different parallelism model | SIMD (`01`) vs threads (`02`) |
| Arithmetic intensity decides if the GPU wins | `02` note → `03` matmul |
| Write a kernel vs. use a shipped one | Mojo `03` vs PyTorch `bench_torch.py` |
| Inference vs. training is a tool boundary | MAX (serve/generate) vs PyTorch `train_torch_mlp.py` |
| Training = matmul forward + backward; you are the autograd | `04` (CPU), `04b` (GPU) |
| Same matmul kernel runs forward, backward, and both tiers | `03` → `04b` reuse |
| Training with a framework vs. by hand | Nabla vs `04`/`04b` |
