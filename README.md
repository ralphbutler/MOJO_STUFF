# 🔥 MOJO_STUFF

Experiments in **Mojo** and the **Modular (MAX)** stack on Apple Silicon — a
learning curriculum, a from-scratch agent, a head-to-head performance benchmark
against Rust and C++, and notes on porting to HPC hardware.
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

### `MOJO_TURBOQUANT_TURBOVEC/`
A from-scratch **100% Mojo** reimplementation of Google's **TurboQuant** — a
*data-oblivious* vector quantizer for approximate nearest-neighbor search (fixed
random rotation + fixed Lloyd–Max codebook, so it needs no training pass over the
data). The project benchmarks it head-to-head against Ryan Codrai's Rust
[`turbovec`](https://github.com/ryancodrai/turbovec) and Meta's **FAISS** on the
same dataset, metric, and memory budget (DBpedia OpenAI-1536, 100k vectors,
768 B/vector), to answer whether a clean-room Mojo port can match a mature,
BLAS-backed Rust crate. It can: equal recall (0.959 vs 0.964 @1), **faster index
build than Rust** (0.7 s vs 0.9 s) and ~23× faster than FAISS, with search within
2.5× of Rust and ~10× faster than FAISS.

Beyond the numbers, it's a record of *how* the Mojo caught up — a FastScan SIMD
scan kernel and a register-blocked GEMM micro-kernel written in-language with no
BLAS dependency — with pre-optimization snapshots (`BAK_V1/`, `BAK_V2/`) kept so
the before/after of each fix is inspectable, plus a list of Mojo 1.0 gotchas found
the hard way. Includes a self-contained HTML writeup in `talk/`.

### `MOJO_WORLD_MODELS/`
A **world model** over a hide-and-seek gridworld: learn the transition function
from traces, then discard the real simulator and run agents inside the learned
one — with both compute-bound halves (simulator, dream loop) written in **Mojo**.
The deliverable is a measurement of the small-sequential-branchy regime, where
BLAS has no edge: the Mojo sim is byte-exact with the Python reference and
**300× faster** on 12 threads, and the Mojo dream loop beats PyTorch/MPS below
~32 concurrent rollouts. It also measures what the dream *costs* — a controller
evolved in the dream transfers only 2.4% of the gain of one evolved in reality,
at 98× the cost. Includes a side-by-side browser viewer of reality vs. dream.

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

Exception: `MOJO_TURBOQUANT_TURBOVEC/` expects an installed Mojo 1.0.0b2 toolchain
(`mojo build …`), plus a Python venv for the comparison harness and Rust only if
you want the `turbovec` baseline — see its README.
