# 🔥 MOJO_STUFF

Experiments in **Mojo** and the **Modular (MAX)** stack on Apple Silicon — a
learning curriculum, a from-scratch agent, a head-to-head performance benchmark
against Rust and C++, and notes on porting to HPC hardware.
Each subdirectory is self-contained and has its own README with details.

## ⚠️ Two Mojo versions in here — check before copying code between projects

Mojo changed substantially between `1.0.0b2` and `1.0.0`, and **b2 code will not compile on
1.0.0**. Only `MOJO_TERNARY` is on the release:

| project | Mojo |
|---|---|
| `MOJO_TERNARY` | **`1.0.0`** |
| `MOJO_CURRICULUM` | `1.0.0b2` |
| `MOJO_HARNESS` | `1.0.0b2` |
| `MOJO_WORLD_MODELS` | `1.0.0b2` |
| `MOJO_POLARIS_AURORA` | `1.0.0b2` |
| `MOJO_TURBOQUANT_TURBOVEC` | `1.0.0b2` (installed toolchain, not uv) |

The differences that actually bite, all found by compiling:

| 1.0.0b2 | 1.0.0 |
|---|---|
| `from std.algorithm import parallelize` | `from max.algorithm import parallelize` — left the stdlib; needs the `max` package |
| `from std.gpu.host import DeviceContext` | `from max.gpu.host import DeviceContext` — host side moved to `max` |
| `from std.gpu.sync import barrier` | `from max.gpu import barrier` |
| `from std.gpu.memory import AddressSpace` | `AddressSpace` is in the prelude |
| `UnsafePointer[T, origin]` | `Pointer[T, origin]` |
| `ptr[i]` | `ptr[unsafe_offset=i]` |
| `ptr.load[width=W](i)` / `.store(i, v)` | `.unsafe_load[width=W](i)` / `.unsafe_store(i, v)` |
| `def __del__(deinit self)` | `def __deinit__(deinit self)` |
| `ptr.free()` | `dealloc(allocation^)` — hold an `Allocation`, not a leaked pointer |
| `alloc[T](count)` | `alloc(Layout[T](count=n))` → `Allocation[T]` |
| `read` (argument convention) | `imm` |
| `InlineArray[T, N]` | `Array[T, N]` |

Device-side GPU symbols (`global_idx`, `thread_idx`, `block_idx`) stayed in `std.gpu` in both.
Two 1.0.0 traps worth knowing: **scalar GPU kernel arguments must be fixed-width** (`Int32` — an
`Int` fails with *"does not conform to DevicePassable"*), and **a `Pointer[T, MutUntrackedOrigin]`
struct field is a silent use-after-free**, because an untracked origin does not extend the owner's
lifetime (reported as `modular/skills#11`; hold an `Allocation` instead).

So the b2 projects are good reference for **ideas and structure**, not copy-paste source. When a
signature is in doubt, ask the compiler: a deliberate arity error makes it print every overload,
e.g. `alloc[Scalar[DType.int8]](64, 1, 2, 3)`.

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

### `MOJO_TERNARY/`
A from-scratch inference engine for **ternary (1.58-bit)** LLMs — weights that are only
−1/0/+1 — reading PrismML's **Ternary-Bonsai** models straight from GGUF. It loads the 1.7B and
8B, runs the full transformer graph, and **reproduces HuggingFace transformers' logits exactly**;
correctness is checked against an external oracle rather than self-consistency. Decode runs at
34.7 tok/s (1.7B) and 12.9 tok/s (8B) on the CPU with hand-written NEON `sdot` kernels, roughly
2× slower than llama.cpp's CPU path and 10× slower than its Metal path — **the performance goal
is not met**, and `STATUS_CURR.md` says so plainly along with everything measured on the way.

The interesting content is the measurements rather than the speed: that `parallelize` costs
~300 µs per dispatch, so *small* GEMVs must run single-threaded; that the ternary win is memory
bandwidth and not skipped arithmetic (the branch-per-weight "skip the zeros" version is 25×
slower); that this GPU has no bandwidth advantage over the CPU, so three GPU kernel shapes all
failed to beat it; and that packed 2-bit weights trade a bandwidth limit for a compute limit and
gain only 9%. Two upstream bugs came out of it, including one that makes llama.cpp refuse the
model file PrismML's own card recommends.

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

`MOJO_TERNARY/` pins Mojo `1.0.0` and needs the `max` package as well (that is where
`parallelize` lives now), so use `uv sync` then `.venv/bin/mojo` there — see its README.

Exception: `MOJO_TURBOQUANT_TURBOVEC/` expects an installed Mojo 1.0.0b2 toolchain
(`mojo build …`), plus a Python venv for the comparison harness and Rust only if
you want the `turbovec` baseline — see its README.

Large directories are not committed anywhere in this repo — `.venv/`, `.venv-nabla/`, `vendor/`,
and training data. Each project's README says how to recreate its own.
