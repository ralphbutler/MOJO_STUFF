# 🚀 TurboQuant in Mojo — vs Rust `turbovec` and FAISS

A from-scratch **100% Mojo** reimplementation of Google's **TurboQuant** (data-oblivious
vector quantization), benchmarked head-to-head against Ryan Codrai's Rust
[`turbovec`](https://github.com/ryancodrai/turbovec) and Meta's
[FAISS](https://github.com/facebookresearch/faiss) — same data, same metric, same memory budget.

The goal was a talk showcasing Mojo as a serious language for this kind of
performance-critical numerical work: can a clean-room Mojo port match a mature,
BLAS-backed Rust crate?

## 🏆 Result

DBpedia OpenAI-1536 · 100k base / 1k queries · 4-bit = **768 B/vector (equal memory)** · Apple M4 Max, 16 cores, all engines parallel across queries.

| | TurboQuant (Mojo) | turbovec (Rust) | FAISS IndexPQ |
|---|---|---|---|
| recall@1 | 0.959 | 0.964 | 0.955 |
| recall@2 | 0.994 | 0.996 | 0.997 |
| recall@4+ | 1.000 | 1.000 | 1.000 |
| **index build** | **0.7 s** | 0.9 s | 16.1 s |
| **search** | 0.38 ms/q | 0.15 ms/q | 3.66 ms/q |
| bytes/vector | 768 | 768 | 768 |
| training | none | none | required |

**Takeaways:** Mojo ties on recall (all deltas ≤ 0.005 — noise on 1k queries),
**beats Rust on index build**, lands within 2.5× on search, and is ~10× faster
than FAISS on search. Both TurboQuant implementations are *data-oblivious* (fixed
rotation + fixed codebook, no training), which is why their builds crush FAISS's.

An interactive writeup with charts lives in [`talk/turboquant_results.html`](talk/turboquant_results.html)
(open in a browser; dark/light toggle top-right). Static exports:
[`talk/render_dark.png`](talk/render_dark.png), [`talk/render_light.png`](talk/render_light.png).

## 🧠 The algorithm

TurboQuant is *data-oblivious*: it never trains on the dataset. Encoding is:

1. **Normalize** each vector to unit length.
2. **Rotate** by a fixed random orthogonal matrix `Q` (a Householder-QR construction
   from a seeded PRNG). The rotation makes each coordinate of a unit vector follow a
   known Beta/Gaussian-limit distribution — so a *fixed* quantizer is near-optimal.
3. **Quantize** each rotated coordinate with a fixed Gaussian-limit Lloyd–Max
   codebook (validated against textbook levels).
4. **Bit-pack** the codes (4-bit → 2 codes/byte) and store a per-vector `scale`
   (RaBitQ-style, makes the self-score exact).

Search rotates the query, builds a per-coordinate lookup table
(`lut[d][c] = q_rot[d]·centroid[c]`), scores every vector as a sum of table
lookups, and keeps the top-k. The fast path uses a **FastScan** kernel: codes are
repacked into a SIMD-blocked layout and scored 16 vectors per instruction via NEON
table lookup (`SIMD._dynamic_shuffle`).

Correctness was cross-checked three ways: against a NumPy oracle
(`compare/reference_tq.py`), against FAISS, and against Ryan's Rust `turbovec`.

## ⚙️ How the Mojo caught up to Rust (and why the `BAK_V*` dirs exist)

The first working Mojo version (Phase A) already matched the algorithm and beat
FAISS, but was ~5× slower than Rust on search and ~8× on build. Reading the Rust
source pinpointed exactly why — both gaps were *implementation maturity, not the
language* — and both were fixed with standard techniques ported to readable Mojo:

- **`BAK_V1/`** — snapshot *before* the search fix. Baseline: search 0.71 ms/q, build 7.0 s.
- **`BAK_V2/`** — snapshot *before* the build fix. After search fix: 0.37 ms/q, build 7.0 s.
- Current — both fixes applied: search 0.38 ms/q, **build 0.7 s**.

These snapshots are kept so the talk can show the before/after of each optimization.

**Fix (b) — FastScan scan kernel (search 0.71 → 0.37 ms/q, 1.9×).**
The first cut cast each `uint8` table lookup to `uint32` and accumulated in `uint32`
*every byte* — 2–4× the SIMD register traffic it needed. FAISS and Rust cap the LUT
at ≤127 so both nibble sub-tables combine in `uint8`, then do **one** widening add
into a `uint16` accumulator, flushing to `uint32` before overflow. Porting that
trick (with a branch-free chunked flush) nearly doubled search throughput at a
negligible recall cost (7-bit LUT: 0.960 → 0.959).

**Fix (c) — rotation as a register-blocked GEMM (build 7.0 → 0.7 s, 7.6× on encode).**
The naive encode rotated one vector at a time as a matrix–vector product, which
re-streamed the entire 9.4 MB rotation matrix for all 100k vectors — memory-bound
at ~34 GFLOP/s. Rust batches the rotation through a real GEMM (`faer`/`ndarray`).
The Mojo fix is a ~20-line **register-blocked micro-kernel** (MR×NR tile of SIMD
accumulators) written *in-language* — no BLAS dependency — and it out-builds the Rust.

### ⚠️ Mojo 1.0 gotchas worth knowing (talk gold)

- **`comptime for`, not `@parameter for`** (the latter is deprecated). This was the
  *unlock* for the GEMM: the micro-kernel's accumulator indices must be
  compile-time so they live in registers. With a runtime index, `InlineArray[SIMD]`
  spills to the stack every FMA and you get **zero** speedup (we were stuck at 5.2 s
  until switching to `comptime for`).
- **A per-iteration branch in the hot loop kills vectorization.** An
  `if flush_now: …` inside the FastScan inner loop *regressed* search to 1.0 ms/q;
  the fix was a branch-free chunked loop with the flush hoisted between chunks.
- **`fn` is gone** — Mojo 1.0 unified on `def`. `alias` → `comptime`; stdlib under
  `std.` (`std.math`, `std.algorithm`, `std.memory`, `std.sys`, `std.time`).
  `InlineArray` is builtin (no import): `InlineArray[T, size](fill=x)`.
- **Never `alloc`/`free` inside `parallelize` workers** — it races nondeterministically.
  Pre-allocate one disjoint scratch slice per work-item; workers only index into it.
- **`rebind` to `MutUntrackedOrigin` severs lifetime tracking** — the compiler can
  free the backing `List` early and reuse the memory (silent NaN/garbage). Keep such
  Lists alive to the end of `main()`.

## 📁 Repository layout

```
turboquant_mojo/       The 100% Mojo implementation
  codebook.mojo          Gaussian-limit Lloyd–Max quantizer
  rotation.mojo          Seeded PRNG + Householder-QR random orthogonal matrix
  turboquant.mojo        Reference single-thread encode + search
  run_dbpedia.mojo       Scalar-SIMD search baseline
  run_dbpedia_fast.mojo  FastScan + register-blocked GEMM  ← the benchmarked build
  test_synth.mojo,
  test_real.mojo         Validation drivers

compare/               Python — comparison harness ONLY (not the implementation)
  fetch_data.py          Streams the dataset → raw .f32/.npy
  faiss_baseline.py      FAISS IndexPQ baseline (equal memory)
  turbovec_baseline.py   Ryan's Rust turbovec, via its Python bindings
  eval_mojo.py           Scores the Mojo output
  report.py              Prints the three-way head-to-head table
  reference_tq.py        NumPy oracle (algorithm cross-check)

talk/                  Presentation artifact (self-contained HTML + PNG exports)
data/                  Dataset + benchmark results (large files are gitignored)
BAK_V1/, BAK_V2/       Pre-optimization code snapshots (see above)
ryan_turbovec/         Cloned reference Rust repo (read for spec; gitignored)
RESUME1.md, RESUME2.md Detailed working notes for both sessions
```

## 🛠️ Setup

**1. Mojo.** This project targets **Mojo 1.0.0b2**. Install the Modular/Mojo
toolchain per the official instructions at <https://docs.modular.com/mojo/> and
confirm with `mojo --version`. (Mojo 1.0 is a beta; syntax differs from older
releases — see the gotchas above.)

**2. Python deps** (for the comparison harness only — the Mojo side needs none):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**3. Rust** (only for the `turbovec` comparison): a stable toolchain (`rustc`/`cargo`,
1.90 used here) via <https://rustup.rs>.

Hardware note: developed on Apple Silicon (M4 Max, NEON SIMD). The FastScan kernel
uses `SIMD._dynamic_shuffle` (maps to NEON `tbl` / x86 `pshufb`) and should build on
any Mojo-supported target, but the reported timings are Apple-Silicon-specific.

## ▶️ Reproduce

All commands are run **from the repository root**, except the Mojo binary, which
must be run from `turboquant_mojo/` (it reads the dataset via the relative path
`../data/`).

```bash
# 1. Fetch the dataset (writes data/*.f32, *.npy, meta.json, true_top1.i32)
python compare/fetch_data.py

# 2. FAISS baseline
python compare/faiss_baseline.py

# 3. Rust turbovec — build its Python bindings into your env, then run
#    (see RESUME2.md; VIRTUAL_ENV must be set for maturin to find the venv)
git clone https://github.com/ryancodrai/turbovec ryan_turbovec
cd ryan_turbovec/turbovec-python && VIRTUAL_ENV=$VIRTUAL_ENV maturin develop --release && cd ../..
python compare/turbovec_baseline.py

# 4. Mojo TurboQuant (FastScan) — build + run + score
cd turboquant_mojo && mojo build run_dbpedia_fast.mojo -o /tmp/run_fast && /tmp/run_fast && cd ..
python compare/eval_mojo.py

# 5. Head-to-head table
python compare/report.py
```

> Note: Mojo runs may emit a harmless `Failed to initialize Crashpad` line — filter with `grep -v Crashpad`.

## 📦 What's not in the repo (deleted to keep it small)

To keep the repository lean (it drops from ~1.6 GB to ~17 MB), a few large,
easily-regenerated files are removed and gitignored. Regenerate them as needed:

| Removed | Size | How to regenerate |
|---|---|---|
| `data/base.f32`, `data/base.npy` | ~1.2 GB | `python compare/fetch_data.py` (re-streams the dataset) |
| `ryan_turbovec/target/` | ~430 MB | `cd ryan_turbovec/turbovec-python && maturin develop --release` |
| `data/results/mojo_serial.i32` | — | `cd turboquant_mojo && mojo build run_dbpedia.mojo -o /tmp/rs && /tmp/rs` (unused intermediate) |

The small benchmark outputs (`data/results/*.json`), `data/meta.json`,
`data/queries.*`, and `data/true_top1.i32` are kept so the results are inspectable
without re-fetching the full dataset.

## 🔗 Links & credit

- **TurboQuant** — Google Research (data-oblivious vector quantization). Search
  *"TurboQuant Google Research arXiv"* for the paper.
- **turbovec** — Ryan Codrai's Rust implementation: https://github.com/ryancodrai/turbovec
- **FAISS** — Meta's vector-search library (baseline): https://github.com/facebookresearch/faiss
- **Qdrant** — vector DB that integrates the rotation trick natively: https://github.com/qdrant/qdrant

The TurboQuant algorithm is Google's; the Rust `turbovec` reference is Ryan Codrai's.
This repository is an independent Mojo reimplementation and benchmark.
