# 🧪 Experiment 01 — Matmul: CPU vs PyTorch-MPS vs Mojo

**Date:** 2026-07-03
**Machine:** Apple M4 Max, 128 GB unified memory, macOS 26.5.1
**Problem:** square matmul, N = 1024, float32, 50 timed iters (5 warmup), throughput = 2·N³ / time

## 🧵 Which tier is this?

This races hand-written **Mojo** kernels (`03a`–`03d`) against the tuned **PyTorch**
library kernel (`bench_torch.py`) on the same Apple GPU — the top two tiers in the
[README](README.md). The third tier, **MAX**, has no row because it's an inference
engine, not a matmul you can call; read the PyTorch-MPS column as a stand-in for the
kind of tuned kernel MAX runs under the hood.

## 📊 Results (N = 1024)

| Implementation | Time | Throughput |
|---|---|---|
| PyTorch MPS (Apple GPU, tuned) | 0.376 ms | **5,711 GFLOP/s** |
| PyTorch CPU (Accelerate BLAS) | 0.852 ms | 2,520 GFLOP/s |
| Mojo (Apple GPU, naive kernel) | 0.948 ms | 2,264 GFLOP/s |

Sanity check on the Mojo kernel: `C[0,0] = 2048.0` (expected 2·N = 2048), correct.

## 📊 Results (N = 2048)

| Implementation | Time | Throughput | vs its own N=1024 |
|---|---|---|---|
| PyTorch MPS (Apple GPU, tuned) | 1.284 ms | **13,381 GFLOP/s** | 2.3× faster |
| PyTorch CPU (Accelerate BLAS) | 5.412 ms | 3,174 GFLOP/s | 1.3× faster |
| Mojo (Apple GPU, naive kernel) | 6.401 ms | 2,684 GFLOP/s | 1.2× faster |

Mojo sanity check: `C[0,0] = 4096.0` (expected 2·N), correct.

**Scaling is the story.** Going 1024 → 2048:
- **MPS scaled hard** (2.3×) — bigger matrices give the GPU more parallelism to fill,
  and tiling keeps it compute-bound. Now **5× faster** than naive Mojo (gap widened
  from 2.5× at N=1024).
- **Naive Mojo barely moved** (1.2×) and **still loses to CPU BLAS**. With no data
  reuse, every output element re-reads a full row + column from global memory, so the
  kernel is **memory-bandwidth-bound, not compute-bound** — extra parallelism doesn't
  help much. This is exactly the wall that tiling removes.

## 🧠 The key insight: memory reuse

The prediction going in was "naive Mojo will overtake CPU at larger N." **It didn't** —
at N=2048 naive Mojo (2,684) still trails CPU BLAS (3,174). The reason *is* the lesson:

- The naive kernel has **zero data reuse**. Each of the N² output elements independently
  reads a full row of A and a full column of B straight from **global GPU memory**. So
  it performs ~N³ *memory reads*, not just N³ math → it is **memory-bandwidth-bound**,
  not compute-bound. Giving it a bigger matrix barely helps, because the bottleneck is
  memory traffic. A GPU with no reuse can't even beat a cache-tuned CPU.
- **MPS tiles.** It loads a block of A and B into fast on-chip **shared memory once**,
  then reuses each value many times — turning the problem from bandwidth-bound into
  compute-bound. That is why MPS scaled 2.3× while naive scaled 1.2×, and why the MPS
  lead grew from 2.5× (N=1024) to 5× (N=2048).

So the experiment succeeded — not by making naive Mojo win, but by **measuring exactly
why it can't**, and pointing straight at the fix (tiling / shared memory).

## 🔁 Tiling experiment — the twist (N = 2048)

| Mojo kernel | Time | Throughput | vs naive |
|---|---|---|---|
| naive | 6.38 ms | 2,693 GFLOP/s | 1.00× |
| tiled (16×16 shared memory) | 8.01 ms | 2,145 GFLOP/s | **0.80× (slower!)** |

Both numerically correct (`C[0,0] = 4096`). **Basic tiling made it slower** — the
opposite of the prediction above. That prediction was NVIDIA folklore; Apple Silicon
behaves differently. Likely reasons (hypotheses, not yet profiled):

- **Apple GPUs have large caches + unified memory.** The naive kernel's repeated A-row /
  B-column reads are already largely served from cache — it gets much of the reuse
  benefit "for free." Manual 16×16 tiling then adds threadgroup-**barrier** and
  shared-load overhead *without* cutting real DRAM traffic enough to pay for itself.
- **Low compute-per-sync.** 2 barriers × 128 K-tiles = 256 barriers, with only 16 MACs
  of work between them. Basic tiling has low arithmetic intensity.
- **The real Apple lever is work-per-thread**, not just staging data in shared memory.
  To beat naive you need **thread coarsening / register blocking** — each thread
  computes a micro-tile of outputs (e.g. 4×4 or 8×8) with larger tiles — which is
  effectively what MPS does to reach 13 TFLOP/s.

**Lesson: "tiling always wins" is NVIDIA lore.** On Apple Silicon a simple shared-memory
tile is not automatically faster than a cache-friendly naive kernel; you must raise
arithmetic intensity (work per thread), not merely relocate data.

## ✅ Register blocking — the fix (N = 2048 leaderboard)

Thread coarsening confirmed the theory: each thread computes an 8×8 micro-tile
(64 register accumulators), doing TM·TN·BK = 512 MACs per barrier (~32× the simple
tile's 16). That raised arithmetic intensity enough to finally pull ahead.

| Implementation | Time | GFLOP/s | vs naive |
|---|---|---|---|
| PyTorch MPS (Apple GPU, tuned) | 1.284 ms | **13,381** | 4.97× |
| **Mojo coarse (register-blocked, 128×128 / 8×8)** | 3.968 ms | **4,330** | **1.61×** |
| PyTorch CPU (Accelerate BLAS) | 5.412 ms | 3,174 | 1.18× |
| Mojo naive | 6.379 ms | 2,693 | 1.00× |
| Mojo simple-tiled (16×16) | 8.010 ms | 2,145 | 0.80× |

The register-blocked kernel is the **first Mojo version to beat CPU BLAS** and is 2.0×
the simple-tiled version — driven entirely by arithmetic intensity, not by "using shared
memory." It still trails MPS by ~3.1×; that remaining gap is Apple's tuned tricks
(SIMD-group matrix instructions, vectorized `float4` loads, double-buffered slabs).

**Correctness: VERIFIED.** `03d_matmul_check.mojo` runs the coarse kernel on non-constant,
asymmetric inputs (N=256, 2×2 block grid, 32 K-slabs) and compares every one of the
65,536 outputs against a CPU reference triple-loop → **0 mismatches, max error 0.0**
(bit-identical: the kernel sums k in the same order as the reference). The
all-ones × all-twos benchmark fill could not have caught an indexing bug; this does.

## 🔍 What we learned

- **Getting onto the GPU was trivial.** ~15 lines of a naive one-thread-per-output
  kernel got 2.26 TFLOP/s on the Apple GPU — but it finished **last**, behind both
  tuned libraries.
- **Tuned beats naive on the same hardware**, and the gap *grows* with size — MPS goes
  from 2.5× (N=1024) to 5× (N=2048) faster than naive Mojo, purely from shared-memory
  tiling.
- **An untuned GPU kernel can lose to a tuned CPU.** Apple's Accelerate BLAS beat the
  naive GPU kernel at *both* sizes, because the naive kernel is bandwidth-bound while
  BLAS is cache-optimized. Hardware doesn't win — reuse does.

## ⚠️ Caveats

- All three times are sub-millisecond, so launch + timing overhead is a real fraction
  of each measurement. This size **flatters the CPU**; the GPU's parallelism advantage
  only dominates once compute outweighs overhead.
- Re-run at **N = 2048 / 4096** to let the GPU numbers pull away from CPU and to widen
  the MPS-vs-naive gap. (Mojo: edit `comptime N` in `03a_matmul_naive.mojo`;
  Torch: `python bench_torch.py 2048`.)

## ➡️ Next steps

1. ~~Scale up N (2048)~~ — **done above.** Confirmed naive is bandwidth-bound; the
   MPS gap widened. (N=4096 optional — would only reconfirm the trend.)
2. ~~Tile the Mojo kernel (`03b_matmul_tiled.mojo`)~~ — **done, and it was slower**
   (see the twist above). Basic shared-memory tiling loses to the cache-friendly naive
   kernel on Apple Silicon.
3. ~~Thread-coarsened / register-blocked kernel~~ — **done: 4,330 GFLOP/s, 1.61× over
   naive** (`03c_matmul_coarse.mojo`). Arithmetic intensity was the real lever.
4. ~~Verify coarse-kernel correctness~~ — **done: PASS, 0/65536 mismatches**
   (`03d_matmul_check.mojo`).
5. **Close on MPS (optional, advanced):** vectorized `float4` loads, SIMD-group matrix
   instructions, double-buffered K-slabs. Diminishing returns for learning; high effort.

## 🔁 Re-measured on Mojo 1.0.0 / MAX 26.5.0 (2026-09-09)

The whole curriculum was migrated from Mojo 1.0.0b2 to 1.0.0 on 2026-09-09
(see `UPDATE_TO_100.md`). Same machine, same `N = 2048`, 50 timed iters. The b2 numbers
above are kept; these are the new ones beside them.

| Implementation | b2 time | **1.0.0 time** | b2 GFLOP/s | **1.0.0 GFLOP/s** |
|---|---|---|---|---|
| PyTorch MPS (Apple GPU, tuned) | 1.284 ms | 1.322 ms | 13,381 | 12,995 |
| Mojo coarse (register-blocked, 128×128 / 8×8) | 3.968 ms | **3.99 ms** | 4,330 | **4,303** |
| PyTorch CPU (Accelerate BLAS) | 5.412 ms | 5.115 ms | 3,174 | 3,359 |
| Mojo naive | 6.379 ms | 6.43 ms | 2,693 | 2,670 |
| Mojo simple-tiled (16×16) | 8.010 ms | 7.85 ms | 2,145 | 2,188 |

**Every row is within run-to-run noise of its b2 value, and the leaderboard order is
unchanged.** `03d_matmul_check.mojo` still reports 0/65536 mismatches, max error 0.0.

### ⚠️ But the coarse kernel only survived after a fix

The first 1.0.0 run of `03c` came in at **6.9 ms / 2,490 GFLOP/s** — a 1.7× regression
that erased the entire register-blocking win and put the coarse kernel level with naive.
PyTorch's numbers were unchanged on the same machine, so it was not thermal drift or a
machine change.

The cause: the per-thread accumulators. `03c`/`03d` held them in
`InlineArray[Scalar[dtype], TM*TN]`, which under 1.0.0b2 stayed in registers. On 1.0.0
the Metal backend **spills that array to memory** — and a register-blocked kernel whose
registers are in memory is just a naive kernel with extra steps. Renaming it to 1.0.0's
`Array` made no difference; the container was the problem, not the spelling.

**Fix:** hold the tiles in a `SIMD[dtype, TM*TN]` instead — one register-resident value
of the same length, indexed identically by the `comptime` loops. That restored
**3.99 ms / 4,303 GFLOP/s**, matching b2 exactly. Both files now use `SIMD` tiles.

The lesson generalizes past this toolchain bump, and is worth teaching: *"these live in
registers" is a property of the generated code, not of your intent.* The only way to know
is to time it — and to keep a recorded number like this one to time it against.

### Training-loop timings (`04` vs `04b`, ms/epoch, warmup excluded)

| size | CPU `04` | GPU `04b` | verdict |
|---|---|---|---|
| `N=256, H=16` (default) | 0.027 ms | 0.27 ms | GPU **10× slower** (was ~30× on b2 — launch overhead fell) |
| `N=8192, H=1024` | 117 ms | 7.5 ms | GPU **15× faster** (unchanged from b2) |

Both files still print the identical loss curve at the default size
(`2.1783555 → 0.00056147523`), bit-for-bit, which is how the GPU kernels are known correct.

## 🗂️ Files

- `03a_matmul_naive.mojo` — Mojo **naive** matmul on the Apple GPU (run: `uv run mojo 03a_matmul_naive.mojo`)
- `03b_matmul_tiled.mojo` — Mojo **simple-tiled / shared-memory** matmul (run: `uv run mojo 03b_matmul_tiled.mojo`)
- `03c_matmul_coarse.mojo` — Mojo **register-blocked** matmul, the fast one (run: `uv run mojo 03c_matmul_coarse.mojo`)
- `03d_matmul_check.mojo` — correctness check for the coarse kernel vs a CPU reference (run: `uv run mojo 03d_matmul_check.mojo`)
- `bench_torch.py` — CPU + MPS matmul via PyTorch (run from an env with torch:
  `python bench_torch.py [N] [iters]`)

**Other tiers (context, not benchmarked here — see [README](README.md)):**

- `max_generate.sh` / `max_serve_litellm.sh` — the **MAX tier** (inference/serving). No
  matmul number of their own; the MPS column above stands in for the kernels they run.
- `train_torch_mlp.py` — the **PyTorch training** demo. A different experiment
  (learning weights, which MAX can't do), not a matmul race — so it has no row here.
