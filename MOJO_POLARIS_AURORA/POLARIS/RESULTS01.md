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

## 🚀 Cross-platform: NVIDIA A100 (Polaris) — same Mojo source, N = 2048

**Date:** 2026-07-07 · **Machine:** NVIDIA A100-SXM4-40GB, ALCF Polaris ·
**Toolchain:** Mojo 1.0.0b2, AOT-compiled via system `ptxas` 12.8 (Polaris driver 570/CUDA
12.8 predates Mojo's required driver 580, so kernels are compiled ahead-of-time — see
`POLARIS_PLAN.md`). The kernels are **byte-for-byte the same files** that ran on the Mac;
the `(Apple GPU / Metal)` strings in the output are just hardcoded labels, not the backend.

| Implementation (N=2048) | Time | GFLOP/s | vs naive |
|---|---|---|---|
| PyTorch CUDA (cuBLAS, true fp32) | 1.241 ms | **13,839** | 5.20× |
| **Mojo coarse (register-blocked, 128×128 / 8×8)** | **1.916 ms** | **8,966** | **3.37×** |
| Mojo tiled (16×16 shared memory) | 4.153 ms | 4,137 | 1.56× |
| Mojo naive | 6.462 ms | 2,659 | 1.00× |
| PyTorch CPU (EPYC Milan) | 12.287 ms | 1,398 | 0.53× |

Correctness (`03d_matmul_check.mojo`, N=256): **PASS, 0/65536 mismatches, max err 0.0.**

**Mojo coarse reaches ~65% of cuBLAS** (8,966 / 13,839) on the same A100 — a hand-written
kernel vs NVIDIA's tuned library. cuBLAS itself is ~71% of the A100's ~19.5 TFLOP/s fp32 peak.
Note the A100 flips the Mac result where naive Mojo *lost* to CPU BLAS: here even naive
(2,659) beats PyTorch-CPU (1,398), because the A100's memory system rewards the GPU with no
reuse, whereas Apple's cache-tuned CPU BLAS was hard to beat.

### 🔀 The tiling flip — Apple vs NVIDIA, identical code

| Kernel (N=2048, GFLOP/s) | Apple M4 Max (Metal) | NVIDIA A100 (CUDA) |
|---|---|---|
| naive | 2,693 (1.00×) | 2,659 (1.00×) |
| simple-tiled 16×16 | 2,145 (**0.80× — slower**) | 4,137 (**1.56× — faster**) |
| coarse register-blocked | 4,330 (1.61×) | 8,966 (3.37×) |

Two things fall out:

- **Simple tiling reverses sign across vendors.** On Apple it *lost* to a cache-friendly
  naive kernel; on the A100 it *wins* by 1.56×. This is the cleanest possible confirmation
  of the earlier claim that **"tiling always wins" is NVIDIA folklore** — it is literally
  true on NVIDIA and false on Apple, from the same source file. NVIDIA's smaller L1/L2 per
  SM and explicit shared memory make staging data pay off; Apple's big caches + unified
  memory already gave naive most of that reuse for free.
- **Naive is nearly identical on both GPUs** (~2.7 TFLOP/s) despite the A100's far larger
  DRAM bandwidth — evidence the naive kernel is bound by per-thread reload latency, not raw
  bandwidth, so it can't cash in the A100's memory system. Reuse (tiling/coarsening) is what
  unlocks the hardware: coarse hits **9.0 TFLOP/s on A100 vs 4.3 on Metal**, a ~2× platform
  gap that only appears once arithmetic intensity is high.

**Takeaway for the Mojo-switch eval:** one unmodified Mojo GPU codebase ran correctly on both
Apple Metal and NVIDIA CUDA, and on the A100 the coarse kernel reached ~46% of fp32 peak
(~19.5 TFLOP/s) — a hand-written kernel, no cuBLAS. Portability + real A100 throughput, which
was the point of P2.

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
