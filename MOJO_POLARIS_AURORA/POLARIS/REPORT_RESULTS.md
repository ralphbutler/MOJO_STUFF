# 📊 REPORT — Mojo/MAX on ALCF Polaris (A100): Results

**2026-07-08.** Status: **P0–P4 complete** — Mojo GPU kernels, training, and MAX inference all
verified on the A100. The only deferred item is an *optional* shared-endpoint serving demo.
Companion: `REPORT_SETUP.md` (how to reproduce), `RESULTS01.md` (matmul deep-dive with the
Apple-Metal baseline), `POLARIS_PLAN.md` (task ladder).

---

## 🧭 Summary

We evaluated the Mojo/MAX toolchain on ALCF **Polaris** (NVIDIA A100) as evidence for a
possible move toward the Modular stack. The headline findings:

- **Portability holds.** A single, *unmodified* Mojo GPU codebase that was written and
  benchmarked on Apple Silicon (Metal) compiled and ran correctly on NVIDIA A100 (CUDA) —
  kernels, and a full training loop, unchanged.
- **Competitive performance.** A hand-written Mojo matmul reached **8,966 GFLOP/s** on the
  A100 (fp32, N=2048) — **~65% of NVIDIA cuBLAS** and ~46% of the card's fp32 peak.
- **Correctness is exact.** A full MLP training loop (forward + backward as hand-written GPU
  matmuls) reproduced its reference loss curve **bit-for-bit** on the A100.
- **MAX inference works too.** MAX 26.4.0 loaded and ran an LLM (Qwen2.5-0.5B) on the A100 — both
  via the `max generate` CLI and the in-process Python API — under the same environment workaround.
- **One environment caveat.** The only Polaris-specific friction was a driver-version
  workaround (compile through a system `ptxas`, covering both Mojo and MAX); **no application
  code changed.**

---

## 🎯 Background & goal

Polaris is the "should just work" NVIDIA target (Mojo has a mature CUDA backend). The goal was
to (a) prove the toolchain installs and runs on Argonne iron, (b) get **real A100 numbers** for
hand-written Mojo kernels vs tuned libraries, and (c) run a real model through MAX. Work
proceeded as a 5-rung ladder (P0–P4), all covered here; only an optional shared-endpoint serving
demo is deferred.

---

## 🖥️ Environment & method

| Item | Value |
|---|---|
| Machine | ALCF Polaris — 4× NVIDIA A100-SXM4-40GB / node, AMD EPYC (Milan), PBS Pro |
| NVIDIA driver | 570.124.06 (CUDA 12.8) |
| Mojo | 1.0.0b2 (`uv`-managed venv) |
| Python | CPython 3.12.11 (ALCF conda 2025-09-25) |
| PyTorch | 2.8.0 (CUDA 12.9 build, ships in conda base) |
| MAX | 26.4.0 (`modular==26.4.0`, separate venv) — P4 |

**Method.** Toolchain installed on the login node (proxy) via `uv sync`; GPU work run in PBS
batch jobs on `debug`-queue A100 nodes. Matmul benchmarks: square fp32, N=2048, 50 timed
iterations, throughput = 2·N³/time. All Mojo GPU kernels were AOT-compiled through a system
`ptxas` (CUDA 12.8) to satisfy the driver — see `REPORT_SETUP.md` §5. The **same Mojo source
files** were previously benchmarked on an Apple M4 Max (Metal); those numbers are the
cross-platform baseline (`RESULTS01.md`).

---

## ✅ P0 — Toolchain feasibility

The Mojo pip wheel installs and runs on Polaris: `uv sync` (mojo 1.0.0b2), `mojo --version`,
`hello.mojo`, and a SIMD exercise (`00_simd_type.mojo`) all succeeded on the login node. This
was the core feasibility gate — cleared with no source changes.

## ✅ P1 — GPU execution on A100

`02_vecadd_gpu.mojo`, N = 1,000,000: `has_accelerator() == True`, kernel launched on an A100,
**RESULT PASS**, 0/1,000,000 mismatches, **0.00766 ms/pass** (kernel-only). The first attempt
hit the driver wall; the system-`ptxas` workaround resolved it and is required for all
subsequent GPU runs.

## ✅ P2 — Matmul throughput (N=2048, fp32)

| Implementation | avg time | GFLOP/s | % of cuBLAS | vs naive |
|---|---:|---:|---:|---:|
| PyTorch CUDA (**cuBLAS**, true fp32) | 1.241 ms | **13,839** | 100% | 5.20× |
| **Mojo coarse** (register-blocked, 128×128 / 8×8) | 1.916 ms | **8,966** | **64.8%** | 3.37× |
| Mojo tiled (16×16 shared memory) | 4.153 ms | 4,137 | 29.9% | 1.56× |
| Mojo naive (1 thread/output) | 6.462 ms | 2,659 | 19.2% | 1.00× |
| PyTorch CPU (EPYC Milan) | 12.287 ms | 1,398 | — | 0.53× |

Correctness (`03d_matmul_check.mojo`, N=256, asymmetric inputs): **PASS**, 0/65,536 mismatches,
max error 0.0. **A hand-written Mojo kernel reaches ~65% of NVIDIA's own tuned library** on
their hardware; cuBLAS itself is ~71% of the A100's ~19.5 TFLOP/s fp32 peak.

## ✅ P3 — Training portability & correctness

`04b_train_mlp_gpu.mojo` (N=256, H=16, 300 epochs) — an MLP whose forward **and** backward
passes are entirely hand-written GPU matmuls — ran on the A100 and reproduced its reference
loss curve **bit-for-bit**: epoch 1 `2.1783555` → final `0.00056147523` (reference:
2.178 → 0.000561), at **0.116 ms/epoch**. The reference curve was established on Apple Metal;
the A100 run is numerically identical, i.e. a complete training loop ports across GPU vendors
with **zero code change and zero numeric drift**.

PyTorch (`train_torch_mlp.py`, `device=cuda`) independently converged to 100% train accuracy.
*Caveat:* the two sides ran different configurations and PyTorch emitted no timing, so P3 is a
**portability/correctness** result, not a training wall-clock race (the matmul-vs-cuBLAS table
above is the like-for-like speed comparison).

---

## 🔀 Cross-platform analysis: the "tiling flip"

The same Mojo source, N=2048, on two GPUs:

| Kernel (GFLOP/s) | Apple M4 Max (Metal) | NVIDIA A100 (CUDA) |
|---|---:|---:|
| naive | 2,693 (1.00×) | 2,659 (1.00×) |
| simple-tiled 16×16 | 2,145 (**0.80× — slower**) | 4,137 (**1.56× — faster**) |
| coarse register-blocked | 4,330 (1.61×) | 8,966 (3.37×) |

Two results worth highlighting:

1. **Simple shared-memory tiling reverses sign across vendors** — it *loses* to a cache-friendly
   naive kernel on Apple but *wins* by 1.56× on the A100, from an identical source file. This is
   a clean demonstration that "tiling always wins" is **NVIDIA-specific folklore**: NVIDIA's
   smaller per-SM caches and explicit shared memory reward staging data; Apple's large caches +
   unified memory already deliver most of that reuse to the naive kernel for free.
2. **Naive is ~equal on both GPUs** (~2.7 TFLOP/s) despite the A100's far larger DRAM
   bandwidth — evidence the naive kernel is bound by per-thread reload latency, not raw
   bandwidth. Reuse (tiling/coarsening) is what unlocks the hardware: coarse hits 9.0 TFLOP/s on
   A100 vs 4.3 on Metal. Correspondingly, on the A100 even *naive* Mojo (2,659) beats
   PyTorch-CPU (1,398), the opposite of the Mac, where naive lost to Accelerate CPU BLAS.

---

## ✅ P4 — MAX inference on A100

MAX **26.4.0** (separate `.venv-max`) loaded **Qwen/Qwen2.5-0.5B-Instruct** onto the A100
(`gpu[0]`) and produced coherent completions, both from the `max generate` CLI and the
in-process Python API. Findings:

- **The Mojo `ptxas` workaround also fixes MAX.** MAX hit the identical driver wall
  (needs ≥580; Polaris has 570) and is resolved by the *same* `MODULAR_NVPTX_COMPILER_PATH`
  env var — one workaround covers the entire Modular stack (compiler + inference).
- **Single-stream performance (0.5B, A100):** token-generation **310.4 tok/s**, time-to-first-
  token 1.17 s, 3.22 ms/token. One-time startup ~110 s (AOT model-graph compile 62.7 s + init) —
  a fixed cost amortized once the model is resident, which is what `serve` provides.

**In-process Python API (done — the primary HPC pattern).** Using
`from max.entrypoints.llm import LLM` under `.venv-max/bin/python` on an A100, the model was
loaded **once (44.9 s)** and then answered **three prompts in 0.58 s (0.19 s/prompt)** — the
load/compile cost amortized across all calls. This is how MAX would be embedded in a job or
pipeline on Polaris/Aurora, with no server involved. One reproducibility note: the driver
program must guard its body with `if __name__ == "__main__":`, because MAX's `LLM` launches a
telemetry worker via Python multiprocessing (spawn) that re-imports the module.

Remaining (optional): `max serve` + a litellm round-trip — a persistent OpenAI-compatible
endpoint, relevant only for a shared/multi-client service, not for embedded inference.

---

## ⚠️ Limitations & honest caveats

- **fp32, one size (N=2048).** We did not sweep sizes or test TF32/bf16; TF32 on the A100 would
  raise cuBLAS substantially and change the ratio. The 65%-of-cuBLAS figure is fp32-vs-fp32.
- **P3 training is not a speed race** (different configs, no PyTorch timing) — it establishes
  correctness/portability only.
- **Small models/kernels.** The MLP is tiny (launch-overhead-dominated); the matmul is a single
  square size. These are capability/portability proofs, not a production performance study.
- **One driver/toolkit combination.** Findings are specific to Polaris's 570/CUDA-12.8 driver
  and the Mojo 1.0.0b2 / MAX 26.4.0 versions; other combinations may not need — or may differ
  in — the `ptxas` workaround.

---

## ✅ Conclusions

For the Mojo-switch evaluation, Polaris delivers strong positive evidence: **the same Mojo GPU
code runs correctly and competitively on NVIDIA A100 with no source changes**, a hand-written
kernel reaches ~65% of cuBLAS, a full training loop reproduces exactly, and MAX 26.4.0 runs LLM
inference on the A100 — all under a single documented, one-line environment workaround for the
site's older driver, with no application code changed. Only optional follow-ups remain.

### ➡️ Optional follow-ups
1. `max serve` + litellm endpoint — the shared/multi-client serving case (not needed for
   embedded inference, which the in-process API already covers).
2. Matched-dim, timed Mojo-vs-PyTorch training race.
3. Size sweep (N=1024/4096) and TF32/bf16 columns for a fuller performance picture.
4. Aurora port (CPU-only Intel target) reusing the workflow from `REPORT_SETUP.md`.
