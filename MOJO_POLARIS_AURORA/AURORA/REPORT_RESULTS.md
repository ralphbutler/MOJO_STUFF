# 📊 REPORT — Mojo on ALCF Aurora (Intel CPU): Results

**2026-07-08.** Status: **L0–L4 complete + a NUMA experiment (L5a)** — all correctness PASS.
Aurora is the CPU-only Intel target (no Mojo GPU backend). Companion: `REPORT_SETUP.md` (how to
reproduce), `AURORA_RESULTS.md` (detailed running log), `AURORA_PLAN.md` (task ladder). Polaris
(A100) analog: `../POLARIS/REPORT_RESULTS.md`.

---

## 🧭 Summary

We evaluated the Mojo toolchain on ALCF **Aurora** (2× Intel Xeon CPU Max, Sapphire Rapids, HBM)
as the *hard*, non-NVIDIA half of a Mojo-adoption study. Aurora has no Intel-GPU backend, so this
is a **performance-portability-on-CPU** test: how far do portable SIMD + `parallelize` scale on
one fat HBM node? Headline findings:

- **Mojo does not install on the bare host.** All Modular wheels need **glibc ≥ 2.34**; Aurora
  ships **glibc 2.31**. Installation required an **apptainer container** (Ubuntu 24.04 / glibc
  2.39) built on a compute node. This is the dominant operational finding. (The container is
  functionally equivalent to a host with a new-enough glibc — same native CPU/cores/HBM — and
  apptainer is an ALCF-recommended way to run on Aurora, so this is a packaging requirement, not
  a runtime penalty, and a container would have been a reasonable choice regardless.)
- **Once containerized, portability holds mechanically.** The *unmodified* Mac-authored Mojo
  sources compiled and ran; `simd_width_of` reports **16** (AVX-512), and `parallelize` scales
  across the **102** cores — L3 by 11.5×, L4 (matmul) by ~966× over naive. All results correct.
- **But naive portable code leaves most of the machine on the table.** A single Sapphire Rapids
  core is weak (matmul 0.43 GFLOP/s; vector-add 15 GB/s); the parallel wins come almost entirely
  from **core count**, not per-core efficiency.
- **HBM bandwidth is *not* free.** Vector-add bandwidth peaked at **~176 GB/s** — ~10× below the
  node's HBM potential — and a NUMA-aware first-touch fix **did not help**, because Mojo's
  high-level `parallelize` exposes no thread-affinity control (Step L5a).

**Net for the adoption question:** Mojo is *portable to* Aurora but not *for free* — it needs a
container to install and lower-level control (affinity/NUMA, tiling/AMX) to exploit the hardware.

---

## 🎯 Background & goal

Aurora is the counterpoint to Polaris: no mature backend (Mojo's GPU backends are CUDA/HIP/Metal
only — no Intel Xe/Level-Zero). The goal was to (a) prove the toolchain installs and runs on
Aurora, and (b) measure how far portable Mojo **CPU** code (SIMD + `parallelize`) scales on the
unusual Sapphire-Rapids-with-HBM node. Work proceeded as a ladder L0–L5.

---

## 🖥️ Environment & method

| Item | Value |
|---|---|
| Machine | ALCF Aurora — 2× Intel Xeon CPU Max 9470 (Sapphire Rapids) / node, PBS Pro 2022.1.7 |
| Cores | **102 physical / 204 logical** usable (of 104/208; ~2 phys reserved, core-specialization) |
| ISA / memory | **AVX-512** (f32 SIMD width 16), on-package **HBM** (no DDR) |
| OS / libc | SLES 15-SP4, **glibc 2.31** (kernel 5.14) |
| Container | Ubuntu 24.04, **glibc 2.39**, apptainer 1.2.5 (built on a compute node) |
| Mojo | 1.0.0b2 (uv-managed venv inside the container) |
| Python | CPython 3.12.13 (uv-managed) |
| GPU | 6× Intel GPU Max — **unused** (no Mojo backend; `has_accelerator()==False`) |

**Method.** The toolchain was installed into a flare `.venv` from *inside* the container on a
`debug`-queue compute node (userns + proxy); all runs are `apptainer exec … .venv/bin/mojo run`
batch jobs. The **same Mojo source files** were authored/validated on an Apple M4 Max (16 cores,
NEON W=4); those are the cross-platform baseline. Each result cites its PBS output-log filename.

---

## ✅ L0 — Toolchain feasibility (via container)

`aurora_l0_smoke.mojo` (`mojo_l0.o8657258`, node x4117c5s0b0n0):

| Field | Aurora | Expected |
|---|---|---|
| `has_accelerator` | **False** | False (no Intel-GPU backend) |
| `simd width f32` / `f64` | **16 / 8** | 16 / 8 (AVX-512) |
| physical / logical cores | **102 / 204** | ~104 / ~208 |
| `sum 0..999999` | 499999500000 | 499999500000 (correct) |

The core feasibility gate — cleared, but **only inside the container**: the bare-host `uv add
mojo` is impossible (glibc 2.31 vs the wheels' 2.34 floor; confirmed by surveying every `mojo`,
`mojo-compiler`, and `max` version on PyPI — all `manylinux_2_34_x86_64`). See `REPORT_SETUP.md`
§4. `num_physical_cores()` sees 102, not 104 — ~2 cores are reserved by core-specialization;
**102 is the scaling ceiling** used below.

## ✅ L2 — Explicit SIMD semantics

`00_simd_type.mojo` (`mojo_run.o8657612`) compiled unchanged on 1.0.0b2 and produced correct
results for construct/index/splat, lane-wise arithmetic, `min`, `reduce_add`/`reduce_max`,
mask+`select` (relu), `cast[int32]`, and `Scalar == SIMD[_,1]`. *Scope:* the file is a hardcoded
width-4 teaching demo — it verifies SIMD *semantics*; native W=16 is established at L0 and
exercised at L3.

## ✅ L3 — SIMD + parallelize bandwidth scaling

`02_vecadd_parallel.mojo` (`mojo_run.o8657706`, node x4312c0s5b0n0): List-owned SIMD vector-add,
N=64M (732 MB moved/pass), W=16, worker sweep. Correctness PASS (0/64M mismatches).

| workers | Aurora GB/s | speedup | Mac GB/s (16c, W=4) |
|---:|---:|---:|---:|
| 1 | 15.2 | 1.0× | 120 |
| 4 | 35.5 | 2.33× | 285 |
| 16 | 118.0 | 7.76× | **304 (peak)** |
| 32 | **176.3 (peak)** | 11.6× | — |
| 64 | 160.3 | 10.5× | — |
| 102 | 174.6 | 11.5× | — |

`parallelize` scales (11.5×), but the **~176 GB/s peak is below the Mac's 304 and ~10× under the
Xeon-Max HBM potential** (~1 TB/s/socket). Single-core BW is just 15 GB/s (vs Mac 120): a lone
Sapphire Rapids core cannot saturate memory — bandwidth only accrues from many cores, the
opposite of the Mac's few fat cores.

## ✅ L4 — CPU matmul ladder (the payoff)

`03e_matmul_cpu.mojo` (`mojo_run.o8657642`, node x4611c6s7b0n0): N=1024, W=16, 102 cores.
Correctness vs naive reference **PASS** (0/1,048,576 mismatches for both SIMD and parallel).

| Stage | Aurora time | Aurora GFLOP/s | Mac (M4 Max, 16c) |
|---|---:|---:|---:|
| naive (scalar) | 4944.8 ms | **0.43** | 2.5 |
| SIMD | 153.3 ms | **14.0** (W=16) | 29 (W=4) |
| **parallel** | 5.12 ms | **419.4** | 212 |

Speedups: SIMD ≈ 32× over naive; parallel ≈ 30× over SIMD, **≈ 966× over naive**. Parallel
throughput is ~2× the Mac — the many-core win. But per-core is *weaker* than the Mac, and the
SIMD stage is inefficient (W=16 yet only ~32× over scalar), pointing to this **untiled** matmul
being cache/latency-bound per thread rather than FLOP-bound.

---

## 🔀 Cross-platform analysis: Aurora vs Apple M4 Max

Same unmodified sources on two very different CPUs:

| Metric | Apple M4 Max (16c) | Aurora Sapphire Rapids (102c) |
|---|---:|---:|
| matmul naive / SIMD / parallel (GFLOP/s) | 2.5 / 29 / 212 | 0.43 / 14 / **419** |
| vector-add peak BW (GB/s) | **304** | 176 |
| vector-add single-core BW | 120 | 15 |

Two takeaways:

1. **Fat-few vs many-weak.** The Mac's handful of wide cores + unified memory give huge per-core
   throughput and saturate bandwidth with a few threads; Aurora's strength is *aggregation* — its
   102 weak cores overtake the Mac only in the fully-parallel matmul (419 vs 212). Portable code
   that isn't parallel/tiled therefore looks *worse* on Aurora despite the bigger machine.
2. **Bandwidth doesn't come for free.** Despite HBM, Aurora's portable vector-add BW (176) trails
   the Mac's unified memory (304). The node's headline bandwidth is unrealized by naive portable
   code — see L5a for why the obvious fix didn't work.

---

## ⚗️ L5a — NUMA first-touch experiment: hypothesis REFUTED

Hypothesis: the ~176 GB/s ceiling was single-socket NUMA (serial init first-touches all pages on
one socket). `02b_vecadd_numa.mojo` (`mojo_run.o8657728`, node x4206c3s5b0n0) repeats L3 with
**parallel per-chunk first-touch init**. Correctness PASS.

| workers | L3 GB/s (serial init) | L5a GB/s (NUMA first-touch) |
|---:|---:|---:|
| 32 | 176.3 | 121.7 |
| 64 | 160.3 | 159.8 |
| 102 | 174.6 | **177.0 (peak)** |

**First-touch did not lift the ceiling** (177 vs 176) → the NUMA hypothesis is **refuted**.
Leading explanation: **`parallelize` gives no affinity/pinning control** — the thread that
initializes chunk *w* is not guaranteed to be the core that later computes chunk *w* (tasks can
migrate/steal across sockets), so any NUMA locality set at init is lost at compute time.
Secondary contributors: per-pass task-launch overhead and read-for-ownership store traffic (no
streaming stores). **The bandwidth wall is a property of the portable abstraction, not of page
placement.** Realizing full HBM BW would need lower-level control (thread pinning + `numactl`,
a persistent pool, non-temporal stores) — deliberately out of scope this session.

---

## ⚠️ Limitations & honest caveats

- **CPU-only by force.** Aurora's FLOPs live on its Intel GPUs, which Mojo cannot target. This is
  a portable-CPU study, not a use-the-whole-node study — the GPUs sat idle.
- **Single sizes.** Matmul at one size (N=1024); vector-add at one N (64M). No size sweeps, no
  bf16/fp16, no AMX. The matmul is untiled (naive→SIMD→parallel), so L4 is a portability +
  scaling proof, not a peak-FLOPs result.
- **`mojo run` (JIT), debug-ish.** We ran via `mojo run` inside the container; we did not explore
  AOT `mojo build -O` flags, thread-affinity, or `numactl` binding — all of which could change
  the bandwidth/FLOP numbers materially.
- **Container overhead unquantified.** We did not measure in- vs out-of-container deltas (expected
  negligible for compute, but not verified here).
- **One toolchain version.** Findings are specific to Mojo 1.0.0b2 / glibc 2.31 / Ubuntu-24.04
  image; a future Mojo with a lower glibc floor could remove the container requirement entirely.

---

## ✅ Conclusions

For the Mojo-adoption evaluation, Aurora gives a **mixed but clear** verdict. **Positive:** the
identical Mojo source that ran on Apple and (in the sibling report) NVIDIA also compiles and runs
correctly on Intel Sapphire Rapids, at full AVX-512 width, with `parallelize` scaling to 102
cores and a matmul reaching 419 GFLOP/s — real performance portability across three vendors with
zero code change. **Negative / caveats:** (1) it **cannot be installed on the bare host** (glibc
2.31 vs the wheels' 2.34 floor) and requires a container — though that container is functionally
equivalent to a correct-glibc host and is an ALCF-endorsed way to run on Aurora anyway, so it is
a packaging requirement rather than a performance cost; and (2) **naive portable code does not
exploit the node** — per-core is weak, the untiled matmul is cache-bound, and HBM bandwidth stays
~10× under potential with no affinity/NUMA knobs in the portable API to close the gap.

Aurora is thus the honest stress-test the plan intended: Mojo's portability *claim* holds, but on
non-NVIDIA HPC iron, *reaching the hardware* still needs lower-level control the high-level
abstractions don't yet expose.

### ➡️ Optional follow-ups
1. **Affinity/NUMA bandwidth retry** — thread pinning + `numactl --localalloc` (or a persistent
   pool) to test whether HBM BW can be unlocked once locality is enforced.
2. **Tiled / AMX matmul** — a register-blocked kernel and a check of whether Mojo taps the Xeon
   AMX units, for a real peak-FLOPs number (mirrors the Polaris "coarse" kernel).
3. **AOT `mojo build -O` + size sweeps + bf16** for a fuller performance picture.
4. **Container-overhead measurement** and a future re-test if Mojo lowers its glibc floor.
