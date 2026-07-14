# 🌌 Mojo on Aurora (Argonne) — Plan

> **Malleable, not gospel.** Revise freely as we learn. Check boxes off as we go.
> Detailed rationale, L0 kit steps, and output-reading tables live in
> **`AURORA_PORT.md`** (this file is the checklist view of it).

## 🧭 Fresh session? Start here (this dir is memory-blind)

A session rooted in this subdir does **not** auto-load the `MOJO_CURRICULUM` project
memory — this plan (+ `AURORA_PORT.md`) is the source of truth. Context you'd otherwise
get from memory:

> 📌 **Read `FROM_POLARIS.md` first.** The Polaris (A100) port is DONE (2026-07-08) and taught
> us how to actually operate on an ALCF PBS machine — storage traps, login/compute split, the
> proven PBS recipe, MAX install — plus what is **NVIDIA-only and must NOT be copied here**
> (the `ptxas`/driver workaround; GPU execution — Aurora is CPU-only). That file is the bridge;
> full detail in `../POLARIS/REPORT_SETUP.md`.

- **Why:** part of an evaluation of switching Argonne's stack to the Mojo/MAX ecosystem.
  Aurora is the *hard* target — Mojo has no Intel-GPU backend, so this is the **CPU-only**
  litmus test: how far do portable SIMD + `parallelize` scale on one fat HBM node?
- **Canonical teaching source:** `../../MOJO_CURRICULUM` — files here are working copies;
  don't edit teaching copies there. Sibling `../POLARIS/` is the NVIDIA A100 target.
- **Run:** `uv run mojo <file>` (venv lives at the `MOJO_WORK/` root, one level up).
  The `mojo-syntax` skill is available and **must** be used when writing Mojo.
- **First action once the ALCF account is live:** L0 below (toolchain feasibility gate) —
  `aurora_setup.sh` on a login node, then `qsub aurora_l0.pbs`.
- Note: on Aurora itself there's no venv yet — `aurora_setup.sh` builds it on a login
  node; `uv run` working locally just means the kit is sound.

## 🖥️ Machine profile

- **GPUs: off the table.** 6× Intel Data Center GPU Max (Ponte Vecchio) per node, but Mojo
  has **no Intel Xe / Level-Zero backend** — `has_accelerator()` returns **False**, GPU
  matmuls won't run. No workaround today. → **This experiment is CPU-only.**
- **CPU (the point):** 2× Intel Xeon CPU Max (Sapphire Rapids), ~104 physical cores,
  **AVX-512 + AMX**, on-package **HBM**. High core count *and* GPU-class bandwidth — rare.
- **Scheduler:** PBS Pro (`qsub`, `-l filesystems=...`). Single node only, no MPI.

## 🎯 Goal

Test Mojo's *performance-portable CPU* claim on non-NVIDIA HPC iron: how far do portable
SIMD + `parallelize` scale on one fat HBM node?

## 🪜 The CPU ladder (check off as accomplished)

- [x] **L0 — Toolchain feasibility (the gate).** ✅ PASSED 2026-07-08 (`mojo_l0.o8657258`).
      NOTE: bare-host install is **impossible** (Mojo wheels need glibc ≥2.34; Aurora=2.31) —
      runs via **apptainer container** (Ubuntu 24.04/glibc 2.39) built on a *compute* node.
      Results: `has_accelerator=False`, `simd f32=16` (AVX-512), `f64=8`, **102 phys / 204 log
      cores**, sum correct. All Mojo runs: `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run <file>`.
      See AURORA_RESULTS.md for the glibc finding + wheel survey.
- [ ] **L1 — Scalar baseline.** Single-core numeric loop.
- [x] **L2 — Explicit SIMD.** ✅ PASSED 2026-07-08 (`mojo_run.o8657612`). `00_simd_type.mojo`
      compiled unchanged on `mojo 1.0.0b2`; all SIMD-type ops correct (arith, reductions,
      mask/select, cast). NOTE: this demo is hardcoded width 4 — it verifies SIMD *semantics*,
      not native W=16 (that was shown at L0 via `simd_width_of`, and gets exercised at L3).
- [x] **L3 — Parallel scaling.** ✅ PASSED 2026-07-08 (`mojo_run.o8657706`,
      `02_vecadd_parallel.mojo`). Worker sweep 1→102, N=64M, W=16, 0 mismatches. `parallelize`
      scales 11.5× — BUT peak only **~176 GB/s** (saturates ~32 workers), *below* the Mac's 304
      and ~10× under Xeon-Max HBM. Single-core BW just 15 GB/s. Suspected **NUMA first-touch**
      (single-thread init pins pages to one socket). Fix = parallel NUMA-aware init / streaming
      stores → L5. Key finding: portable `parallelize` scales, but HBM bandwidth is NOT free.
      See AURORA_RESULTS.md.
- [x] **L4 — CPU matmul (payoff).** ✅ PASSED 2026-07-08 (`mojo_run.o8657642`). N=1024, W=16,
      102 cores, 0 mismatches. **naive 0.43 → SIMD 14.0 → parallel 419.4 GFLOP/s** (~966× naive).
      Parallel ~2× the Mac (212); but per-core is *weaker* than M4 Max and SIMD efficiency is
      low (naive matmul is cache/latency-bound per thread) — tiling/AMX (L5) is the path higher.
      See AURORA_RESULTS.md.
- [~] **L5 — (stretch)** L5a done 2026-07-08 (`mojo_run.o8657728`, `02b_vecadd_numa.mojo`):
      NUMA-aware first-touch **did NOT** lift the ~176 GB/s BW ceiling (177 vs 176) →
      hypothesis refuted; `parallelize` exposes no affinity/pinning, so first-touch locality is
      lost at compute. Realizing HBM BW needs lower-level control (thread pinning + numactl,
      streaming stores). Still open: AMX/tiled matmul, Python interop. See AURORA_RESULTS.md.

## 📦 Files staged here

Container kit (the working path): `ubuntu2404.sif` (glibc 2.39 image) · `aurora_build.pbs`
(installs Mojo into `.venv` inside the image) · `aurora_l0.pbs` (L0 smoke) · `aurora_run.pbs`
(generic runner, `-v MOJOFILE=…`) · `aurora_l0_smoke.mojo` · `aurora_probe.pbs` (probe, archival).
Login prep: `aurora_setup.sh` (uv install only — bare-host `uv add mojo` fails, see below).
CPU material: `00_simd_type.mojo` (L2) · `01_vecadd_safe.mojo` (L1/serial base) ·
`02_vecadd_parallel.mojo` (L3, `parallelize` bandwidth sweep — written+compile-checked on 1.0.0b2) ·
`03e_matmul_cpu.mojo` (L4).
Reference: `AURORA_PORT.md` · `AURORA_RESULTS.md` (running log) · `FROM_POLARIS.md`.
**All Mojo runs go through the container:** `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run <file>`.

## ⚠️ Known risks — resolved outcomes

- **Make-or-break (MATERIALIZED):** the Mojo wheels are `manylinux_2_34` (need glibc ≥ 2.34);
  Aurora is **glibc 2.31**, so the bare-host install is impossible. **Resolved via apptainer
  container** (Ubuntu 24.04 / glibc 2.39) built on a compute node. See `AURORA_RESULTS.md`.
- Compute nodes are offline *except via proxy* (`proxy.alcf.anl.gov:3128`) — used during the
  container build; the L0/run jobs need no network (venv pre-built on flare).

## ❓ Unresolved — RESOLVED (2026-07-08)

1. OS / glibc? → **SLES 15-SP4 / glibc 2.31** (the blocker).
2. Login proxy? → **none on login (direct)**; compute uses `proxy.alcf.anl.gov:3128`.
3. `qsub -I` vs batch? → both work; **batch preferred** (queue depth), matching Polaris.
4. `uv`? → **yes**, curl-installed to `~/.local/bin`; no conda/frameworks module.
5. PBS values? → **`-A ModCon`, `-q debug`, `-l filesystems=flare:home`** (no inline `#PBS` comments).
6. Day's goal? → L0 done; L1–L4 per session.
