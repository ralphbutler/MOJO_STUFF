# 🌌 Porting Mojo to Aurora (Argonne) — One-Node Experiment

**Goal:** get a graduated set of Mojo programs running on a single Aurora node, and see
how far portable Mojo **CPU** code scales on its unusual hardware.

**Status (2026-07-08): L0 PASSED on Aurora — via an apptainer container.** The bare-host
install is **impossible** (all Modular Mojo wheels need glibc ≥ 2.34; Aurora is glibc 2.31),
so Mojo runs inside an Ubuntu-24.04 (glibc 2.39) apptainer image built on a *compute* node.
Confirmed: `has_accelerator=False`, `simd f32=16` (AVX-512), 102 physical cores, correct
execution. Full detail + the PyPI wheel survey in `AURORA_RESULTS.md`. CPU ladder L1–L4 next.

## ⛔ The one hard constraint: Aurora's GPUs are off the table

Aurora's FLOPs come from **6× Intel Data Center GPU Max (Ponte Vecchio)** per node.
Mojo's GPU backends are **CUDA / HIP / Metal only** — there is **no Intel Xe / Level-Zero /
SYCL backend**. So on Aurora:

- `has_accelerator()` will return **False**.
- None of our GPU matmul kernels (`matmul_gpu*.mojo`) will run.

There is no workaround today. The experiment is therefore **CPU-only** — which is still
genuinely interesting here.

## 💡 Why the CPU side is worth it

Each Aurora node has **2× Intel Xeon CPU Max (Sapphire Rapids)** ≈ **104 physical cores**,
with **AVX-512**, **AMX** (on-chip matrix units), and **on-package HBM**. That combination —
high core count *and* GPU-class memory bandwidth — is rare. The real question becomes:

> How far can portable Mojo CPU code (SIMD + `parallelize`) scale on one fat HBM node?

That's a better story than re-running the GPU demo, and it directly tests Mojo's
"performance-portable" claim on non-NVIDIA HPC iron.

## 🪜 The graduated ladder (CPU, single node)

| Lvl | Program | What it proves |
|---|---|---|
| **L0** | `mojo --version` + smoke test: `has_accelerator()`, `simd_width_of`, core count, tiny loop | Mojo installs, compiles, and runs on an Aurora compute node |
| **L1** | Scalar numeric loop | Baseline single-core compute |
| **L2** | Explicit `SIMD[f32, W]` reduction | Does it emit **AVX-512** (W=16)? |
| **L3** | `vectorize` + `parallelize` across ~104 cores | Shared-memory scaling on the node |
| **L4** | CPU matmul: naive → SIMD-tiled → parallel, GFLOP/s | The payoff; stresses HBM bandwidth |
| **L5** | *(stretch)* Python interop and/or check if Mojo uses **AMX** for matmul | Ties into the existing Python stack; probes matrix units |

Each rung builds on the last. All are written/compile-checked locally on the Mac (ARM)
first; the real numbers come from Aurora. **Single node only — no MPI.**

## 🧰 The working kit (as actually used on Aurora)

Everything lives on live ModCon storage: `/lus/flare/projects/ModCon/$USER/MOJO_WORK/AURORA`.

- `ubuntu2404.sif` — apptainer image (glibc 2.39), pulled on a **compute** node from
  `docker://ubuntu:24.04`. Provides the newer glibc the Mojo wheels require.
- `aurora_build.pbs` — **batch** job; from *inside* the container it runs `uv python install
  3.12`, `uv init`, `uv add mojo`, `uv sync` → builds `.venv` on flare (Mojo compiler installed).
- `aurora_l0.pbs` — **batch** job; runs the smoke test via
  `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run aurora_l0_smoke.mojo`.
- `aurora_run.pbs` — reusable generic runner: `qsub -v MOJOFILE=<file>.mojo aurora_run.pbs`.
- `aurora_l0_smoke.mojo` — the smoke test.
- `aurora_setup.sh` — login-node prep (uv install) + notes; the bare-host `uv add mojo` it
  originally did **fails** on Aurora (see below), so the real install is `aurora_build.pbs`.
- `aurora_probe.pbs` — one-off container feasibility probe (kept for the record).

### The actual bring-up sequence (proven 2026-07-08)

```bash
# LOGIN node: install uv (host binary is reused inside the container), stage files on ModCon.
bash aurora_setup.sh          # uv install + write-test; DO NOT expect `uv add mojo` to work here

# COMPUTE node (batch — has BOTH user-namespaces AND proxied internet):
qsub aurora_build.pbs         # pull image (first time) + install Mojo into .venv  → mojo_build.o*
qsub aurora_l0.pbs            # L0 smoke test                                      → mojo_l0.o*
```

Why compute-node batch: apptainer needs **user namespaces** (blocked on login, allowed on
compute), and `uv add mojo` needs **internet** (compute reaches it via
`http_proxy=https_proxy=http://proxy.alcf.anl.gov:3128`). Both conditions coexist only on a
compute node. `qsub -I` works but the interactive queue was backed up; batch was smoother.

### L0 output — expected vs actual

| Field | Mac (ARM) | Aurora expected | **Aurora actual** |
|---|---|---|---|
| `has_accelerator` | True | False | **False** ✅ |
| `simd width f32` | 4 | 16 | **16** ✅ (AVX-512) |
| `simd width f64` | 2 | 8 | **8** ✅ |
| `physical cores` | 16 | ~104 | **102** ✅ (core specialization) |
| `logical cores` | 32 | ~208 | **204** ✅ |
| `sum 0..999999` | 499999500000 | same | **499999500000** ✅ |

## ⚠️ Logistics — how the risks actually resolved

- **Does the Mojo wheel install on Aurora?** **No, not on the bare host.** Every Modular
  wheel (`mojo`/`max`/`mojo-compiler`, all versions) is `manylinux_2_34_x86_64` → needs
  glibc ≥ 2.34; Aurora is **glibc 2.31**. This is the marquee finding. **Resolved via an
  apptainer container** (Ubuntu 24.04 = glibc 2.39). Perf is unaffected (native CPU).
- **Internet on nodes:** login has **direct** outbound (no proxy). Compute nodes reach the
  internet **only via `proxy.alcf.anl.gov:3128`** — set it in the container-build job.
- **Scheduler:** PBS Pro 2022.1.7; `-l filesystems=flare:home`; `-A ModCon`; `-q debug`.
  Aurora PBS **rejects inline comments after `#PBS` directive values** — keep them bare.

## ❓ Unresolved questions — RESOLVED (2026-07-08)

1. OS / glibc? → **SLES 15-SP4, glibc 2.31** (the blocker; forced the container path).
2. Login proxy? → **none needed on login (direct)**; compute uses `proxy.alcf.anl.gov:3128`.
3. `qsub -I` OK? → **yes**, but batch was more reliable (queue depth) — matches Polaris.
4. `uv` allowed? → **yes** (curl install to `~/.local/bin`); no conda/frameworks module.
5. PBS values? → **`-A ModCon`, `-q debug`, `-l filesystems=flare:home`**.
6. Day's goal? → L0 achieved; L1–L4 scope TBD per session.

## 📄 Related

- `AURORA_RESULTS.md` — the running results/experience log (source for the REPORT_ files).
- `RESULTS01.md` — the local Apple-GPU matmul experiment (context for the CPU ladder).
- `../POLARIS/` — the NVIDIA A100 sibling port (`REPORT_SETUP.md`, `REPORT_RESULTS.md`,
  `POLARIS_RESULTS.md` are the template for Aurora's three reports).
