# 🌌 AURORA — Mojo on Argonne's Aurora

Two phases of work on Aurora (Intel Xeon Max CPUs + Intel Data Center GPU Max "Ponte Vecchio"):
a 2026-07 CPU ladder, and a 2026-09 GPU port.

## 📂 What is in here

| Directory | What it is |
|---|---|
| **`pure_port/`** | **The port.** An Intel GPU (`spirv64`) backend added to the open-source Mojo compiler, plus a Level Zero implementation of the runtime under `DeviceContext`. The unmodified `MOJO_CURRICULUM` programs — the same files that run on Polaris' NVIDIA GPUs, host code and all — run on a PVC tile through it. **Start here: `pure_port/README.md`** |
| **`pre_process/`** | **An earlier, superseded approach.** Before the compiler was open-sourced there was no way to make Mojo emit Intel GPU code, so this route preprocessed Mojo's *Metal* compiler output into SPIR-V and drove it with hand-written Level Zero host code. It worked, but the original programs' own host code never ran, which is why the backend replaced it. Frozen — no further work goes into it. See `pre_process/README.md` |

The Phase 1 CPU ladder (2026-07) sits at this level as loose files. It ran Mojo on Aurora's CPUs
inside a container, before the newer system image made native Mojo possible:

| File | What |
|---|---|
| `aurora_setup.sh` | Login node: install `uv`, check project storage |
| `aurora_probe.pbs` | Can a compute node pull and run the Ubuntu 24.04 container Mojo needs? |
| `aurora_build.pbs` | Build the Mojo venv inside that container |
| `aurora_l0_smoke.mojo`, `aurora_l0.pbs` | L0: does Mojo run, and what CPU does it see? |
| `aurora_run.pbs` | Run any of the programs below in the container: `qsub -v MOJOFILE=<file> aurora_run.pbs` |
| `00_simd_type.mojo` | The SIMD type on its own |
| `01_vecadd_safe.mojo` | SIMD vector add, memory owned by `List` |
| `02_vecadd_parallel.mojo` | L3: SIMD + `parallelize`, bandwidth scaling |
| `02b_vecadd_numa.mojo` | L5: NUMA-aware first touch for the HBM |
| `03e_matmul_cpu.mojo` | Matmul on the CPU: naive → SIMD → parallel |
| `claude_on_containers.txt` | Notes on why the container route was chosen |

## 📄 Documents at this level

| File | What |
|---|---|
| `ASSESSMENT.md` | **The short write-up** (3 pages): what the port gets, what it cost, what it does not cover. Read this first if you want the answer rather than the evidence |
| `AURORA_RESULTS.md` | The evidence log for **both** phases and both GPU routes, in the order things happened — including the runs that failed and why. Start with "What we can and can't claim" at the top |
| `AURORA_PLAN.md` | The rung-by-rung checklist (Phase 1 L0–L5, Phase 2 G0–G6) |
| `AURORA_PORT.md` | Longer-form notes on porting to this machine |
| `REPORT_SETUP.md`, `REPORT_RESULTS.md` | The Phase 1 (2026-07, CPU-only) write-ups, kept as written |
| `FROM_POLARIS.md` | What carried over from the Polaris work, and what did not |

## 🔁 A note on duplicated files

`pure_port/` and `pre_process/` each keep their **own copy** of anything both need — the vendored
`mojo_intel_gpu/` Level Zero bindings, the batch-job runner, the oneMKL yardstick
(`mkl_sgemm.cpp`, `bench_torch_xpu.py`) and `sync_to_aurora.sh`. Nothing is shared between them.

That is deliberate: either directory can be deleted whole without breaking the other, and neither
one's scripts reach across into its neighbour. The copies are identical where the filename is the
same; `pre_process/` keeps the original `gN_` rung-numbered names, while `pure_port/` uses names
that say what a file does.

## 🧭 Reading the rung numbers

`AURORA_RESULTS.md` and `AURORA_PLAN.md` label everything with rung numbers, which are
chronological and do **not** line up with the two directories:

- **L0–L5** — Phase 1, the CPU ladder (2026-07).
- **G0–G4, G5.0, G5-lite** — Phase 2 groundwork, which is what `pre_process/` holds.
- **G5 steps 1–4, G5 hardening** — the backend and runtime, which is what `pure_port/` holds.
- **G6** — tuning and the stencil, also in `pure_port/` (renamed; mapping in `pure_port/README.md`).
