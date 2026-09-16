# 🛠️ MOJO_POLARIS_AURORA — Mojo/MAX on Argonne HPC

Getting the Mojo/MAX stack running on Argonne's big machines: prototype on the Mac, then push to
the target boxes. The canonical *teaching* source is the sibling `../MOJO_CURRICULUM`; files here
are working copies plus machine-specific kit.

## 📁 Layout

- **`POLARIS/`** — NVIDIA A100 target. Mojo ships a CUDA backend, so the GPU kit runs essentially
  unchanged. See `POLARIS/POLARIS_PLAN.md` (P0→P4 checklist).
- **`AURORA/`** — Intel Data Center GPU Max (Ponte Vecchio) target.
  - Phase 1 (2026-07): the CPU ladder — SIMD and `parallelize` on Sapphire Rapids + HBM.
    See `AURORA/AURORA_PLAN.md` and `AURORA/AURORA_PORT.md`.
  - Phase 2 (2026-09): **the GPU port.** Mojo had no Intel GPU backend and the compiler was
    closed; Modular open-sourced it on 2026-09-13, and `AURORA/pure_port/` is an `spirv64`
    backend we added to it, plus a Level Zero implementation of the runtime under
    `DeviceContext`. An earlier superseded approach is kept in `AURORA/pre_process/`.
    **Start at `AURORA/README.md`.**

## 🟢 Where Phase 2 got to

The unmodified `MOJO_CURRICULUM` programs `02_vecadd_gpu`, `03c_matmul_coarse` and
`04b_train_mlp_gpu` — the same files that run on Polaris' NVIDIA GPUs, host code and all — run on
an Aurora PVC tile through that backend. Vector add matches on all 1,000,000 elements, matmul
matches its CPU reference, and the training loop reproduces its loss curve
(2.1783555 → 0.0005614754) to every digit printed.

This is our own build of the Mojo compiler, not an official release; Modular does not support
Intel GPUs. `AURORA/pure_port/README.md` states the limits in full — one tile, float32, an 18-function
runtime slice, newer-system-image nodes only, and kernels untuned for this hardware.

Evidence for every rung, including the runs that failed and why, is in `AURORA/AURORA_RESULTS.md`.

## 🗺️ Also at root

- `mac_local_check.sh` — Mac environment check; template for a future `polaris_check.sh`.

Project infra (`pyproject.toml`, `uv.lock`) lives at the root — run with `uv run mojo ...`.
Conceptual explainers (MAX vs Mojo, MLIR) live canonically in `../MOJO_CURRICULUM`.
