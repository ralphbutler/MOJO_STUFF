# 🛠️ MOJO_POLARIS_AURORA — Mojo/MAX on Argonne HPC

> ⚠️ **LOCATION MATTERS — MOVE BACK BEFORE RESUMING.**
> This directory must live at `/Users/rbutler/Desktop/DEMO/MOJO_POLARIS_AURORA` for
> work to continue. Its memory/recall bucket, sibling references (`../MOJO_CURRICULUM`),
> and tooling assume that path. It has been moved to a backup location for now.
> **If we're going to do more work here, move this whole directory back to that path first.**

Work dir for getting the Mojo/MAX stack running on Argonne's big machines. Prototype
on the Mac, then push to the target boxes. The canonical *teaching* source is the
sibling `../MOJO_CURRICULUM`; files here are working copies plus machine-specific kit.

## 📁 Layout

- **`POLARIS/`** — NVIDIA A100 target. Mojo's CUDA backend applies, so the GPU kit runs
  essentially unchanged. See `POLARIS/POLARIS_PLAN.md` (P0→P4 checklist).
- **`AURORA/`** — Intel Max GPU target, but **CPU-only** (no Mojo Intel-GPU backend).
  L0 probing kit + SIMD/parallel CPU work. See `AURORA/AURORA_PLAN.md` (L0→L5) and the
  detailed `AURORA/AURORA_PORT.md`.

Project infra (`.venv`, `pyproject.toml`, `uv.lock`) lives at the root — run with
`uv run mojo ...`. Plans are **malleable, not gospel**; revise as the machines teach us.

## 🗺️ Also at root

- `mac_local_check.sh` — Mac environment check; template for a future `polaris_check.sh`.

Conceptual explainers (MAX vs Mojo, MLIR) live canonically in `../MOJO_CURRICULUM`.
