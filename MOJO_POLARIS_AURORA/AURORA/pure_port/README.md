# 🟢 Mojo on Aurora's Intel GPUs — an `spirv64` backend for the open-source Mojo compiler

The same Mojo GPU programs that run on Polaris (NVIDIA A100) run **unmodified** on an Aurora
Intel Data Center GPU Max (Ponte Vecchio) tile — host code and all — through an Intel GPU
backend we added to the open-source Mojo compiler, plus a Level Zero implementation of the
runtime that sits under `DeviceContext`.

```
unchanged .mojo
  -> our fork of the open-source Mojo compiler   (spirv64 target: backend/)
  -> LLVM's SPIR-V codegen
  -> our C++ Level Zero runtime under DeviceContext  (runtime/)
  -> PVC tile
```

No Metal backend, no `air2spir.py`, no `clang`, no `llvm-spirv` anywhere in that path.

## 📊 What runs

Three unmodified programs from the sibling `MOJO_CURRICULUM`, the same files that run on Polaris:

| Program | Result | Speed |
|---|---|---|
| `02_vecadd_gpu.mojo` | 0 mismatches / 1,000,000 | 0.0403 ms/pass |
| `03c_matmul_coarse.mojo` | `C[0,0]` exact vs CPU reference | 1,489 GFLOP/s |
| `04b_train_mlp_gpu.mojo` | loss 2.1783555 → 0.0005614754 | 0.354 ms/epoch |

The Aurora-built compiler emits **byte-identical** SPIR-V to the Mac-built one.
Full evidence, including the runs that failed and why, is in `../AURORA_RESULTS.md`;
raw job logs are in `evidence/`.

## 📁 Layout

| Path | What |
|---|---|
| `backend/` | The `spirv64` target: patches + new files against `modular` @ `6417db28`. See `backend/README.md` |
| `fork_overlay.tar.gz` | The same files packed for `prepare_fork.sh`; rebuilt by `make_overlay.sh` |
| `prepare_fork.sh` | Aurora: clone `modular` @ `6417db28`, apply the overlay, set up the Bazel cache |
| `build_fork.pbs` | Build the fork on a `next-eval` node (~51 min cold, ~90 s incremental) |
| `runtime/mojo_level_zero_rt.cpp` | Our Level Zero implementation of the 18 `AsyncRT_*` entry points `DeviceContext` calls |
| `build_runtime.sh` | Build `libmojo_level_zero_rt.so` (`-lze_loader`) |
| `e2e_run.pbs` | **The end-to-end run**: unmodified 02 / 03c / 04b on a PVC tile (JIT path) |
| `e2e_aot.pbs` | The ahead-of-time `mojo build` path |
| `mojo_intel_gpu/` | Vendored Level Zero bindings for Mojo (MIT — see its `LICENSE`) |
| `evidence/` | Raw job output from the two end-to-end runs |

**Backend bring-up** (kept as the record of how the target was built up):
`bringup_skeleton.mojo` (first valid SPIR-V) → `bringup_thread_ids.mojo` / `bringup_barrier.mojo`
(thread indexing, shared memory, barrier) → `ocloc_check.sh` (does Intel's compiler accept it?).

**Kernel emitters and harnesses** (step 3 — backend-built kernels driven by a hand-written host,
before the runtime made the unmodified host code work): `emit_vecadd.mojo`, `emit_matmul.mojo` and
`emit_mlp.mojo` produce the `g5_*.spv` kernels; `run_vecadd.mojo`, `run_matmul.mojo` and
`run_train_mlp.mojo` run them through `run_job.pbs`
(`qsub -v MOJOFILE=run_matmul.mojo run_job.pbs`).

## 🏷️ Reading the filenames

Names here say what a file **does**. The `gN_` prefixes in `../AURORA_RESULTS.md` are **rung
numbers** — chronological milestones, not routes — and they do not line up with the split: rungs
G1–G4 and G5.0 belong to the superseded `../pre_process/`, while G5 steps 1–4 are this
directory. Because that is confusing, the port's files were renamed to their roles; only the
generated `g5_*.spv` kernels keep a prefix, since `AURORA_RESULTS.md` cites them by name.

**The directory is what tells you which route a file belongs to**, and the content agrees: Metal
AIR, `air2spir` and `--target-accelerator apple-m4` appear only under `../pre_process/`, while
`intel-pvc`, `spirv64` and the fork appear only here.

## ⚠️ What this is not

- **Our own build of the Mojo compiler, not an official release.** Modular does not support Intel
  GPUs and we do not expect that to change, so this is a community path someone has to maintain.
- One GPU tile. float32. The slice of the runtime these three programs use — no streams, events
  or graphs (18 functions).
- `-I` flags are still needed for the `max` / `layout` packages. They *do* build with the fork
  (`max.mojoc` in about 40 s), they are just not on the default import path.
- Aurora's **newer system image only** — the `uan-0007` / `uan-0008` login nodes and the
  `next-eval` queue — until the October 2026 system-wide update. The rest of the machine runs an
  image whose glibc is too old for Mojo.
- **Kernels are untuned for this hardware.** `03c` reaches about 7% of oneMKL on the same GPU.
  That kernel is hand-tuned for Apple and NVIDIA GPUs and Intel's compiler reports it spilling
  registers here; the identical kernel gets 19–65% of cuBLAS on Polaris. oneMKL on one Max 1550
  tile is about 1.5× cuBLAS on a Polaris A100, so the headroom is real and the gap is in the
  kernel, not in the backend. Tuning is work we have not done.

## 🖥️ Environment

- Anything that runs Mojo: `aurora-uan-0007` / `0008`, or `qsub -q next-eval`.
- PBS: `-A ModCon -q next-eval -l select=1 -l filesystems=flare:home`. No inline comments after
  `#PBS` values.
- Rebuild `fork_overlay.tar.gz` with `make_overlay.sh` after **any** change to the fork —
  the `modular` repo's `.gitignore` contains `target/`, which macOS git matches case-insensitively
  against `Mojo/lib/**/Target/IntelGPU/`, so `git status` and `git diff` hide the backend's sources.

## 📦 `mojo_intel_gpu` provenance

`../pre_process/` keeps its own identical copy of these bindings; the notes below apply to both.

- Upstream: <https://github.com/andomeder/mojo-intel-gpu>, commit `d3936ac` (2026-08-14), MIT
  license (`mojo_intel_gpu/LICENSE`). Tested upstream on Intel Arc B580 only.
- Compile-checked unmodified against Mojo 1.0.0 (`ed45d567`) on the Mac, 2026-09-14.
- Local changes, each marked `AURORA PATCH` in the source:
  1. `core/kernel.mojo`: pass a NUL-terminated copy of the kernel name to `zeKernelCreate`
     (a Mojo `String` isn't guaranteed to be NUL-terminated).
  2. `l0/loader.mojo`: load `libze_loader.so.1` first, then fall back to `libze_loader.so`.
  3. `l0/loader.mojo` + `core/context.mojo`: if `zeInitDrivers` is missing or finds no
     drivers, fall back to `zeInit(GPU_ONLY)` + `zeDriverGet`.
  4. `l0/loader.mojo` + `core/kernel.mojo`: `zeKernelSetIndirectAccess` binding and
     `Kernel.set_indirect_access(flags)`. Needed by G2 kernels that dereference pointers stored in
     buffers.
  5. `l0/loader.mojo` + `core/kernel.mojo`: `Kernel.set_arg_local(index, size)` binds OpenCL
     `__local` / shared-local-memory args (NULL value pointer). Needed by G3 barrier kernels.
  6. `l0/loader.mojo` + `core/context.mojo`: `zeCommandListAppendBarrier` binding and
     `IntelGPUContext.barrier()`. An immediate command list is **not in-order by default**, so
     dependent kernels appended back to back can execute concurrently. G1–G3 never saw this
     (one kernel, launched repeatedly); G4's 15-launch dependency chain did — later kernels read
     buffers their producer had not written yet. Orders on the device, no host round-trip.
- Patch 1 originally had an ASAP-destruction bug (the name buffer was freed before the C call); fixed
  2026-09-15 with `_ = name_c^` after the call.
