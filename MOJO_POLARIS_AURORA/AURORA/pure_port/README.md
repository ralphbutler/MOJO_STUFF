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

| Program | Result | Aurora, one PVC tile | Mac M4 Max, stock Mojo |
|---|---|---|---|
| `02_vecadd_gpu.mojo` | 0 mismatches / 1,000,000 | 0.039 ms/pass | 0.102 ms/pass |
| `03c_matmul_coarse.mojo`, N=2048, as shipped (Apple-tuned geometry) | exact vs CPU reference | 1,480 GFLOP/s | 4,356 GFLOP/s |
| `matmul_tunable.mojo` — the same kernel, retuned for PVC | exact, all 4,194,304 cells | **7,782 GFLOP/s** | — |
| `04b_train_mlp_gpu.mojo`, 300 epochs | loss 2.1783555 → 0.0005614754 | 0.344 ms/epoch | 0.273 ms/epoch |
| `stencil.mojo`, 5-point Jacobi, 2048² | exact, all 8,388,608 cells | 631 GB/s | ~666 GB/s |

Aurora figures are means of three runs; quote **±2.5%** (node-to-node), not the within-job spread.
7,782 GFLOP/s is 36.8% of oneMKL SGEMM on the same tile and 86.8% of the identical kernel on an A100.
The Aurora-built compiler emits **byte-identical** SPIR-V to the Mac-built one. Both compilation
paths work: `mojo run` (JIT) and `mojo build` (standalone executables).

Full evidence, including the runs that failed and why, is in `../AURORA_RESULTS.md`; the
write-up is `../ASSESSMENT.md`; raw job logs are in `evidence/`.

## 🚀 Download and test

Needs an Aurora account and allocation. Runs only on the **newer system image**: login nodes
`aurora-uan-0007` / `0008` and the `next-eval` queue.

1. **Set your paths.** The scripts hard-code `W=/lus/flare/projects/ModCon/rbutler/MOJO_WORK` and
   `#PBS -A ModCon`. Change both to your project directory and allocation.
2. **Lay out the working directory** on Aurora. The jobs expect this directory at `$W/AURORA/gpu`
   and the sibling `MOJO_CURRICULUM` programs at `$W/AURORA/gpu/curriculum/`. Either copy it there
   yourself, or run `sync_to_aurora.sh` (set `AURORA_HOST` and `AURORA_DEST=$W/AURORA`), which
   lands it at `$W/AURORA/pure_port`, then `ln -s pure_port $W/AURORA/gpu`.
3. **Build** (from `$W/AURORA/gpu`):
   ```bash
   bash prepare_fork.sh     # uan-0007: clone modular @ 6417db28, apply fork_overlay.tar.gz
   qsub build_fork.pbs      # compile the fork (~51 min cold)
   bash build_runtime.sh    # uan-0007: libmojo_level_zero_rt.so -> $W/runtime/lib
   ```
4. **Run:**
   ```bash
   qsub e2e_run.pbs         # unmodified 02 / 03c / 04b via mojo run
   qsub e2e_aot.pbs         # the same three via mojo build
   qsub stencil.pbs         # the stencil
   qsub matmul_sweep.pbs    # the tuning sweep
   ```
   Compare against `evidence/`.

`stencil.mojo` needs no fork: on a Mac or Polaris run it with stock Mojo, `mojo run stencil.mojo`.

## 📁 Every file, by what you would use it for

**Where things run.** *Mac* = on a Mac with the fork or stock Mojo; *uan* = an Aurora login node
(`aurora-uan-0007` / `0008`); *qsub* = a batch job on `next-eval`. Every `.pbs` file is submitted
from `$W/AURORA/gpu` with `qsub <file>`.

### 1. Build the compiler and runtime — needed for everything else

| File | Where | What |
|---|---|---|
| `backend/` | — | The `spirv64` target: new and patched files for `modular` @ `6417db28`, at their repo paths. Read these; see `backend/README.md` |
| `fork_overlay.tar.gz` | — | The same files as `backend/`, packed. This is what actually gets applied |
| `prepare_fork.sh` | uan | Clone `modular` @ `6417db28` into `$W/modular`, unpack the overlay, point Bazel's cache at Lustre |
| `build_fork.pbs` | qsub | Build the fork (~51 min cold, ~90 s incremental), smoke-test it, re-emit one kernel and compare with the Mac build |
| `runtime/mojo_level_zero_rt.cpp` | — | The runtime: Level Zero implementations of the 18 `AsyncRT_*` functions `DeviceContext` calls |
| `build_runtime.sh` | uan | Compile the runtime to `$W/runtime/lib/libmojo_level_zero_rt.so` |
| `make_overlay.sh` | Mac | Only if you change the fork: repack `fork_overlay.tar.gz` from a fork checkout |
| `syntax_check_runtime.sh`, `ze_stub/` | Mac | Only if you edit the runtime: syntax-check it without Aurora. `ze_stub/` is stand-in Level Zero headers, never linked |
| `sync_to_aurora.sh` | Mac | Copy this directory to Aurora in one `ssh` stream. Set `AURORA_HOST` and `AURORA_DEST`; see step 2 of the quick start |

### 2. The main results — unmodified curriculum programs

| File | Where | What |
|---|---|---|
| `e2e_run.pbs` | qsub | **Start here.** Runs unmodified `02_vecadd_gpu`, `03c_matmul_coarse`, `04b_train_mlp_gpu` with `mojo run`. Expected output: `evidence/e2e_run2.txt` |
| `e2e_aot.pbs` | qsub | The same three built with `mojo build` into standalone executables, then run |
| `stencil.mojo` | any | 5-point Jacobi stencil, 2048², checks itself exactly. Plain Mojo, no Intel code: runs with stock Mojo on a Mac or Polaris |
| `stencil.pbs` | qsub | The stencil on Aurora, 3 runs under each register-file setting |

The curriculum programs are not in this directory. They come from the sibling `MOJO_CURRICULUM/`,
copied to `$W/AURORA/gpu/curriculum/`.

### 3. Tuning — how matmul went from 1,480 to 7,782 GFLOP/s

| File | Where | What |
|---|---|---|
| `matmul_tunable.mojo` | — | The curriculum's coarse matmul with its tile sizes (`BM BN BK TM TN`) as editable constants. The default values give a kernel identical to `03c` |
| `set_matmul_geometry.sh` | any | Rewrite those constants: `bash set_matmul_geometry.sh BM=64 BN=64 BK=16 TM=4 TN=4` (the winning geometry) |
| `matmul_sweep_configs.txt` | — | The eight geometries tried |
| `emit_matmul_tunable.mojo` | Mac | Compile one geometry to SPIR-V with the fork |
| `matmul_sweep_check.sh` | Mac | Compile and validate all eight before spending queue time |
| `matmul_sweep.pbs` | qsub | Every geometry × both register-file settings. Log: `evidence/matmul_sweep_8836109.txt` |
| `confirm_runs.pbs` | qsub | 3× repeats of the key cells for error bars, and the register-file setting on the three real programs |
| `launch_overhead.mojo`, `launch_overhead.pbs` | qsub | Cost per kernel launch, host vs device. Log: `evidence/launch_overhead_8836110.txt` |
| `ordering_test.pbs` | qsub | The three `MOJO_LZ_ORDERING` modes, for speed and correctness |

**Runtime settings** (environment variables read by `runtime/mojo_level_zero_rt.cpp`):

| Variable | Effect |
|---|---|
| `MOJO_LZ_BUILD_FLAGS` | Passed to Intel's GPU compiler. `-ze-opt-large-register-file` gives 256 registers per thread instead of 128. **Use it only when the compiler reports a register spill**: ×2.98 on matmul, −35% on streaming bandwidth, −3.2% on 04b |
| `MOJO_LZ_BUILD_LOG=1` | Print the compiler's register/spill report |
| `MOJO_LZ_ORDERING` | `barrier` (default), `inorder`, or `none`. `none` is diagnostic only: 04b page-faults the GPU |
| `MOJO_LZ_PROFILE=1` | Print per-launch host cost |

### 4. Vendor-library yardsticks (oneMKL SGEMM, for comparison)

| File | Where | What |
|---|---|---|
| `mkl_sgemm.cpp` | uan, then compute node | oneMKL SGEMM called directly. Build on uan: `icpx -fsycl -qmkl -O2 mkl_sgemm.cpp -o mkl_sgemm`; run on a compute node: `ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 ./mkl_sgemm 2048 50` |
| `bench_torch_xpu.py` | compute node | The same through PyTorch XPU: `module load frameworks`, then `python bench_torch_xpu.py 2048 50` |

### 5. History — how the backend was brought up (not needed to use it)

Before the runtime existed, kernels built by the backend were checked by hand-written host programs.
Kept as the record behind `../AURORA_RESULTS.md`; you do not need them to run the programs above.

| File | Where | What |
|---|---|---|
| `bringup_skeleton.mojo` | Mac | First kernel through the backend → `g5_add_one.spv` |
| `bringup_thread_ids.mojo` | Mac | Thread indexing, two ways → `g5_ids_a.spv`, `g5_ids_b.spv` |
| `bringup_barrier.mojo` | Mac | Shared memory + barrier, two ways → `g5_barrier_a.spv`, `g5_barrier_b.spv` |
| `emit_vecadd.mojo` | Mac | Curriculum vecadd kernel → `g5_vecadd.spv` |
| `emit_matmul.mojo` | Mac | Curriculum matmul kernels → `g5_check.spv` (N=256), `g5_coarse.spv` (N=2048) |
| `emit_mlp.mojo` | Mac | The 12 kernels of `04b` → `g5_fwd1`, `g5_fwd2`, `g5_dW1`, `g5_dW2`, `g5_da1`, `g5_biasrelu`, `g5_addb2`, `g5_dz2`, `g5_relugrad`, `g5_db1`, `g5_db2`, `g5_sgd` (`.spv`) |
| `g5_*.spv` | — | The compiled kernels above, SPIR-V binaries, kept as produced |
| `g5_*.spv.name` | — | The kernel's entry-point name inside the `.spv` (a long mangled Mojo name) |
| `g5_*.spv.args` | — | One line per kernel argument, `arg <i> buffer` etc., read by the host programs to bind arguments |
| `ocloc_check.sh` | uan | Does Intel's offline compiler accept the `.spv` files? |
| `run_vecadd.mojo`, `run_matmul.mojo`, `run_train_mlp.mojo` | qsub | Hand-written host programs that load those `.spv` files and check results |
| `run_job.pbs` | qsub | Runs one of them: `qsub -v MOJOFILE=run_matmul.mojo run_job.pbs`. Needs stock Mojo 1.0.0 in `$W/AURORA/gpu/.venv` (`uv init --no-workspace`, `uv add mojo==1.0.0`) |
| `mojo_intel_gpu/` | — | Level Zero bindings for Mojo (MIT, see its `LICENSE`), used only by the `run_*.mojo` programs |

### 6. Evidence

`evidence/` holds raw job output: `e2e_run1.txt`, `e2e_run2.txt` (the end-to-end runs),
`matmul_sweep_8836109.txt`, `launch_overhead_8836110.txt`. The index below says which job proves what.

## 🏷️ Reading the filenames

Names here say what a file **does**. The `gN_` prefixes in `../AURORA_RESULTS.md` are **rung
numbers** — chronological milestones, not routes — and they do not line up with the split: rungs
G1–G4 and G5.0 belong to the superseded `../pre_process/`, while G5 steps 1–4 are this
directory, as is G6 (tuning). Because that is confusing, the port's files were renamed to their
roles; only the generated `g5_*.spv` kernels keep a prefix, since `AURORA_RESULTS.md` cites them by
name. G6 renames, for reading `AURORA_RESULTS.md`: `g5_matmul_pvc` → `matmul_tunable`,
`g5_emit_matmul_pvc` → `emit_matmul_tunable`, `g6_set_geometry` → `set_matmul_geometry`,
`g6_configs` → `matmul_sweep_configs`, `g6_sweep` → `matmul_sweep`, `g6_sweep_check` →
`matmul_sweep_check`, `g6_launch` / `g6_launch_overhead` → `launch_overhead`, `g6_ordering` →
`ordering_test`, `g6_publish` → `confirm_runs`, `g6_stencil` → `stencil`, `g5_syntax_check` →
`syntax_check_runtime`, `g5_ze_stub` → `ze_stub`.

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
- **Not vendor-library speed.** The tuned matmul is 36.8% of oneMKL. The remaining gap is Intel's
  systolic matrix instructions (DPAS), which the backend does not emit. For a dense matmul, call
  the vendor library.
- **No trustworthy efficiency denominator.** `stencil.mojo` carries a naive copy kernel as a
  "bandwidth ceiling"; on the Mac the stencil beats it, so it is not a ceiling. Quote absolute
  GB/s only. A vectorised streaming benchmark is the missing piece.

## 📎 Evidence index — which job proves what

All ran on Aurora; job numbers are ALCF PBS jobs. Logs in `evidence/` where noted; every run is
written up in `../AURORA_RESULTS.md`.

| What | Evidence |
|---|---|
| Unmodified 02 / 03c / 04b on a PVC tile | source parity, 2026-09-16 — `evidence/e2e_run1.txt`, `e2e_run2.txt` |
| Compiler built on Aurora itself | 51 minutes; emits SPIR-V byte-identical to a macOS build |
| Matmul tuning sweep, 16 configurations, all exact | job 8836109 — `evidence/matmul_sweep_8836109.txt` |
| Launch-overhead measurement | job 8836110 — `evidence/launch_overhead_8836110.txt` |
| Ordering experiment (barrier / in-order / none) | job 8836165 |
| Confirmation runs, error bars, flag effect on real programs | job 8836259 |
| Stencil, 631 GB/s, exact on 8,388,608 cells | job 8836260 |
| Mac rows, stock Mojo 1.0.0, same files (incl. stencil) | run 2026-09-17 |
| Ahead-of-time `mojo build`: standalone executables, all three exact | job 8836303 |

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
