# 🧊 Preprocess route — superseded, frozen

This is **not** the port. It is an earlier approach, kept for the record, and it is finished:
no more work goes into it.

Before the Mojo compiler was open-sourced there was no way to make Mojo emit Intel GPU code.
The workaround here got Mojo-written kernels running on a PVC tile anyway, by **preprocessing
compiler output**:

```
unchanged .mojo
  -> mojo build --target-accelerator apple-m4    (Mojo's Metal backend -> AIR kernel IR)
  -> air2spir.py                                  (retarget AIR -> SPIR)
  -> clang -> llvm-spirv -> ocloc -device pvc
  -> hand-written Level Zero host code            (g1/g2/g3/g4_*_lz.mojo)
```

It worked — vector add, four matmul variants, and a 300-epoch MLP training loop all produced
correct results on one tile. But the kernels only ran because a separate hand-written host
program drove them; the original programs' own host code never ran. That is why it was replaced.

**The real port is `../`** — an `spirv64` backend inside the compiler, so the unmodified programs
run host code and all, with no preprocessing step and no hand-written host.

## 🎁 What it contributed

- **Correctness was settled early.** PVC and Intel's compiler handle Mojo's GPU semantics
  correctly — thread indexing, `barrier()`, shared memory, a 15-launch training loop reproducing
  its loss curve to ~8 significant figures.
- **A spec for the backend's output.** `air2spir.py` encodes how Mojo's GPU semantics map onto
  SPIR. The backend's last stage could reproduce that mapping instead of inventing it.
- **Reference kernels.** The backend's first output was diffed and run against the known-good
  `.spv` files from this route.
- **The Level Zero traps**, which the real runtime then had to handle: buffer/const/local argument
  binding, `zeKernelSetIndirectAccess` for kernels that follow stored pointers, and above all that
  a Level Zero **immediate command list is not in-order** — dependent launches overlapped and read
  buffers their producer had not written, giving a wrong result with no wrong values, only missing
  writes. `DeviceContext` presents an in-order stream, so the real runtime owes that ordering itself.

## 📁 Files

| Path | Rung | What |
|---|---|---|
| `air2spir.py` | G2+ | Retargets Metal AIR kernel IR → SPIR |
| `air_roles.py` | G5-lite | Maps hash-named AIR sidecars to stable role names by signature |
| `build_kernels.sh` | G5-lite | One command: `.mojo` → AIR → SPIR-V → `ocloc` check |
| `g1_vecadd.cl`, `g1_build_kernel.sh`, `g1_vecadd_lz.mojo` | G1 | OpenCL C kernel — Mojo host drives a PVC tile through Level Zero |
| `g2_*` | G2 | First **Mojo-written** kernel on PVC (unchanged `MOJO_CURRICULUM/02`) |
| `g3_*` | G3 | Four matmul variants + the oneMKL yardstick |
| `g4_*` | G4 | `04b` MLP training — 12 distinct kernels, 15 launches/epoch |
| `g5_check_air_on_aurora.sh` | G5.0 | Proves Aurora emits the kernel IR itself, byte-identical to the Mac |
| `*.air.ll` | — | Generated Metal AIR; regenerate with `build_kernels.sh` |
| `mojo_intel_gpu/` | — | Vendored Level Zero bindings (MIT). Same copy as `../pure_port/mojo_intel_gpu` |

`g1_run.pbs`, `g3_mkl_sgemm.cpp` and `g3_bench_torch_xpu.py` are also duplicated from `../` so this
directory can be deleted without breaking anything above it.

Results for every rung here, including the failures, are in `../AURORA_RESULTS.md`.

### Regenerating the `g4_*.air.ll` files (Mac, same command as G2)

`04b_train_mlp_gpu.mojo` emits **14** kernel sidecars for its 15 launches per epoch: the
generic kernels are instantiated once per `TensorLayout`. Three of the 14 (the SGD
instantiations) are byte-identical, so only **12** distinct kernels are built. The sidecar
filenames carry a mangling hash, not a role, so the mapping below was derived from the layout
strides baked into each body (N=256, D=2, H=16):

| Staged as | 04b launch | Identified by |
|---|---|---|
| `g4_fwd1.air.ll` | `K_fwd1` = `matmul_kernel[L_ND, L_DH, L_NH]` | A row stride 2, B row stride 16 |
| `g4_fwd2.air.ll` | `K_fwd2` = `matmul_kernel[L_NH, L_H1, L_N1]` | A row stride 16, B row stride 1 |
| `g4_dW1.air.ll` | `K_dW1` = `matmul_at_b_kernel[L_ND, L_NH, L_DH]` | C row stride 16, A row stride 2 |
| `g4_dW2.air.ll` | `K_dW2` = `matmul_at_b_kernel[L_NH, L_N1, L_H1]` | C row stride 1, A row stride 16 |
| `g4_da1.air.ll` | `K_da1` | only `matmul_a_bt_kernel` instantiation |
| `g4_db1.air.ll` | `K_db1` = `colsum_kernel[L_NH, L_1H]` | input row stride 16 |
| `g4_db2.air.ll` | `K_db2` = `colsum_kernel[L_N1, L_11]` | input row stride 1 |
| `g4_sgd.air.ll` | `K_sgdW1` / `K_sgdb1` / `K_sgdW2` / `K_sgdb2` | all four bodies byte-identical |
| `g4_biasrelu`, `g4_addb2`, `g4_dz2`, `g4_relugrad` | one launch each | single instantiation |

**If `04b` ever changes, re-derive this mapping** — the role names are ours, not Mojo's.

```bash
cd <scratch dir>
<MOJO_CURRICULUM>/.venv/bin/mojo build --emit asm --target-accelerator apple-m4 \
    <MOJO_CURRICULUM>/04b_train_mlp_gpu.mojo -o host.s
# then match each host__04b_*_<hash>.ll to a role by its stride constants (table above)
```
Generated 2026-09-15 from `04b_train_mlp_gpu.mojo`.

## 🔁 One-command build (G5.0 onward)

Since G5.0, Aurora emits the Metal AIR itself — byte-identical to the Mac's — so the build starts
from the `.mojo` source and no second machine is involved:

```bash
# on aurora-uan-0007, from MOJO_WORK/AURORA/gpu
bash build_kernels.sh curriculum/04b_train_mlp_gpu.mojo g4 g4_roles.map
bash build_kernels.sh curriculum/02_vecadd_gpu.mojo      g2          # no map: base names are unique
```

Needs `max` in `gpu/.venv` (`uv add "max==26.5.0"` — pin it; an unpinned add can move the compiler).

**Role names.** Mojo names each sidecar with a mangling hash, and a generic kernel instantiated per
`TensorLayout` yields several sidecars sharing a base name. `air_roles.py` identifies each by a
signature — base name, argument counts, and the layout strides compiled into the body — which is
stable across rebuilds where the hash is not. `<role>` becomes the `.spv` basename and the SPIR-V
entry point. If the source's kernel set changes, the signatures stop matching and the build **fails
with an error naming the offending kernels**, rather than binding a kernel to the wrong role. To
author or refresh a map:

```bash
python3 air_roles.py survey <dir-of-ll-files>    # prints a fillable map
```

The per-rung `g2_build_kernel.sh` / `g3_build_kernels.sh` / `g4_build_kernels.sh` remain as the
record of how G2–G4 were actually built (from Mac-generated `.air.ll` files).

## 📦 `mojo_intel_gpu` provenance

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
