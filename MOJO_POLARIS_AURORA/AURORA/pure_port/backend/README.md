# 🛠️ The `spirv64` (Intel GPU) target

New and modified files for the open-source Mojo compiler, adding an Intel GPU target so that
`--target-accelerator intel-pvc` compiles Mojo GPU code to SPIR-V for Ponte Vecchio.

**Base commit:** `modular` @ `6417db28ceef430d067755e9826cec65e19f9333`
(`github.com/modular/modular`).

These files are laid out at the paths they occupy in that repo, so they can be read here and
applied by copying the tree over a clone at that commit. `../prepare_fork.sh` does exactly
that on Aurora, using `../fork_overlay.tar.gz` (the same files, packed). Rebuild the tarball
with `../make_overlay.sh` after any change.

## 📄 What each file does

**The target itself (new):**

| File | What |
|---|---|
| `Mojo/lib/Target/IntelGPU/IntelGPUTraits.{h,cpp}` | `TargetTraits` for `spirv64` — address spaces, pointer sizes, what the target supports |
| `Mojo/lib/KGENToLLVM/Target/IntelGPU/IntelGPULowering.cpp` | Lowers Mojo's GPU semantics — `global_idx` / `thread_idx` / `block_idx` to SPIR-V BuiltIns, `barrier()` to `OpControlBarrier` with `Workgroup` scope, shared memory to the right address space |
| `Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUBackend.cpp` | Drives LLVM's SPIR-V codegen and emits the kernel binary |
| `Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUAddressSpaces.{h,cpp}` | The address-space remap. Without it kernel arguments land in `Function` storage (private), so kernels run and write nothing — no fault, no wrong values |

**Modified, to register the target and let it through:**

| File | Change |
|---|---|
| `MODULE.bazel` | Enable LLVM's SPIR-V target in the build |
| `bazel/internal/cc-toolchain/macos_sysroot_repository.bzl` | Pin `MacOSX26.5.sdk` — the hermetic `ld64.lld` cannot read macOS 27 SDK `.tbd` files. Inert on Linux |
| `Mojo/lib/Target/TargetTraits.{h,cpp}` | Register `IntelGPU`; exempt base targets from the `--target-accelerator` MAX gate |
| `Mojo/stdlib/std/sys/{info,__init__}.mojo` | Target detection for the new accelerator |
| `Mojo/stdlib/std/_gpu/primitives/id.mojo` | Thread-index primitives for `spirv64` |
| `Mojo/stdlib/std/_gpu/host/_builtin_targets.mojo` | Add `intel-pvc` as a known accelerator |
| `Mojo/stdlib/std/gpu/{__init__,host/__init__}.mojo` | Expose the target through `std.gpu` (1.0-compatible surface) |
| `Mojo/stdlib/std/builtin/debug_assert.mojo`, `std/os/os.mojo`, `std/memory/_poison.mojo`, `std/collections/string/string.mojo` | Small fixes needed to compile the stdlib for this target |
| `max/mojo/max/gpu/sync/sync.mojo` | Synchronization primitives for the new target |

The kernel ABI this produces: every argument is `ptr addrspace(1) byref(T)`.

## ⚖️ License

The Modular repository is licensed under the **Apache License v2.0 with LLVM Exceptions**
(`LICENSE.upstream`, copied from the base commit). The files in this directory are either
verbatim copies or **modifications** of files from that repository, and are distributed under
the same license. Modified files are listed in the table above.

Apache-2.0 §4(b) requires modified files to carry prominent notices stating that they were
changed. Most files here carry an `AURORA PATCH` marker in their header saying so; this README
is the notice for all of them. This is not an official Modular release and is not endorsed by
Modular.

**Two items to settle before this is relied on** (tracked, not yet done):
`Mojo/lib/Target/IntelGPU/IntelGPUTraits.cpp` and `Mojo/stdlib/std/sys/__init__.mojo` are missing
their `AURORA PATCH` marker; and the six new `IntelGPU*` files still carry the upstream
`Copyright (c) 2026, Modular Inc.` header they were templated from, which should name their
actual authors instead.
