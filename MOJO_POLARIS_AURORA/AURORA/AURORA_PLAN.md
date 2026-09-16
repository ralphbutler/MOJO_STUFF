# 🌌 Mojo on Aurora (Argonne) — Plan

> **Malleable, not gospel.** Revise freely as we learn. Check boxes off as we go.
> Rationale lives in **`AURORA_PORT.md`**; the evidence for the 2026-09-14 pivot is in
> **`../WHY_NOT_GPUS_ON_AURORA.txt`** (UPDATE section). This file is the checklist view.

## 🧭 Fresh session? Start here (this dir is memory-blind)

A session rooted in this subdir does **not** auto-load the `MOJO_CURRICULUM` project
memory — this plan (+ `AURORA_PORT.md`) is the source of truth.

- **THE PROJECT (Ralph, restated 2026-09-15):** add an Intel GPU (`spirv64`) backend to the
  open-source Mojo compiler, so the same Mojo GPU programs that run on Polaris run **unchanged** on
  Aurora's GPUs (Intel Max 1550 / Ponte Vecchio). This is the main line (= **G5**), not a stretch goal.
  Done = `02_vecadd_gpu`, `03c` and `04b` run unmodified on PVC, host code and all.
  **Next step: G5-scope.** Direction rules: see "Direction discipline" in `HANDOFF.md`.
- **Why we looked again:** the Mojo compiler was open-sourced (Apache 2.0,
  `github.com/modular/modular`). Reading it showed an open backend interface and stdlib vendor
  hooks, while the NVIDIA/AMD/Metal code generators and MAX's GPU runtime (`DeviceContext`) are
  still closed.
- **Groundwork done, now frozen (G1–G5-lite ①, 2026-09-15):** a workaround route that uses only **prebuilt** tools and
  builds nothing from the repo: Mojo's **Metal** backend emits kernel IR → `gpu/air2spir.py` → SPIR-V
  → Level Zero from a Mojo host. See the claims section at the top of `AURORA_RESULTS.md`.
- **Phase 1 (CPU ladder L0–L5) is DONE** (2026-07-08) and stays below as the record plus the
  CPU baseline the GPU numbers get compared against.
- 📌 **Read `FROM_POLARIS.md` first** for ALCF operating lessons (storage, PBS, login/compute).
- **Canonical teaching source:** `../../MOJO_CURRICULUM` (on Mojo 1.0.0; its
  `UPDATE_TO_100.md` is the b2→1.0.0 migration record). Sibling `../POLARIS/` holds the GPU
  kernels we're trying to reproduce here.
- The `mojo-syntax` and `mojo-gpu-fundamentals` skills **must** be used when writing Mojo.
- **Working style:** Claude describes a step; Ralph runs it on Aurora and reports.
- ⚠️ **Name clash:** the rung **"L0"** (CPU toolchain smoke, `aurora_l0*.pbs`) is *not*
  **Level Zero**. In this plan Level Zero is written **"LZ"**.

## 🖥️ Machine profile

- **GPUs (the target now):** 6× Intel Data Center GPU Max 1550 (PVC) per node, each 2 tiles
  → **12 LZ devices** per node. 128 GB HBM2e per GPU. Native stack: Level Zero, SPIR-V,
  oneAPI (DPC++/SYCL, oneMKL). **fp64 is supported** on PVC (unlike consumer Arc).
- **CPU:** 2× Xeon CPU Max (Sapphire Rapids), 102 usable physical cores on the July image (Mojo sees
  104/208 on the new image), AVX-512, AMX, HBM.
- **OS:** SLES 15-SP4, **glibc 2.31** in July → Mojo ran only inside an apptainer image
  (Ubuntu 24.04). **New image: SLES 15-SP7, glibc 2.38 → Mojo 1.0.0 runs on the bare host (G0.2).**
- ⚠️ **Major system update in progress (ALCF docs, checked 2026-09-14):** SLES 15 **SP7**
  (kernel 6.4), Intel GPU drivers Agama 1146.78, **oneAPI 2026.1.0** (PE 26.181.0). Available
  since 2026-09-01 in the **`next-eval`** queue (up to 2,112 nodes; compile on UANs
  `aurora-uan-0007/0008`). **Full rollout expected 2026-10.** ALCF: major updates require
  recompiling. → **Do Phase 2 on the new image** so our work doesn't break in October.
  **Until the rollout:** anything that runs Mojo (build or run) goes on `aurora-uan-0007/0008` or
  in `-q next-eval` jobs. The `gpu/.venv` needs glibc ≥ 2.34 and won't run on old-image nodes
  (other UANs, `debug` queue). Editing files and running `qsub -q next-eval` work from any UAN.
  After the rollout, any node works; rebuild the venv if ALCF changes the image again.
- **Directories:** keep `MOJO_WORK/` as is (don't rename). Phase 1 = `MOJO_WORK/AURORA/`
  (container + b2 `.venv`); Phase 2 = `MOJO_WORK/AURORA/gpu/` (bare-host 1.0.0 `.venv`). Renaming
  would break the Phase 1 venv's absolute paths and the PBS scripts.
- **Scheduler:** PBS Pro; `-A ModCon -q debug -l filesystems=flare:home`; no inline
  comments after `#PBS` values. Compute nodes reach the internet via
  `proxy.alcf.anl.gov:3128`.
- **GPU facts from ALCF docs:** Level Zero headers and loader are part of the node image
  (`/usr/include/level_zero/ze_api.h`, link `-lze_loader`). Pin a tile with `ZE_AFFINITY_MASK`;
  **the syntax depends on the hierarchy mode (verified G0.3):** FLAT → `ZE_AFFINITY_MASK=0` (flat
  device index); COMPOSITE (default) → `ZE_AFFINITY_MASK=0.0` (gpu.tile). `ZE_FLAT_DEVICE_HIERARCHY=FLAT` exposes each
  tile as its own device. `xpu-smi` is a module (`module load xpu-smi`). CPU cores 0 and 52 are reserved for the OS.
- **Reference numbers (ALCF node-performance page, one tile):** SGEMM **21 TFlop/s**
  (≈ 21,000 GFLOP/s; oneMKL, peak size), DGEMM 14 TFlop/s. The A100 cuBLAS number from Polaris
  was 13,839 GFLOP/s at N=2048. Compare at matched N in G3.

## 🎯 Decision 2026-09-15 — goal and order

Goal = **feasibility + user-level portability** (same Mojo GPU program on Polaris and Aurora). Order:
**G4 (MLP) → G5-lite (DeviceContext-compatible LZ host + AIR generated on Aurora + one-command build) → small
PVC tuning sweep.** Updated at close (Ralph): **G4 → G5-lite → full port (G5, real `spirv64` backend)**; G5-lite designed so the backend slots in underneath; tuning fits wherever useful. Details: `HANDOFF.md`.

## 🎯 Direction reset 2026-09-15 (later, Ralph) — the backend is the project

The backend was Ralph's request from the start; the plan had wrongly relabeled it "stretch", and at the G3 gate
the options were framed against starting it. Now: **G5-scope → 🚦 gate (Question 4) → G5 backend.** The workaround route is
frozen (no more work unless Ralph asks). The standalone G5-lite ② host layer is **dropped**; its Level Zero
`DeviceContext` runtime is built inside G5. Before any step, say whether it advances the backend or is a
detour; detours need Ralph's yes. Gates are hard stops.

## 🔁 What "the same as Polaris" means — three parity levels

| Level | Meaning | Needs |
|---|---|---|
| **A — workload parity** | Same kernels (vecadd, matmul, MLP), written in Mojo, running on PVC, with numbers | Mojo→SPIR-V pipeline + our own LZ host shim (G1–G4) |
| **B — source parity ← THE PROJECT** | The *same `.mojo` GPU programs* run unchanged, host code and all | a `spirv64` compiler backend + a Level Zero runtime behind `DeviceContext` (G5) |
| **C — MAX parity** | `max generate` / in-process `LLM` on PVC (Polaris P4) | Closed MAX runtime — **not in our hands**. Out of scope. |

**Level A is complete** (2026-09-15, groundwork): vector add (G2), matmul (G3) and MLP training (G4) all
run on PVC from unchanged `MOJO_CURRICULUM` kernel sources. **Level B — source parity — is complete
(2026-09-16)**: 02, 03c and 04b run unmodified, host code and all, through our `spirv64` backend in the
open-source compiler and our Level Zero `DeviceContext` runtime. Caveats in `AURORA_RESULTS.md`
("SOURCE PARITY"): our fork, one tile, float32, `mojo run`, `-I` flags for `max`.

## 🪜 Phase 2 — the GPU ladder (mirrors Polaris P0→P4)

- [x] **G0 — Recon + GPU-capable toolchain.** ✅ DONE 2026-09-14. *(≈ Polaris P0)* Use the **new image**
      (`next-eval` queue, UANs `aurora-uan-0007/0008`).
      - [x] **G0.1 login recon (2026-09-14):** new image = SLES 15 SP7, **glibc 2.38** → Mojo
        should install on the **bare host**, so the container is now a fallback only.
        `llvm-spirv` = `$CMPLR_ROOT/bin/compiler/llvm-spirv` (not on PATH); `ocloc` =
        `/usr/bin/ocloc`; LZ headers/loader in `/usr`; **no `spirv-val`** (validate on the Mac
        via `brew install spirv-tools`, or rely on `ocloc`). Detail in `AURORA_RESULTS.md`.
      - [x] **G0.2 bare-host Mojo 1.0.0 install (2026-09-14):** `Mojo 1.0.0 (ed45d567)` in
        `MOJO_WORK/AURORA/gpu/.venv` (uv `--no-workspace`!). L0 smoke passes unchanged. No Intel
        accelerator. **The prebuilt LLVM has no `spirv` target** → G2 uses `--emit llvm` + oneAPI
        `llvm-spirv`; watch for IR version skew. Phase 1 venv accidentally went to 1.0.0; **decided
        (2026-09-14) not to restore b2.** Everything goes forward on 1.0.0. If a Phase 1 number
        ever needs re-measuring, port that file to 1.0.0 rather than reviving b2.
      - [x] **G0.3 GPU visibility (2026-09-14, job 8827177):** 6 GPUs (COMPOSITE) / 12 tiles (FLAT)
        via Level Zero; single tile works with FLAT+`ZE_AFFINITY_MASK=0` or COMPOSITE+`0.0`.
        `ocloc` driver 25.18.33578; **`llvm-spirv` is LLVM 22.1** (Mojo is LLVM 24 → G2 risk).
        `xpu-smi` needs `module load xpu-smi`. Mojo 1.0.0 runs on compute (104/208 cores).
        The container steps below were **not needed**; kept as fallback reference.
      - **Host GPU visibility** on a compute node: `xpu-smi discovery`, `sycl-ls`; with
        `ZE_FLAT_DEVICE_HIERARCHY=FLAT` expect 12 devices.
      - **Tool locations:** `ls /usr/include/level_zero/ze_api.h`, `ldconfig -p | grep ze_loader`,
        `which ocloc` (usually from the Intel compute runtime), `which llvm-spirv spirv-val`
        (probably inside the oneAPI compiler tree; not in the ALCF docs, so verify).
      - **Container + GPU — ALCF's documented recipe** (`docs/aurora/containers`): build with
        `apptainer build --fakeroot X.sif docker://intel/oneapi-hpckit` on a compute node. The
        image carries the Intel GPU user-space runtime; the host provides the kernel driver.
        Pass GPU env vars in with `APPTAINERENV_` prefixes:
        `APPTAINERENV_ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
        `APPTAINERENV_ONEAPI_DEVICE_SELECTOR=level_zero:gpu`,
        `APPTAINERENV_ZE_AFFINITY_MASK=0` (flat index under FLAT). No `/dev/dri` bind appears in their recipe
        (apptainer binds `/dev` by default).
        → *(Fallback only, since G0.2 made containers unnecessary.)* If ever needed, use `intel/oneapi-hpckit` rather than `ubuntu:24.04`. One image
        would then carry the new-enough glibc for Mojo, the LZ runtime, and the oneAPI tools.
        Verify inside: `ldd --version` (≥ 2.34), `sycl-ls` sees GPUs, `uv add mojo` works.
        Their note: don't `apt-get install` under `--fakeroot`.
      - Move the Aurora kit from `mojo 1.0.0b2` → **`mojo 1.0.0`** (pin in `pyproject.toml`)
        so it matches the open-sourced compiler. Re-run L0 smoke to confirm.
- [x] **G1 — GPU is live, from Mojo.** ✅ PASS 2026-09-15 (job 8827488): 0/1M mismatches, 0.00561 ms/pass (A100: 0.00766). *(≈ Polaris P1, `02_vecadd_gpu`)*
      A Mojo host program calls LZ via FFI and runs a **SPIR-V vector-add kernel** on one PVC
      tile. The kernel may be OpenCL C (`clang -target spirv64` / `ocloc`) at this rung; the
      point is to prove the **host side** (context, device, module, kernel, buffers, queue).
      Reuse/adapt `mojo-intel-gpu`'s LZ bindings (tested on Arc, never on PVC).
      Pass = 0 mismatches over 1M elements.
      *Drafted 2026-09-14 (see `pre_process/README.md`):* vendored `mojo-intel-gpu` @ `d3936ac`
      with 3 `AURORA PATCH`es; `g1_vecadd.cl` + `g1_build_kernel.sh` (uan-0007) →
      `g1_vecadd.spv`; `g1_vecadd_lz.mojo` mirrors `POLARIS/02` (N=1M, BLOCK=256, 100 iters);
      `g1_run.pbs` on `next-eval`. Compile-checked on the Mac (Mojo 1.0.0); not yet run.
- [x] **G2 — A Mojo-written kernel on PVC (the key spike).** ✅ PASS 2026-09-15 (job 8827519): unchanged `MOJO_CURRICULUM/02` kernel via Metal AIR → `air2spir` → SPIR-V; 0/1M mismatches, 0.00664 ms/pass.
      Replace G1's OpenCL C kernel with one **written in Mojo**:
      `mojo build --emit llvm` (plus `--target-triple` experiments) → retarget IR to `spir64`
      → `llvm-spirv` → `spirv-val` → load in the G1 host.
      Start with a pure function (index passed as an argument, no GPU built-ins).
      *(G0.2 finding: the prebuilt LLVM has no SPIR-V target, so translation must happen in
      `llvm-spirv`, not Mojo. Check LLVM IR/bitcode version compatibility with oneAPI's
      `llvm-spirv` first thing.)*
      **Route found 2026-09-14 (Mac): reuse Mojo's Metal backend.**
      `mojo build --emit asm --target-accelerator apple-m4` writes each GPU kernel as a small,
      optimized `.ll` (triple `air64`). AIR descends from SPIR and uses the same address spaces,
      so `gpu/air2spir.py` only rewrites the shell (triple, `spir_kernel`, thread params →
      `get_group_id`/`get_local_id`/`get_local_size`). The body uses no LLVM-24-only syntax, which
      sidesteps the version-skew risk. The kernel source is `MOJO_CURRICULUM/02_vecadd_gpu.mojo`
      **unchanged**. Buffer args arrive as pointers to TileTensor structs, so the host passes
      8-byte USM holders (`gpu/g2_vecadd_lz.mojo`). Drafted + compile-checked; not yet run.
      Risks for G3+: threadgroup (shared) memory, `barrier()`, WARP/SIMD sizes (Apple 32 vs PVC
      sub-groups 16/32), any `@air.*` math intrinsics, no fp64 on the Metal path.
      IR cleanup and `spirv-val` can be iterated **on the Mac**; only execution needs Aurora.
      **Tells us:** does Mojo-generated IR become valid SPIR-V (no stray libc calls, host
      intrinsics, assert traps)?
- [~] **G3 — Real workload numbers.** 2026-09-15 (job 8827590): all 4 kernels PASS (check FULL 0/65,536); naive 2,449 / tiled 2,177 / coarse 1,903 GFLOP/s (A100: 2,659 / 4,137 / 8,966). oneMKL yardstick 21,148 GFLOP/s (job 8827619) → Mojo kernels at 9–12% of oneMKL on PVC vs 19–65% of cuBLAS on A100. Decision gate superseded by the 2026-09-15 direction reset: the backend (G5) is the project. *(≈ Polaris P2 + P3b)*
      Port `03a` naive / `03b` tiled / `03c` coarse matmul to the G2 path; `03d` correctness
      check. Report GFLOP/s on one PVC tile vs **oneMKL** (the cuBLAS analog), vs A100
      (8,966 GFLOP/s coarse), and vs Aurora CPU (419 GFLOP/s, L4).
      *Prep 2026-09-15 (Mac):* all four curriculum kernels emit Metal IR. `03a` needed no
      `air2spir` change. `03b`/`03c`/`03d` add `@air.wg.barrier` (→ OpenCL `barrier(CLK_LOCAL_MEM_FENCE)`)
      and `addrspace(3)` shared-memory globals (→ `__local` kernel args bound by size, PVC SLM
      131 KB). The NVIDIA (`.ptx`) and AMD (`.amdgcn`) sidecars are Modular GPU *assembly*, not IR,
      so Metal AIR remains the only IR route. **Open risk: does IGC truly synchronize on
      `barrier()` for GPU work groups?** `03d` at N=256 with a full CPU check answers it.
      Kit: `gpu/g3_*`. Compile-checked; not yet run.
      🚦 **Decision gate:** is the hand-built pipeline good enough (stay at parity A), or
      commit to G5 (parity B)?
- [x] **G4 — The money shot: MLP training on PVC.** ✅ PASS 2026-09-15 (job 8828472). *(≈ Polaris P3a)*
      Unchanged `04b_train_mlp_gpu` on the G2/G3 path: 12 distinct kernels, 15 launches/epoch, 300
      epochs. Loss curve reproduced to ~8 significant figures at every printed epoch
      (2.1783555 → 0.0005614754 vs reference 2.17835617 → 0.000561475754); **0.366 ms/epoch**
      (A100: 0.116, but our host pays an explicit barrier per launch — see the caveat in
      `AURORA_RESULTS.md`).
      Two earlier runs failed and are worth remembering: the loss came out wrong because a Level
      Zero **immediate command list is not in-order**, so the 15 dependent launches overlapped
      (AURORA PATCH 6 adds `zeCommandListAppendBarrier`). G1–G3 could not expose this — they launch
      one kernel repeatedly. **G5's Level Zero runtime must provide `DeviceContext`'s in-order stream
      semantics itself.**
- [x] **G5.0 — Kernel IR generated on Aurora (Mac step removed).** ✅ 2026-09-15.
      Mojo's **Linux** build ships the Metal code generator (`apple-m1`…`apple-m5` all listed), so
      `mojo build --emit asm --target-accelerator apple-m4` works on uan-0007 and produces
      **byte-identical** IR to the Mac's — verified against the very file G2 ran, plus 04b's 14
      sidecars. Required `uv add "max==26.5.0"` in `gpu/.venv` (pinned; Mojo stayed 1.0.0). The
      pipeline is now single-machine. Check script: `gpu/g5_check_air_on_aurora.sh`.
- [~] **G5-lite — `DeviceContext`-compatible host layer + one-command build.**
      - [x] **AIR generated on Aurora** (G5.0 above).
      - [x] **One-command build 2026-09-15:** `build_kernels.sh <source>.mojo <prefix> [roles.map]`
            runs source → AIR → SPIR-V → `ocloc`; 12/12 of 04b's kernels built and validated.
            `air_roles.py` maps hash-named sidecars to stable roles by signature (base name, arg
            counts, layout strides) and **fails loudly** if the source's kernel set changes, rather
            than silently mismapping. It independently reproduced the hand-derived G4 mapping.
      - ⏹ ~~**The host layer**~~ — **dropped as a standalone workaround step (2026-09-15).** The Level
      Zero runtime behind `max.gpu.host.DeviceContext`, **including in-order stream semantics** (the
      G4 lesson), is built inside G5 instead.
- [x] **G5-scope — ✅ finding delivered 2026-09-15; 🚦 gate (Question 4) pending.** Cost: ~1 session,
      15.5 min build after one SDK fix (macOS 27 `.tbd` vs hermetic lld). Open compiler (`1.2.0.dev0`,
      LLVM 24) builds on the Mac with the SPIR-V target. **Answer: GPU lowering is open and
      target-parameterized** (stdlib vendor branches + C++ target hooks); closed = the vendor backend
      implementations and the `AsyncRT_Device*` GPU runtime behind `DeviceContext`. Revised estimate
      ≈ 5–8 months. Detail: `AURORA_RESULTS.md` "G5-scope". Original brief below.
      Build the open compiler from source (LLVM with the SPIR-V target enabled), read the **Host**
      backend's `TargetTraits`/`TargetLowering`/`TargetBackend` as the only worked example, and answer
      the question that sizes the work: **is Mojo's GPU lowering (`global_idx`, `barrier`,
      shared-memory address spaces) in the open tree, or closed?** Rough estimates: open → ≈ 4–6
      months; closed → ≈ 9–15 months, with `air2spir.py` as the spec. Deliverable: a one-page finding
      with file references, a revised estimate and a proposed build order. Detail in `HANDOFF.md`.
      🚦 **Gate (hard stop): ask Question 4 and wait for Ralph.**
- [ ] ▶️ **G5 — THE PROJECT: source parity via a `spirv64` compiler backend.** Build order in
      `AURORA_RESULTS.md` ("Revised estimate and build order").
      - [~] **Step 1 — target skeleton (2026-09-15):** `IntelGPU` Traits/Lowering/Backend in the clone.
        `compile_info` for `spirv64` emits a `spir_kernel` → SPIR-V binary that passes `spirv-val`.
        ✅ IGC (`ocloc -device pvc`) builds it on uan-0007. **Step 1 done.**
      - [~] Step 2 — stdlib intrinsics + kernel arg ABI. **Done (Mac):** `is_intel_gpu`, `id.mojo`
        branches, `barrier` (max, untested); ids + shared memory + barrier kernels → valid SPIR-V.
        ✅ IGC builds all 4 (variant A kept). ✅ 1.0 compat `std.gpu` re-exports (Ralph chose (a)).
        ✅ Kernel ABI: every arg `ptr addrspace(1) byref(T)`. ✅ Unchanged curriculum 02 kernel compiles
        via the fork (open `max`/`layout` source) → `g5_vecadd.spv`. **Pending:** stdlib GPU target entry.
      - [~] Step 3 — run on PVC. Run 1 (job 8829357): kernel created + launched, **no writes**
        (Function-storage data pointers) → backend addrspace 0→1 remap added.
        ✅ **Run 2 PASS (job 8829448): unchanged curriculum 02 kernel via our backend, 0/1M, 0.00633 ms/pass.**
        03c/03d: stdlib no-print/no-heap spots now treat Intel like Apple → `g5_check.spv`/`g5_coarse.spv`
        valid (barrier + Workgroup shared memory). ✅ **PASS (job 8829553): check FULL 0/65,536; coarse
        0/260, 1,856 GFLOP/s** (Metal route 1,903).
      - [x] Step 3 (rest) — ✅ **04b PASS (job 8829859): forward check 0 wrong, loss curve identical to G4
        at every printed epoch, 0.370 ms/epoch.** Step 3 closed: 02, 03c, 04b kernels all via our backend.
      - [ ] ▶️ Step 4 — whole programs. Decided (Ralph): runtime = C++ Level Zero implementation of the needed
        `AsyncRT_*` subset; fork built on a `next-eval` node; stdlib target entry on the Mac meanwhile.
        - [x] 4a fork on Aurora ✅ **job 8829889: built in 51 min, hello runs, `intel-pvc` listed, and the
          Aurora-built `g5_vecadd.spv` is byte-identical to the Mac's.** (Rebuild pending for the 4b patches.)
        - [x] 4b stdlib: Intel GPU target entry, vendor, `has_intel_gpu_accelerator()`, MAX-gate exemption →
          `mojo build --target-accelerator intel-pvc` compiles unchanged 02/03c/04b and emits valid kernel
          sidecars (02's = the PVC-passing kernel). Runtime surface measured: **18 `AsyncRT_*` functions**.
        - [ ] 4c `max` built with the fork
        - [~] 4d C++ Level Zero `AsyncRT_*` runtime — **written + compile-checked**
          (`gpu/g5_runtime/mojo_level_zero_rt.cpp`, 18 entry points, SPIR-V-derived arg sizes, barrier
          per launch). ✅ builds on Aurora with icpx, 18 symbols. **E2E run 1: all three unmodified
          programs PASS** (02 0/1M, 03c C[0,0]=4096, 04b loss curve exact) but ~15 ms/launch because
          `enqueue_function` reloads the kernel each call → compiled-kernel cache added.
          ✅ **Run 2: PASS at harness speed — 0.0403 ms/pass, 1,489 GFLOP/s, 0.354 ms/epoch** (04b beats
          our own harness). **Source parity (level B) reached.**
      - [ ] Step 5 — done = 02, 03c, 04b unmodified via `DeviceContext`.
      Original brief:
      In a fork of the OSS compiler: `TargetTraits` / `TargetLowering` / `TargetBackend`
      for `spirv64` (map `global_idx` etc. to SPIR-V built-ins, `spir_kernel` entry points,
      address spaces), an `IntelPlugin` for `std/_plugin`, and a Level Zero runtime exposing the
      `max.gpu.host.DeviceContext` API with in-order streams. Pass = `MOJO_CURRICULUM`
      `02_vecadd_gpu`, `03c` and `04b` run **unmodified** on PVC, host code and all.
- [ ] **(later) Multi-tile / multi-GPU** — 12 tiles per node; Polaris never went past one A100.

## ⚠️ Phase 2 risks

- **Container ↔ GPU:** Mojo needs glibc ≥ 2.34 (container), and the GPU driver lives on the
  host. If GPU passthrough fails, fallbacks are: (a) compile kernels to SPIR-V in the
  container and launch them with a host-side LZ program; (b) build a static/older-glibc
  Mojo from the OSS source.
- **Compiler fork maintenance** (G5): no upstream compiler PRs until ~end 2026; heavy churn.
- **A forked compiler can't build `max`**, and in 1.0.0 `parallelize` lives in
  `max.algorithm` → the CPU ladder files may need the prebuilt compiler.
- **G5 needs LLVM/MLIR C++ skills**, and the open tree has no GPU backend to copy (only
  Host).
- **`mojo-intel-gpu` is untested on PVC:** expect multi-tile device enumeration issues.
- **MAX inference on PVC (parity C) is not reachable.** For an inference comparator use
  PyTorch-XPU or vLLM on Aurora.

## ✅ Phase 1 — the CPU ladder (DONE 2026-07-08; kept as record + CPU baseline)

- [x] **L0 — Toolchain feasibility.** (`mojo_l0.o8657258`) Bare-host install impossible
      (wheels need glibc ≥2.34; Aurora 2.31) → apptainer Ubuntu 24.04 built on a compute
      node. `has_accelerator=False` (expected: no Intel backend in prebuilt Mojo),
      `simd f32=16`, `f64=8`, 102 phys / 204 log cores. Runs:
      `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run <file>`.
- [ ] **L1 — Scalar baseline.** Not run separately (serial numbers are folded into L3/L4).
- [x] **L2 — Explicit SIMD.** (`mojo_run.o8657612`) `00_simd_type.mojo` unchanged on b2;
      SIMD semantics correct (hardcoded width 4; native W=16 shown at L0/L3).
- [x] **L3 — Parallel scaling.** (`mojo_run.o8657706`) `parallelize` 11.5× at 102 cores,
      but peak only ~176 GB/s (saturates ~32 workers); single-core 15 GB/s.
- [x] **L4 — CPU matmul.** (`mojo_run.o8657642`) N=1024: naive 0.43 → SIMD 14.0 →
      parallel **419.4 GFLOP/s**. ← the CPU baseline for G3.
- [~] **L5 — stretch.** L5a NUMA first-touch (`mojo_run.o8657728`) did **not** lift the
      176 GB/s ceiling (`parallelize` has no affinity control). AMX / tiled matmul / Python
      interop not done.

## 📦 Files staged here

Container kit: `ubuntu2404.sif` (on flare) · `aurora_build.pbs` · `aurora_l0.pbs` ·
`aurora_run.pbs` (`-v MOJOFILE=…`) · `aurora_l0_smoke.mojo` · `aurora_probe.pbs` (archival) ·
`aurora_setup.sh` (login prep).
CPU material: `00_simd_type` · `01_vecadd_safe` · `02_vecadd_parallel` · `02b_vecadd_numa` ·
`03e_matmul_cpu`.
GPU material: the backend is in `pure_port/` (see `pure_port/README.md`); the superseded
Metal-AIR route is in `pre_process/` (see `pre_process/README.md`) — G1 vecadd · G2
Mojo-written kernel · G3 matmul + oneMKL yardstick · G4 MLP training.

## 🧾 Decisions

- **2026-09-14 — Kernel source = `../../MOJO_CURRICULUM` (Mojo 1.0.0),** not `../POLARIS/`.
  The Polaris copies are b2 syntax and won't compile on 1.0.0 (which the open compiler matches).
  Use `POLARIS/` only for the A100 numbers and job recipes.
- **2026-09-14 — Single tile through G4.** Standard env: `ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0`
  (FLAT is what ALCF's frameworks and container recipes use). Polaris used one A100, so
  one tile keeps the comparison fair. Multi-tile comes after G4.
Reference: `AURORA_PORT.md` · `AURORA_RESULTS.md` · `FROM_POLARIS.md` · `REPORT_*.md`
(Phase 1 reports).

## ❓ Open questions — and WHEN Claude must raise them

Claude: don't answer these by guessing. Ask Ralph at the trigger point, one at a time.

| # | Question | Raise when | Resolved by |
|---|---|---|---|
Resolved: ~~Q4 who does the G5 C++ work~~ → **Claude writes it, Ralph reviews** and runs Aurora steps
(Ralph, 2026-09-15, at the G5-scope gate). ~~kernel source~~ and ~~single tile~~ (see Decisions). ALCF container recipe found
(see G0). ~~Q2 tool locations + glibc~~ → answered by G0.1 (2026-09-14): glibc 2.38,
`llvm-spirv` and `ocloc` present, no `spirv-val`. ~~Q3 Arc card~~ → **Aurora only for now** (Ralph, 2026-09-15); SPIR-V translation/inspection runs on uan-0007 (no queue), only execution waits on `next-eval`. ~~Q1 bare-host Mojo + GPU visibility~~ → answered by
G0.2/G0.3 (2026-09-14): yes to both, no container needed.

Phase 1 questions (OS/glibc, proxy, `qsub -I`, `uv`, PBS values) are all resolved — see
`AURORA_PORT.md`.
