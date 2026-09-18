# 🌌 Aurora Mojo Port — Results & Experience Log

Running, chronological log of the Aurora (Argonne, Intel CPU) Mojo bring-up, kept so the
full experience is reportable later. Companion to `AURORA_PLAN.md` (checklist) and
`AURORA_PORT.md` (rationale). Polaris analog: `../POLARIS/REPORT_RESULTS.md`.

> **2026-09-14:** everything below through L5a is **Phase 1 (CPU)**. Phase 2 (GPU, G0–G5)
> results get appended as new dated sections at the end of this log.

## 📏 What we can and can't claim (as of 2026-09-15, after G4)

Keep this section current. Any external statement should fit inside it.

### ✅ Can claim
- Using **prebuilt Mojo 1.0.0**, a **GPU kernel written in Mojo** ran correctly on an **Aurora Intel Max
  1550 tile**: `vecadd_kernel` from the unchanged `MOJO_CURRICULUM/02_vecadd_gpu.mojo`, 0 / 1,000,000
  mismatches (job 8827519). That's the same kernel code that ran on Polaris' A100.
- A **Mojo host program** can drive Aurora's GPUs directly through Level Zero (G1, G2).
- Mojo 1.0.0 installs and runs on the **bare host** of Aurora's new image (glibc 2.38), with no
  container (G0.2).
- *(Added after G3, 2026-09-15)* The unchanged `MOJO_CURRICULUM` **matmul** kernels (naive, tiled, coarse)
  run **correctly** on a PVC tile, including Mojo `barrier()` and shared (threadgroup) memory: full check
  0/65,536 at N=256, sampled checks at N=2048. Measured N=2048 throughput: naive 2,449, tiled 2,177,
  coarse 1,903 GFLOP/s, vs oneMKL SGEMM 21,148 on the same tile (one run each).
- *(Added after G4, 2026-09-15)* The unchanged `MOJO_CURRICULUM/04b_train_mlp_gpu.mojo` **MLP training
  loop** — forward and backward, 12 distinct kernels, 15 launches per epoch, 300 epochs — runs on a PVC
  tile and **reproduces its loss curve to ~8 significant figures at every printed epoch** (2.1783555 →
  0.0005614754 vs the reference 2.17835617 → 0.000561475754), at 0.366 ms/epoch (A100: 0.116).

- *(Added after G5 step 3, 2026-09-15)* An **Intel GPU (`spirv64`) target we added to the open-source Mojo
  compiler** compiled the unchanged `MOJO_CURRICULUM/02_vecadd_gpu.mojo` kernel to SPIR-V (via LLVM's in-tree
  SPIR-V backend), and it ran correctly on a PVC tile: 0 / 1,000,000 mismatches (job 8829448). The host side
  was still our Level Zero test harness, not `DeviceContext`.
- *(Added 2026-09-15)* The same backend compiles the unchanged coarse matmul kernels (`03c`/`03d`, shared memory +
  `barrier()`) and they run correctly on a PVC tile: FULL check 0 / 65,536 at N=256, 1,856 GFLOP/s at N=2048
  (job 8829553). That is the same speed as the Metal-route kernel (1,903, one run each).
- *(Added 2026-09-16 — the headline)* **The unchanged `MOJO_CURRICULUM` GPU programs 02, 03c and 04b run on an
  Aurora PVC tile with no source changes at all, host code included**, compiled by the `spirv64` backend we
  added to the open-source Mojo compiler and executed through our Level Zero implementation of
  `max.gpu.host.DeviceContext`. Correctness: vecadd 0/1,000,000; matmul `C[0,0]` exact; 04b's loss curve
  2.1783555 → 0.0005614754, matching the NVIDIA/Apple reference. Speed: 0.0403 ms/pass, 1,489 GFLOP/s,
  0.354 ms/epoch — the last of which beats our own hand-written Level Zero harness.
- *(Added 2026-09-15)* The same backend compiles all 12 kernels of the unchanged `04b_train_mlp_gpu.mojo`, and
  300 epochs of training on a PVC tile reproduce the loss curve digit-for-digit (2.1783555 → 0.0005614754,
  job 8829859). **All three curriculum GPU workloads now run from our backend's kernels**; the host code is
  still our Level Zero harness.

### ❌ Can't claim (not true, or not shown)
- *(G5, revised 2026-09-16)* That Mojo **officially** supports Intel GPUs, or that this is finished. It is our
  fork of the open-source compiler plus an 18-function Level Zero runtime, covering what 02/03c/04b use:
  one tile, float32, `mojo run` (JIT; `mojo build` untested), and `-I` flags still needed because `max` is
  not built as a package with the fork. No multi-tile, no fp64, no streams/events/graphs, nothing upstream.
- That Mojo **supports** Intel GPUs. It has no Intel/SPIR-V/Level Zero backend, and
  `has_accelerator()` is False. We use a workaround route through Mojo's **Metal** backend.
- That the **open-source release** is what made it work. The release prompted the re-look and showed
  what's still closed, but the working pipeline uses only prebuilt tools (Mojo 1.0.0 with its closed Metal
  backend, Intel oneAPI, community `mojo-intel-gpu`) plus our `air2spir.py`. The key clue came from the
  prebuilt compiler's `mojo build --help`. *(G5-scope, 2026-09-15: the open compiler has since been built
  from source on the Mac, for backend work. The GPU pipeline that runs does not use it.)*
- That we "tried Mojo GPU on Aurora in July and failed." In July we assessed the stack, ruled GPUs out,
  and ran CPU-only.
- That **Polaris programs run unchanged** on Aurora. The host side is custom Level Zero code, not
  `DeviceContext`.
- **Performance parity** with Polaris or with vendor BLAS. On PVC the Mojo kernels reach 9–12% of oneMKL
  (the A100 reached 19–65% of cuBLAS with the same kernels), untuned for PVC, one run each.
- Anything beyond **vector add + matmul + this MLP training loop, float32, one tile, one node**. Math
  functions, fp64 and multi-tile are untested.
- That the **ms/epoch comparison with the A100 is like-for-like.** Our host appends an explicit Level Zero
  barrier after each of the 15 launches (CUDA streams are in-order for free), so the 3.2× gap at N=256,
  H=16 is host overhead on tiny kernels, not kernel speed.
- **Production** status. It ran on the `next-eval` preview image, and it depends on an undocumented
  Metal intermediate file that a Mojo release could change.
  *(No longer a two-machine pipeline: since G5.0 Aurora emits the kernel IR itself, and G4 was
  re-run end to end from source-built kernels with identical results — job 8828588.)*

### 🗣️ Suggested wording
> "Using prebuilt Mojo 1.0.0, we ran a Mojo-written GPU kernel correctly on an Aurora Intel Max 1550
> tile. Mojo still has no Intel GPU backend; we compile the kernel through Mojo's Metal backend,
> translate the result to SPIR-V, and launch it through Level Zero from a Mojo host program.
> Demonstrated for vector add, matrix multiply (including shared-memory/barrier kernels), and a full MLP
> training loop — forward and backward — which reproduces its reference loss curve to float32 print
> precision on one tile. Matmul throughput is well below oneMKL and not yet tuned for PVC."

> **Note on `mojo_*.o<jobid>` names:** each rung below cites the PBS **job-output log**
> filename that was `cat`'d at the time (e.g. `mojo_run.o8657642`). These are the actual files
> as they existed during the work; they may be cleaned up / deleted later, but the names are
> recorded here as the provenance of each result.

---

## 🧭 Login-node recon — 2026-07-08

| Item | Finding |
|---|---|
| OS / libc / kernel | **SLES 15-SP4**, **glibc 2.31**, kernel 5.14.21 |
| System Python | 3.6.15 (`/usr/bin/python3`) — too old for Modular wheels (need ≥3.9) |
| Module stack | spack "unified" stack; `gcc/13.4.0` default; **no conda/frameworks module**; `apptainer/1.2.5` present |
| uv | absent → `curl \| sh` install works; installs to `~/.local/bin` |
| Outbound net (login) | **direct https OK, no proxy needed** |
| Scheduler | PBS Pro 2022.1.7; `debug` queue live; `qsub -I` available |
| Live project FS | **ModCon** on `/lus/flare` — 54.1T used / **1000T** quota |
| Dead FS trap | `candle_aesp_CNDA` on `/lus/flare` = 60.67T used / **1M** (zeroed). Old `~/.cache` symlink pointed here → repointed to a live ModCon dotcache dir. |
| Work/install root | `/lus/flare/projects/ModCon/rbutler/MOJO_WORK/AURORA` |

## 🧱 L0 BLOCKER #1 — glibc floor (core finding)

`uv add mojo` fails:

```
Distribution `mojo==1.0.0b2` ... doesn't have a source distribution or wheel for the
current platform. You're on Linux (`manylinux_2_31_x86_64`), but `mojo` (v1.0.0b2)
only has wheels for: manylinux_2_34_aarch64, manylinux_2_34_x86_64, macosx_13_0_arm64
```

**PyPI wheel survey (all linux x86_64 wheels, every version):**

| Package | Versions surveyed | Platform tag |
|---|---|---|
| `mojo` | 0.25.6.0 … 1.0.0b2 (7) | **all `manylinux_2_34_x86_64`** |
| `mojo-compiler` | 0.25.6.0 … 1.0.0b2 (7) | **all `manylinux_2_34_x86_64`** |
| `max` | 25.3.0 … 26.4.0 (10) | **all `manylinux_2_34_x86_64`** |
| `modular` | — | **no linux x86_64 wheel at all** |

**Conclusion:** Every Modular Mojo-compiler wheel needs **glibc ≥ 2.34**; Aurora has **2.31**.
No older/more-portable build exists to pin to. **There is no pip-installable Mojo compiler
for Aurora's bare userspace.** (Contrast: the Polaris port used the *same* `mojo==1.0.0b2`
and it worked — Polaris login therefore has glibc ≥ 2.34. Aurora is the older userspace.)

This is a legitimate evaluation result for the "switch Argonne's stack to Mojo/MAX" question:
on Aurora, Mojo is not installable without a newer-glibc container.

## 📦 Pivot — containers (apptainer)

Aurora kernel (5.14) is new enough; only the **userspace glibc** is short. That is the
canonical container use case, and it does **not** compromise the CPU-scaling thesis
(AVX-512/SIMD + `parallelize` run at native speed inside apptainer — only libs come from
the image).

**Probe (login node, 2026-07-08):**
- `apptainer/1.2.5` loads; `docker://ubuntu:24.04` blobs pulled + unpacked, but SIF creation **failed**:
  ```
  ERROR : Failed to create user namespace: maximum number of user namespaces exceeded,
          check /proc/sys/user/max_user_namespaces
  INFO  : ... apptainer-suid, or compile with ./mconfig --with-suid
  ```
- → **Login node blocks unprivileged user namespaces**, and apptainer is not setuid-installed.

**Resolution path (from ALCF Aurora docs, confirmed 2026-07-08):**
- Apptainer **builds/pulls must run on a compute node** — login nodes block unprivileged user
  namespaces by design; compute nodes allow them. (So the login-node probe failure is expected.)
- **Aurora compute nodes have proxied internet:** `http_proxy=https_proxy=ftp_proxy=http://proxy.alcf.anl.gov:3128`.
- ∴ an **interactive compute node has both userns *and* network** → build *and* run happen there;
  the feared build-vs-run node split does not exist.

**Chosen approach (avoids fakeroot / def-file build):** on an interactive compute node, pull
`docker://ubuntu:24.04` (glibc 2.39) → `.sif`, then run the whole toolchain **inside**
`apptainer exec` with `/lus` bound and proxy set: `uv` (host binary, auto-bound via `$HOME`)
sees `manylinux_2_39`, so `uv add mojo` resolves the 2.34 wheel and builds `.venv` on flare.
The PBS job then runs `apptainer exec ubuntu2404.sif .venv/bin/mojo run …`.

Sources: ALCF Aurora Containers + Getting Started guides (docs.alcf.anl.gov/aurora/).

**Probe result — compute node, batch job `mojo_probe.o8656983` (2026-07-08):** ✅ all green
- `apptainer pull`/SIF assembly succeeded on the compute node — **no user-namespace error**
  (confirming the login-node failure was purely a login policy limitation).
- glibc **inside** the image = **Ubuntu GLIBC 2.39** — clears the 2.34 floor.
- host `uv 0.11.28` runs **inside** the container.
- (Batch PBS was used instead of `qsub -I` — the interactive queue was backed up; batch is also
  the preferred mode carried over from the Polaris port.)

→ Container path is proven. Next: `aurora_build.pbs` installs Mojo into a flare `.venv` from
inside the image; `aurora_l0.pbs` (container form) runs the L0 smoke test via
`apptainer exec … .venv/bin/mojo run …`.

**Gotcha (build attempt 1, `mojo_build.o8657182`):** `uv add` failed with *"No interpreter
found for Python 3.6"*. Cause: the earlier bare-host `uv init` (system CPython 3.6.15) had
left a stale **`.python-version` pinned to 3.6** in the flare dir; uv only ships managed
Python ≥3.7, so it couldn't fetch 3.6. Fix: cleanup now also removes `.python-version` +
uv-init sample files, and the build explicitly `uv python install 3.12` before init/add.

**Build success — `mojo_build.o8657204` (2026-07-08):** ✅
- uv fetched managed **CPython 3.12.13**, resolved + installed **`mojo==1.0.0b2`** (+ `mojo-compiler`,
  `mojo-compiler-mojo-libs`, `mojo-lldb-libs`, `mblack`, deps) into `.venv` on flare — all from
  inside the container (uv saw `manylinux_2_39`, so the `manylinux_2_34` wheel resolved).
- `uv run mojo --version` → **`Mojo 1.0.0b2 (2cf4d08a)`**. Toolchain feasibility gate CLEARED.

**Gotcha (PBS syntax):** Aurora PBS does **not** allow inline comments after `#PBS` directive
values — `#PBS -A ModCon   # comment` makes the account string literally `ModCon # comment`
and `qsub` rejects it with `directive error: -A ...`. Keep directive lines bare; put notes on
separate full-line comments.

## ✅ L0 PASS — capability smoke test (`mojo_l0.o8657258`, 2026-07-08)

**Command run:** `qsub aurora_l0.pbs` (dedicated job; internally does
`apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run aurora_l0_smoke.mojo`).
Node `x4117c5s0b0n0`:

| Field | Aurora result | Expected | |
|---|---|---|---|
| `has_accelerator` | **False** | False | ✅ no Intel-GPU backend, by design |
| `simd width f32` | **16** | 16 | ✅ AVX-512 (512-bit) |
| `simd width f64` | **8** | 8 | ✅ AVX-512 |
| `physical cores` | **102** | ~104 | ✅ (see core-specialization note) |
| `logical cores` | **204** | ~208 | ✅ |
| `sum 0..999999` | 499999500000 | 499999500000 | ✅ codegen+exec correct |

**Core-count finding:** PBS/`nproc` report **208** logical, but Mojo's `num_*_cores()` see **204/102**
— the node reserves ~2 physical (4 logical) cores (core specialization for OS/NIC services).
**Use 102 physical cores as the scaling ceiling for L3/L4**, not 104.

**Noise (harmless):** `Failed to initialize Crashpad ...` (no crash handler in the image) and
`gocryptfs not found` — both benign, same class as the Polaris job-log noise.

→ Toolchain gate CLEARED and capabilities confirmed. CPU ladder L1–L4 unblocked. All further
Mojo runs use the same wrapper: `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run <file>`.

## ✅ L2 PASS — explicit SIMD type semantics (`mojo_run.o8657612`, 2026-07-08)

Node `x4305c5s7b0n0`, `qsub -v MOJOFILE=00_simd_type.mojo aurora_run.pbs`. All operations
produced correct results:

| Op | Output | Correct? |
|---|---|---|
| construct / index / lane-write | `[1,2,3,4]` → `v[2]=3` → `[1,2,30,4]` | ✅ |
| splat (broadcast) | `[7,7,7,7]` | ✅ |
| lane-wise `a+b`, `a*2` | `[11,22,33,44]`, `[2,4,6,8]` | ✅ |
| `min(a, 7)` (free fn) | `[1,2,3,4]` | ✅ |
| reductions `reduce_add`/`reduce_max` | `10.0` / `4.0` | ✅ |
| mask + `select` (relu) | `[0,2,0,4]` | ✅ |
| `cast[int32]` | `[1,2,3,4]` | ✅ |
| `Scalar == SIMD[_,1]` | `42.0` | ✅ |

**Scope note (accuracy):** this file hardcodes **width 4** as a teaching demo, so it proves
SIMD-type *semantics* compile+run on Aurora — it does **not** itself demonstrate native W=16.
The AVX-512 native width (16) was already established at L0 via `simd_width_of`, and will be
genuinely exercised at **L3** where `vectorize` uses the native width. `mojo 1.0.0b2` compiled
the file unchanged (same as Mac) — no syntax porting needed.

## ✅ L4 PASS — CPU matmul ladder (`mojo_run.o8657642`, 2026-07-08)

**Command run:** `qsub -v MOJOFILE=03e_matmul_cpu.mojo aurora_run.pbs`.
Node `x4611c6s7b0n0`, N=1024 (2.147 GFLOP/pass), W=16, 102 cores.
Correctness vs naive reference: **0 mismatches** (both SIMD and parallel).

| Stage | Aurora time | Aurora GFLOP/s | Mac (M4 Max, W=4/16c) |
|---|---|---|---|
| naive (scalar) | 4944.8 ms | **0.43** | 2.5 |
| SIMD (W=16 / W=4) | 153.3 ms | **14.0** | 29 |
| parallel | 5.12 ms | **419.4** | 212 |

Speedups on Aurora: SIMD ≈ 32× over naive; parallel ≈ 30× over SIMD, **≈ 966× over naive**.

**Findings worth reporting:**
- **Parallel throughput scales hard:** 419 GFLOP/s on 102 cores — ~2× the Mac's 212, the
  expected win from many cores.
- **Per-core is *slower* than the Mac**, though: Aurora naive 0.43 vs Mac 2.5 GFLOP/s, and
  Aurora SIMD 14 (at W=16!) vs Mac 29 (at W=4). A single Sapphire Rapids core is weaker than
  an M4 Max core, and the SIMD stage's low efficiency (W=16 yet only ~32× over scalar) points
  to this naive (untiled) matmul being **cache/latency-bound per thread**, not FLOP-bound — the
  parallel win comes from core count, not per-core SIMD efficiency. A tiled/blocked kernel (or
  tapping AMX, L5) would be the path to higher single-core and aggregate numbers.
- Confirms `parallelize` + AVX-512 codegen both work and compose on Aurora.

## ✅ L3 PASS — SIMD + parallelize bandwidth scaling (`mojo_run.o8657706`, 2026-07-08)

**Command run:** `qsub -v MOJOFILE=02_vecadd_parallel.mojo aurora_run.pbs`.
Node `x4312c0s5b0n0`, N=64M (732 MB moved/pass), W=16, ITERS=30. Correctness: **0 mismatches**.

| workers | Aurora GB/s | Aurora speedup | | Mac GB/s (16c) |
|---|---|---|---|---|
| 1 | 15.2 | 1.0× | | 120 |
| 2 | 29.2 | 1.92× | | 168 |
| 4 | 35.5 | 2.33× | | 285 |
| 8 | 66.2 | 4.36× | | 296 |
| 16 | 118.0 | 7.76× | | **304 (peak)** |
| 32 | **176.3 (peak)** | 11.6× | | — |
| 64 | 160.3 | 10.5× | | — |
| 102 | 174.6 | 11.5× | | — |

**This is the most surprising result of the port — and a real evaluation finding:**
- **`parallelize` scales well** (11.5× at 102 cores) and composes with AVX-512 — the portability
  claim holds mechanically. ✅
- **But the absolute bandwidth is disappointing:** peak **~176 GB/s**, saturating by ~32 workers.
  That is *below the Mac's 304 GB/s* and **~10× below the Xeon-Max HBM potential** (~1 TB/s/socket,
  ~2 TB/s/node in HBM mode). The fat HBM node is **not** being exploited by naive portable code.
- **Very weak single-core BW:** 15.2 GB/s (vs Mac 120). A single Sapphire Rapids core cannot
  saturate memory — bandwidth only comes from aggregating many cores (opposite of the Mac's few
  fat cores). Consistent with the L4 per-core finding.
- **Likely cause — NUMA first-touch:** the arrays are filled by a *single* thread, so every page
  lands in **one socket's HBM** (first-touch). The 102 workers then span 2 sockets, so ~half hit
  *remote* HBM over the socket link, and all allocation is on one socket → aggregate BW is capped
  and the 64-worker dip / 32-worker plateau are the NUMA/scheduling signature. **Fix to try:**
  parallel NUMA-aware initialization (first-touch each chunk from the worker that will use it),
  and/or streaming (non-temporal) stores to avoid read-for-ownership traffic. Strong L5 follow-up.

Takeaway for the report: *portable Mojo `parallelize` scales, but realizing Aurora's HBM
bandwidth requires NUMA-aware data placement — it does not come for free from portable code.*

## ⚗️ L5a — NUMA first-touch experiment: hypothesis REFUTED (`mojo_run.o8657728`, 2026-07-08)

**Command run:** `qsub -v MOJOFILE=02b_vecadd_numa.mojo aurora_run.pbs`.
Same kernel/N/ITERS as L3, but with **parallel per-chunk first-touch init**. Correctness PASS.

| workers | L3 GB/s (serial init) | L5a GB/s (NUMA first-touch) |
|---|---|---|
| 1 | 15.2 | 15.4 |
| 2 | 29.2 | 30.2 |
| 4 | 35.5 | 39.2 |
| 8 | 66.2 | 70.4 |
| 16 | 118.0 | 74.5 |
| 32 | 176.3 | 121.7 |
| 64 | 160.3 | 159.8 |
| 102 | 174.6 | **177.0 (peak)** |

**Result: first-touch did NOT lift the ~176 GB/s ceiling** (177 vs 176). So single-socket page
placement is *not* the bottleneck — the earlier NUMA hypothesis is **refuted**.

**Leading explanation — `parallelize` gives no affinity/pinning control.** For first-touch to
help, the OS thread that *initializes* chunk `w` must be the same core that later *computes*
chunk `w`. Mojo's high-level `parallelize` scheduler does not guarantee that mapping (tasks can
migrate / be stolen across the 2 sockets), so NUMA locality established at init is lost at
compute time — making first-touch a no-op. Other contributors: per-pass task-launch overhead
(30 `parallelize` calls × up to 102 tasks) and read-for-ownership store traffic (no streaming
stores). Net: **the ~176 GB/s wall is a property of the portable abstraction, not of page
placement.**

**Evaluation takeaway (important for the Argonne question):** portable Mojo `parallelize`
*scales* (11.5×) and is correct, but **reaching a 2-socket HBM node's bandwidth needs affinity /
NUMA-binding control that the portable API does not expose.** Realizing full HBM BW would require
lower-level control (thread pinning + `numactl` binding, a persistent thread pool, non-temporal
stores) — i.e. it does **not** come for free. We did not pursue that deeper path this session;
it is the natural follow-up, alongside an AMX/tiled matmul for L4.

---
_Status: **L0 + L2 + L3 + L4 COMPLETE; L5a experiment done** (all correctness PASS). Findings:
(1) glibc-2.31 forces a container; (2) `parallelize`+AVX-512 work & scale (matmul 419 GFLOP/s);
(3) per-core is weak and portable code leaves HBM bandwidth / FLOPs unrealized — and a simple
NUMA first-touch fix did **not** help (no affinity control in `parallelize`). Remaining: L1
(fold in serial baseline as documentation) · deeper L5 (affinity/AMX) optional. Ready to
assemble REPORT_SETUP.md + REPORT_RESULTS.md._

---

# 🟢 Phase 2 — GPU ladder (G0→G5)

## 🧭 G0.1 — login-node recon, old vs new image (2026-09-14)

Same read-only recon block run on an old-image and a new-image UAN. Output files on flare:
`MOJO_WORK/AURORA/g0_login_recon_OLD.txt` (uan-0011) and `g0_login_recon.txt` (uan-0007).

| Item | Old image (`aurora-uan-0011`) | **New image (`aurora-uan-0007`)** |
|---|---|---|
| OS / kernel | SLES 15-SP4 / 5.14.21 | **SLES 15-SP7 / 6.4.0** |
| **glibc** | 2.31 | **2.38** ✅ (≥ 2.34 → Mojo wheels should install on the bare host) |
| Default modules | gcc 13.4.0, oneapi 2025.3.1, mpich 5.0.0 (3c70a61), libfabric 1.22.0 | **gcc 14.3.0, oneapi 2026.1.0**, mpich 5.0.0 (87e2045), libfabric 2.3.1 |
| PE root | `/opt/aurora/26.26.0` | `/opt/aurora/26.181.0` |
| Level Zero | `/usr/include/level_zero/ze_api.h`, `/usr/lib64/libze_loader.so{,.1}` | same ✅ |
| `icx`, `sycl-ls` | in `$CMPLR_ROOT/bin` | in `$CMPLR_ROOT/bin` ✅ |
| `ocloc` | `/usr/bin/ocloc` | `/usr/bin/ocloc` ✅ |
| **`llvm-spirv`** | `…/26.26.0/oneapi/compiler/2025.3/bin/compiler/llvm-spirv` | **`$CMPLR_ROOT/bin/compiler/llvm-spirv`** (= `/opt/aurora/26.181.0/oneapi/compiler/2026.1/bin/compiler/llvm-spirv`) ✅ — not on `PATH` |
| `spirv-val` | not found | **not found** ❌ |
| `clang` | not on PATH | not on PATH (use `icx`) |
| `xpu-smi` | not on login | not on login (expected; check on compute) |
| apptainer module | 1.2.5 | 1.5.1 |
| `next-eval` queue | — | enabled; ~4 queued / 23 running at check time |

**Findings:**
- **The glibc blocker from Phase 1 is gone on the new image.** Plan: install Mojo 1.0.0 directly on
  the host (no container). The container becomes a fallback only.
- **SPIR-V toolchain is present** except `spirv-val`. Options: validate on the Mac
  (`brew install spirv-tools`), look for a Spack/`/soft` module, or rely on `ocloc`/IGC rejecting bad
  modules at build or load time.
- Old image = oneAPI 2025.3.1; ALCF says the new image needs recompiles → all Phase 2 work
  targets the new image (`next-eval` until the October rollout).

## ✅ G0.2 PASS — Mojo 1.0.0 installs and runs on the BARE HOST (2026-09-14)

Run on `aurora-uan-0007` (new image, glibc 2.38). Output: `MOJO_WORK/AURORA/gpu/g0_install.txt`.

- `uv init --bare --no-workspace` + `uv add "mojo==1.0.0"` in `MOJO_WORK/AURORA/gpu/` → own `.venv`
  (managed CPython 3.12.13). `mojo --version` → **`Mojo 1.0.0 (ed45d567)`**.
- **No container needed.** The Phase 1 glibc blocker is gone on the new image.
- L0 smoke (`../aurora_l0_smoke.mojo`, b2-era file) **compiled unchanged on 1.0.0**: `has_accelerator=False`,
  simd f32=16 / f64=8, sum correct. Cores 52/104 because it ran on a login node (expect 102/204 on compute).
- Noise: `Failed to initialize Crashpad` (harmless, same as Phase 1).

**`--print-supported-accelerators`:** AMD (ROCm/HIP), Apple Silicon (M1–M5, Metal 4), NVIDIA CUDA
(sm_52 … sm_121a). **No Intel entry.** Confirms the gap.

**`--print-supported-targets` (LLVM registered targets):** aarch64 family, `amdgpu`, `hexagon`,
`nvptx`/`nvptx64`, `r600`, riscv family, `x86`/`x86-64`, `xtensa`. **No `spirv`/`spirv64`.**
→ The prebuilt compiler's LLVM was built **without the SPIR-V target**, so
`--target-triple spirv64` won't work. G2 must go **Mojo `--emit llvm` → retarget IR → oneAPI
`llvm-spirv`** (translation outside Mojo). A G5 compiler fork would need the SPIR-V target
enabled in its LLVM build.
- G2 risk this raises: **LLVM IR version skew.** The Mojo LLVM (lldb libs say 24.0.0git) is newer than
  the Intel LLVM behind oneAPI 2026.1 `llvm-spirv`, and newer textual IR/bitcode may not parse. The
  OSS tree has `BitcodeWriter19/21` (downgraded bitcode writers, probably used for NVPTX/libNVVM), which
  might help. Check early in G2.

**Gotcha — uv workspace capture (cost one retry):** the first `uv init` in `AURORA/gpu/` found the
Phase 1 `AURORA/pyproject.toml` one directory up and **added `gpu` as a workspace member**. Workspaces
share one venv, so `uv add "mojo==1.0.0"` **upgraded the Phase 1 `AURORA/.venv` from 1.0.0b2 → 1.0.0**
and created no `gpu/.venv`. Fix: delete the `[tool.uv.workspace]` block from the parent
(`pyproject.toml.bak` kept), recreate `gpu` with **`uv init --no-workspace`**.
**Rule: always `--no-workspace` when nesting a uv project under another.**
- Restoring Phase 1 to b2 with `uv add "mojo==1.0.0b2"` **failed**: transitive `mojo-compiler==1.0.0b2` is a
  pre-release, and uv won't pick it without `--prerelease=allow`. **Decided not to restore:** Phase 1
  `AURORA/.venv` stays on 1.0.0 and all work goes forward on 1.0.0. Phase 1 b2 files that depend on
  moved APIs (e.g. `std.algorithm.parallelize` → `max.algorithm`) won't compile there as-is; port them to
  1.0.0 if they're ever re-run.

## ✅ G0.3 — PVC GPUs visible on a `next-eval` compute node (2026-09-14)

Interactive job `8827177`, node `x4006c4s4b0n0` (SLES 15-SP7, glibc 2.38, oneapi 2026.1.0).
Output: `MOJO_WORK/AURORA/gpu/g0_gpu.txt`.

| Check | Result |
|---|---|
| `/dev/dri` | `card0–5`, `renderD128–133` ✅ (6 GPUs) |
| `sycl-ls` default (COMPOSITE) | **6×** `[level_zero:gpu] Intel(R) Data Center GPU Max 1550 12.60.7 [1.6.33578+77]` ✅ |
| `sycl-ls` `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | **12×** (one per tile) ✅ |
| `sycl-ls` FLAT + `ZE_AFFINITY_MASK=0.0` | **"No platforms found"** ❌ → in FLAT mode the mask takes a *flat index* (`0`), not `gpu.tile` (`0.0`). See the mask note below. |
| Default env | `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` is preset (sycl-ls prints a filter notice) |
| `xpu-smi` | not on PATH on compute either (may need a module; not needed) |
| `ocloc query` | OCL driver **25.18.33578**, NEO revision 33578; IGC_REVISION printed nothing |
| **`llvm-spirv --version`** | **LLVM 22.1.0** (vs Mojo's LLVM 24.0.0git) |
| Mojo 1.0.0 smoke on compute | ✅ `has_accelerator=False`, W=16/8, sum correct; **104 physical / 208 logical** (July image showed 102/204) |

**Findings:**
- **Q1 answered:** Mojo 1.0.0 runs on the bare host of a compute node, and all 6 PVC GPUs / 12 tiles
  are visible through Level Zero. No container needed.
- **Tile-mask syntax depends on hierarchy mode:** COMPOSITE → `ZE_AFFINITY_MASK=0.0` (GPU 0, tile 0);
  FLAT → `ZE_AFFINITY_MASK=0` (flat device 0). **Confirmed:** each form shows exactly one device.
  Standard for Phase 2: `ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0`.
- `xpu-smi` is a module (`xpu-smi/1.2.43` default, `1.3.5`).
- **G0 COMPLETE.**
- **G2 risk confirmed and sized: a 2-major-version LLVM gap** (Mojo 24 → `llvm-spirv` 22). LLVM bitcode
  is readable by *newer* LLVM, not older, so Mojo's IR may not load into oneAPI `llvm-spirv`. The prebuilt
  `mojo build` has no bitcode-version flag. Options for G2: (a) try it anyway (unoptimized IR often uses
  no new features); (b) build upstream SPIRV-LLVM-Translator against a matching LLVM; (c) use an LLVM
  24 build with the in-tree SPIR-V backend (`llc -mtriple=spirv64`); (d) lead to check: Mojo's
  **Metal** path emits a kernel `.ll` sidecar (`--emit asm`), which may already be GPU-shaped and
  older-LLVM-compatible (Apple's toolchain is older LLVM).

## 🔧 G1 kernel build — OpenCL C → SPIR-V on uan-0007 (2026-09-14/15)

`bash g1_build_kernel.sh` (outputs `gpu/g1_build.txt`, `gpu/g1_build2.txt`).
Tools: oneAPI DPC++ `clang` 2026.1.0 at `$CMPLR_ROOT/bin/compiler/clang`; `llvm-spirv` (LLVM 22.1.0).

| Route | Result |
|---|---|
| **A** `clang -cl-std=CL3.0 -target spir64 -O2 -emit-llvm` → `llvm-spirv` | ✅ 1560 B, magic `03 02 23 07`, `EntryPoint … "vecadd"` (after fix) |
| **B** `ocloc compile -device pvc -spv_only` | ✅ 1544 B, magic `03 02 23 07`, "Build succeeded" |

- **Gotcha:** oneAPI's clang driver rejects `-finclude-default-header` / `-fdeclare-opencl-builtins`
  as driver flags; they must be `-Xclang -finclude-default-header -Xclang -fdeclare-opencl-builtins`.
  The first attempt failed route A for this reason only; the job was submitted with the route B `.spv`.
- **Why it matters for G2:** route A is the same `LLVM bitcode → llvm-spirv` hop that Mojo-generated
  IR must take, and it now works end to end on Aurora (for LLVM-22 bitcode from oneAPI clang).

## 💡 G2 prep (Mac) — Mojo's Metal backend gives us GPU-lowered kernel IR (2026-09-14)

- `mojo build --emit asm --target-accelerator apple-m4 MOJO_CURRICULUM/02_vecadd_gpu.mojo` writes the
  kernel as `*_vecadd_kernel_*.ll` (6.5 KB, triple `air64-apple-macosx15.0.0`). Without
  `--target-accelerator` no sidecar is written.
- Contents: one optimized function, `global_idx` already lowered to
  `threadgroup_position_in_grid * threads_per_threadgroup + thread_position_in_threadgroup`
  (Metal passes these as `<3 x i32>` args); data buffers `ptr addrspace(1)`; the `size` scalar arrives by
  pointer in `addrspace(2)`; each TileTensor arg is a pointer to a struct whose field 0 is the
  data pointer. No host runtime calls, no `@air.*` intrinsics, no metadata in the body.
- `gpu/air2spir.py` rewrites it to SPIR (see its header). Output for vecadd: `spir_kernel void @vecadd`
  with 4 `ptr addrspace(1)` args and 3 OpenCL builtins (`get_group_id`, `get_local_id`,
  `get_local_size`); unused Metal params are dropped.
- **Why it matters:** this avoids writing a GPU lowering ourselves and avoids the LLVM 24→22 skew
  (the Metal IR is emitted for Apple's older toolchain). Next: `gpu/g2_build_kernel.sh` on uan-0007
  (clang → llvm-spirv → ocloc), then `qsub -v MOJOFILE=g2_vecadd_lz.mojo g1_run.pbs`.

## ⚠️ G1 run 1 — failed at zeKernelCreate (job 8827457, 2026-09-15)

Node `x4006c4s4b0n0`, FLAT + mask 0, `sycl-ls` showed exactly 1 device. Used route-A `g1_vecadd.spv`.

**Worked (first Mojo → Level Zero contact with PVC):** loader + driver init, device discovery,
`DeviceProperties(vendor=0x8086, device=0xbd6, type=GPU, clock=1500MHz, max_mem=62244MB, name=Intel(R) Data
Center GPU Max 1550)`, `ComputeProperties(max_group=1024, slm=131072B, subgroups=[16, 32])`, context +
immediate command list, USM host/device allocation, host→device copies, **`zeModuleCreate` on the SPIR-V**.

**Failed:** `Kernel creation failed: 0x78000011` = `ZE_RESULT_ERROR_INVALID_KERNEL_NAME`.
**Cause (our bug, in AURORA PATCH 1):** the NUL-terminated name buffer `name_c` was only used via
`name_c.unsafe_ptr()`, so Mojo's ASAP destruction freed it before `zeKernelCreate` read it.
**Fix:** `_ = name_c^` after the call keeps it alive. Lesson for all FFI code here: **a raw pointer does
not keep its owner alive; add an explicit use of the owner after the C call.**

## 🔧 G2.1 kernel build, attempt 1 (uan-0007, 2026-09-15)

`bash g2_build_kernel.sh` → `gpu/g2_build.txt`.
1. `air2spir` ✅ `@vecadd`, builtins `get_group_id`/`get_local_id`/`get_local_size`.
2. oneAPI clang (LLVM 22) **parsed Mojo's Metal-path IR** ✅ → 3332-byte bitcode. **The LLVM 24→22 skew
   risk did not materialize for this kernel.**
3. `llvm-spirv` ❌ `InvalidTargetTriple … x86_64-unknown-linux-gnu`: clang silently overrides the
   module triple with its default (`-Woverride-module`). **Fix:** pass `-target spir64-unknown-unknown`
   to clang.

## ✅ G2.1 — Mojo-written kernel compiles to a PVC binary (uan-0007, 2026-09-15)

`bash g2_build_kernel.sh` (with `-target spir64-unknown-unknown`) → `gpu/g2_build.txt`.

| Step | Result |
|---|---|
| 1. `air2spir` | ✅ `@vecadd`, 3 builtins |
| 2. clang (LLVM 22) text IR → bitcode | ✅ 3292 B |
| 3. `llvm-spirv` | ✅ 3280 B, magic `03 02 23 07`; `EntryPoint 6 15 "vecadd"`; capabilities Addresses, Linkage, Kernel, Int64, Int8 |
| 4. `ocloc compile -spirv_input -device pvc` | ✅ "Build succeeded" → `g2_vecadd_pvc_pvc.bin` (7984 B) |

**Meaning:** `vecadd_kernel` from the **unchanged** `MOJO_CURRICULUM/02_vecadd_gpu.mojo` goes Mojo Metal
backend → `air2spir.py` → LLVM 22 → SPIR-V → IGC native PVC code. The compile side of G2 is done;
execution waits on the G1 host fix.

**Watch item:** no `BuiltIn` decorations and a `Linkage` capability, which means `get_group_id` etc. were
kept as *imported functions* (not SPIR-V builtin variables). IGC resolved them (the build succeeded), so
this is probably fine. If a runtime problem shows up, add `!opencl.ocl.version` / `!spirv.Source`
metadata in `air2spir.py` so `llvm-spirv` lowers them to `BuiltIn` variables.

## ✅ G1 PASS — Mojo drives a PVC tile through Level Zero (job 8827488, 2026-09-15)

Node `x4001c5s4b0n0`, FLAT + `ZE_AFFINITY_MASK=0`, route-A `g1_vecadd.spv` (OpenCL C kernel), with the
`name_c` lifetime fix. Output: `gpu/mojo_g1.o8827488`.

| Field | Aurora PVC tile (G1) | Polaris A100 (P1, `02_vecadd_gpu`) |
|---|---|---|
| N / block / groups | 1,000,000 / 256 / 3907 | 1,000,000 / 256 / 3907 |
| mismatches | **0 / 1000000** | 0 / 1000000 |
| RESULT | **PASS** | PASS |
| kernel time | **0.00561 ms/pass** | 0.00766 ms/pass |
| module load | 38.9 ms (`zeModuleCreate` + `zeKernelCreate`) | — |
| spot check | `c[1]=1.0`, `c[999999]=-999997.0` ✅ | — |

**Caveats:** vector add is memory-bound (≈12 MB touched per pass), so this is a plumbing proof, not a
performance claim. Timing = 100 async launches + one host synchronize, same method as Polaris. Sanity-check
later by varying ITERS/N. The kernel here is OpenCL C; the Mojo-written kernel is G2.

## ⚠️ G2 run 1 — GPU page fault (job 8827498, 2026-09-15)

Node `x4001c5s4b0n0`. Module + kernel creation succeeded (no error before launch), then:
`Segmentation fault from GPU at 0xff00ffffff444000, ctx_id: 1 (CCS) type: 0 (NotPresent), level: 1 (PDE),
access: 1 (Write)` → compute-runtime abort (`drm_neo.cpp:288`), exit 134.

- The address is in the PVC USM device range, so this is a real device buffer (almost certainly `d_c`,
  the write target), just not mapped for the kernel.
- **Cause:** the Metal-path kernel reaches `d_a/d_b/d_c` **indirectly** (it loads the data pointer from a
  holder buffer). Level Zero makes only direct kernel-argument allocations resident.
- **Fix (AURORA PATCH 4):** `zeKernelSetIndirectAccess(kernel, HOST|DEVICE|SHARED)` via
  `Kernel.set_indirect_access(7)` before launch.
- Alternative for later (perf): have `air2spir.py` strip the holder indirection so data pointers are
  direct args (no extra load, no indirect-access residency cost).

## ✅ G2 PASS — a Mojo-written kernel runs on an Aurora PVC tile (job 8827519, 2026-09-15)

Node `x4001c5s4b0n0`, FLAT + `ZE_AFFINITY_MASK=0`, `g2_vecadd.spv` built by `g2_build_kernel.sh`, with
`set_indirect_access(7)`. Output: `gpu/mojo_g1.o8827519`.

**Kernel source:** `vecadd_kernel` in `MOJO_CURRICULUM/02_vecadd_gpu.mojo`, **unchanged** (Mojo 1.0.0).
**Pipeline:** Mojo Metal backend → `air2spir.py` → oneAPI clang (LLVM 22) → `llvm-spirv` → Level Zero
(IGC JIT) → PVC.

| Field | G2 (Mojo kernel, PVC) | G1 (OpenCL C kernel, PVC) | Polaris P1 (Mojo kernel, A100) |
|---|---|---|---|
| mismatches | **0 / 1000000** | 0 / 1000000 | 0 / 1000000 |
| RESULT | **PASS** | PASS | PASS |
| kernel time | **0.00664 ms/pass** | 0.00561 | 0.00766 |
| module load | 24.9 ms | 38.9 ms | — |
| spot check | `c[1]=1.0`, `c[999999]=-999997.0` ✅ | same | — |

**Findings:**
- **Workload parity (level A) achieved for vector add:** the same Mojo kernel code that ran on the A100
  now runs on PVC, correctly.
- G2 is ~18% slower than G1, plausibly the extra pointer load per launch plus indirect-access
  residency. The `air2spir` holder-stripping idea would remove both. Vector add is memory-bound, so
  it's not a performance claim (G3 is).
- The imported-builtin watch item (`get_group_id` etc. as imports) was a non-issue: IGC resolves them.
- **G2 COMPLETE.**

## 🧰 G3 prep (Mac) — matmul kernels through the Metal-AIR route (2026-09-15)

`mojo build --emit asm --target-accelerator apple-m4` on unchanged `MOJO_CURRICULUM` files:

| Kernel | AIR size | New IR features | air2spir |
|---|---|---|---|
| `03a_matmul_naive` (N=2048, 16×16 blocks) | 8 KB | 2D thread positions only | ✅ no change needed |
| `03b_matmul_tiled` (N=2048, 16×16) | 30 KB | 2 barriers in a runtime loop; 2 × `[256 x float]` `addrspace(3)` globals; `convergent` attr | ✅ after extension |
| `03c_matmul_coarse` (N=2048, 256-thread blocks) | 228 KB | 2 barriers; 2 × `[1024 x float]` shared globals | ✅ after extension |
| `03d_matmul_check` (N=256, coarse kernel) | 228 KB | same as 03c | ✅ after extension |

- **air2spir extension:** `call @air.wg.barrier(i32, i32)` → `call spir_func void @_Z7barrierj(i32 1)`;
  each `internal addrspace(3) global [K x T]` → an extra `ptr addrspace(3) %slmI` kernel arg (OpenCL
  `__local`), plus an args manifest (`arg <i> buffer|const|local <bytes>`). The vecadd output is
  byte-identical to before (regression check).
- **Other backends checked:** `--target-accelerator nvidia:sm_80` → `.ptx` ("Generated by LLVM NVPTX
  Back-End", but Modular GPU assembly with `%tid.x`, `.shared`); `amdgpu:gfx942` → `.amdgcn` (lane-level
  assembly). Neither is LLVM IR, so **Metal AIR is the only IR-level route**.
- **Main risk:** Metal barriers are backed by Apple's runtime. Whether Intel IGC honors OpenCL
  `barrier()` with real synchronization across a 256-thread GPU work group is unknown. `03d` (full check
  at N=256) runs before the timed tiled/coarse kernels, and naive runs first so a crash can't lose it.
- **Bindings (AURORA PATCH 5):** `Kernel.set_arg_local(index, size)` → `zeKernelSetArgumentValue(..., NULL)`.
- **Yardstick:** `gpu/g3_bench_torch_xpu.py` (oneMKL via PyTorch XPU, same GFLOP/s formula), run in the
  same job after `module load frameworks`.

## ⚠️ G3.1 kernel build, attempt 1 (uan-0007, 2026-09-15)

`bash g3_build_kernels.sh` → `gpu/g3_build.txt`.

| Kernel | Result |
|---|---|
| naive | ✅ 4152 B SPIR-V, `ocloc` PVC build succeeded |
| **tiled** | ✅ 11208 B, **2 `barrier` calls**, `ocloc` PVC build succeeded. IGC accepts OpenCL `barrier()` and `__local` args (runtime semantics still untested). "Workgroup refs: 0" is a grep artifact: local args are Function-storage params in this translator output. |
| check | ❌ `llvm-spirv`: `LLVM ERROR: Cannot translate source of bitcast instruction` in `SPIRVLowerBitCastToNonStandardTypePass` |
| coarse | ❌ same |

**Cause:** 03c/03d hold register tiles as `SIMD[float32, 64]` → `phi/insertelement/extractelement <64 x float>`
(non-standard SPIR-V vector width; standard max is 16). Mojo's Metal IR also has 772 no-op
`bitcast ptr → ptr`, and the translator's lowering pass aborts tracing them near the wide vectors.
**Fix:** `air2spir.py` now rewrites no-op pointer bitcasts as `getelementptr i8, <ptr>, i64 0` (numbering
preserved; `--keep-bitcasts` restores the old behaviour). **Next risk:** IGC may still reject 64-wide
vectors, in which case scalarize them.

## ⚠️→🔧 G3.1 build attempt 2 + fix (2026-09-15)

`g3_build2.txt`: naive ✅ (4392 B), tiled ✅ (13932 B, 2 barriers, `ocloc` OK), check/coarse ❌ same abort. The
no-op-bitcast rewrite was the wrong theory.

**Actual cause (from the translator source, `SPIRVLowerBitCastToNonStandardType.cpp`):** the pass collects every
`extractelement` on a vector whose width isn't 2/3/4/8/16 and tries to re-type its source. It can only do that
when the source is a `load`/`bitcast`/`addrspacecast`. In 03c/03d the `<64 x float>` register tiles come
from `phi`/`insertelement`, so it calls `report_fatal_error`. The pass is skipped only when
`SPV_EXT_long_vector` or `SPV_INTEL_vector_compute` is enabled, and IGC support for those in OpenCL kernels
is uncertain.

**Fix — scalarize in `air2spir.py`:** in 03c/03d the long vectors are touched only by `phi` (2),
`insertelement` (512) and `extractelement` (576), all with constant indices. The transform renames unnamed
values (`%N`→`%uN`, explicit entry label), turns each long-vector `phi` into 64 scalar `phi`s, resolves
insert/extract chains to the underlying scalars, and dies on any other long-vector use. Standard
`<8 x float>` ops are untouched.

**Local verification (Mac):** Apple clang 21 parses and verifies all four G3 kernels plus vecadd
(`clang -x ir -c -emit-llvm`). coarse: 0 long vectors, 128 scalar phis (2 × 64), final `store float
%u3619.e62/e63` into C. Not yet run through oneAPI `llvm-spirv`/`ocloc`.

## ✅ G3.1 — all four matmul kernels build for PVC (uan-0007, 2026-09-15)

`bash g3_build_kernels.sh` → `gpu/g3_build3.txt` (with the long-vector scalarizer).

| Kernel | SPIR-V | Capabilities (beyond Addresses/Linkage/Kernel/Int64/Int8) | Barriers | `ocloc` PVC |
|---|---|---|---|---|
| naive (03a) | 4,392 B | — | 0 | ✅ |
| check (03d) | 128,172 B | Vector16 | 2 | ✅ + warning |
| tiled (03b) | 13,932 B | — | 2 | ✅ |
| coarse (03c) | 128,188 B | Vector16 | 2 | ✅ + warning |

**IGC warning (coarse/check):** `compiled SIMD32 allocated 128 regs and spilled around 247`. IGC vectorizes
the work group 32 threads at a time. The coarse kernel's per-thread register tiles (64 accumulators +
2 × 8) exceed the EU register budget at that width, so values spill to memory. This is the PVC analog of
the Apple `Array` register-tile spill in `RESULTS01.md`. **Expect coarse to underperform on PVC; a
PVC-tuned variant (smaller TM×TN, or SIMD16) is a G3 follow-up, not a blocker.**

## ✅ G3.2 — all four Mojo matmul kernels PASS on a PVC tile (job 8827590, 2026-09-15)

Node `x4001c5s4b0n0`, FLAT + `ZE_AFFINITY_MASK=0`, kernels from `g3_build3.txt`. Output: `gpu/mojo_g3.o8827590`.
All kernel sources are **unchanged** `MOJO_CURRICULUM` files (Metal AIR → `air2spir` → SPIR-V → Level Zero).

| Variant | N | Correctness | Avg time | **PVC tile GFLOP/s** | A100 (Polaris P2) | PVC / A100 |
|---|---|---|---|---|---|---|
| naive (03a) | 2048 | SAMPLED 0/260 ✅ | 7.01 ms | **2,449** | 2,659 | 0.92 |
| check (03d, coarse kernel) | 256 | **FULL 0/65,536** ✅ | 0.10 ms | (333, too small to mean anything) | — | — |
| tiled (03b) | 2048 | SAMPLED 0/260 ✅ | 7.89 ms | **2,177** | 4,137 | 0.53 |
| coarse (03c) | 2048 | SAMPLED 0/260 ✅ | 9.03 ms | **1,903** | 8,966 | 0.21 |

Aurora CPU (L4, 102 cores): 419 GFLOP/s. PVC naive is ~5.8× that.

**Findings:**
- **Barriers + shared local memory work on PVC.** The coarse kernel's two barriers per K-slab over a
  256-thread group gave an exact full-matrix match. A race would have produced errors, so IGC honors OpenCL
  `barrier()` for these GPU work groups.
- **Naive is close to the A100 (92%).** The optimizations tuned for Apple/NVIDIA go *backwards* on PVC:
  tiled < naive, coarse < tiled. That matches the IGC build warning (coarse compiled SIMD32 and spilled
  ~247 register values) and the earlier Apple finding that tiling choices are architecture-specific
  (`RESULTS01.md` "tiling flip").
- JIT warmup is small (7–12 ms per kernel).
- **oneMKL yardstick did not run:** `python: command not found`, because `module load frameworks | tail`
  ran in a subshell (script bug, fixed; rerun with `qsub -v G3_VARIANTS=none g3_run.pbs`).

**Caveats:** one run each (no repeat statistics); N=2048 correctness is sampled (260 cells), the full check
is N=256 only; geometry is the Apple/NVIDIA-tuned one, untuned for PVC.

## ⚠️ G3.3 — oneMKL yardstick via PyTorch XPU failed (job 8827599, 2026-09-15)

`module load frameworks` now works (`frameworks/2026.1.0`, python at
`/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/bin/python`), but `import torch` fails:
`torchcomms` → `OSError: libglog.so.0: cannot open shared object file`. **The ALCF frameworks module on the
`next-eval` preview image is broken** (not our code). Workaround: `gpu/g3_mkl_sgemm.cpp` calls oneMKL SYCL SGEMM
directly (same method/formula/generators; sampled correctness check). `g3_run.pbs` runs it first and keeps the
torch attempt as optional, with a `libglog` search. Worth reporting to ALCF support.

## ✅ G3.3 — oneMKL yardstick: 21,148 GFLOP/s on one PVC tile (job 8827619, 2026-09-15)

`gpu/g3_mkl_sgemm` (oneMKL SYCL `column_major::gemm`, USM device buffers, built on uan-0007 with
`icpx -fsycl -qmkl -O2`, no warnings). N=2048, 50 iters: **0.812 ms/pass → 21,147.6 GFLOP/s**, sampled check
0/260 PASS. Consistent with ALCF's published single-tile SGEMM peak (21 TFlop/s).

**Torch correction:** `libglog.so.0` *does* exist at
`/opt/aurora/26.181.0/frameworks/aurora_frameworks-2026.1.0/lib/`; the `frameworks/2026.1.0` module doesn't put
that directory on `LD_LIBRARY_PATH`. (Likely workaround: prepend it. Not needed; oneMKL is measured directly.)

### 📊 G3 summary — same Mojo kernels, PVC tile vs A100, each relative to its vendor BLAS

| Kernel (unchanged MOJO_CURRICULUM source) | PVC GFLOP/s | % of oneMKL (21,148) | A100 GFLOP/s | % of cuBLAS (13,839) |
|---|---|---|---|---|
| naive (03a) | 2,449 | 11.6% | 2,659 | 19.2% |
| tiled (03b) | 2,177 | 10.3% | 4,137 | 29.9% |
| coarse (03c) | 1,903 | 9.0% | 8,966 | 64.8% |
| **vendor BLAS** | **21,148** | 100% | **13,839** | 100% |

**Reading:**
- Hardware headroom: oneMKL on one PVC tile is **1.53×** cuBLAS on the A100.
- The Mojo kernels reach only 9–12% of it. On the A100 the Apple/NVIDIA-tuned tiling took Mojo from 19%
  to 65% of vendor BLAS; on PVC the same geometry makes things worse (IGC SIMD32 register spill, see G3.1).
- So on PVC today: **correctness parity yes, performance parity no.** The gap looks like geometry/tuning (the
  kernels were tuned for Apple M4 and happened to suit the A100), not a pipeline defect. Unproven until
  a PVC-tuned variant is measured.

## 🧰 G4 prep (Mac) — 04b's training kernels through the Metal-AIR route (2026-09-15)

`mojo build --emit asm --target-accelerator apple-m4` on the unchanged
`MOJO_CURRICULUM/04b_train_mlp_gpu.mojo`.

| Item | Finding |
|---|---|
| Kernel sidecars emitted | **14** (not 9): the generics instantiate once per `TensorLayout` |
| Distinct bodies | **12** — the 4 SGD launches share one byte-identical kernel |
| Launches per epoch | 15, over those 12 kernels |
| `addrspace(3)` shared globals | **0** in every kernel → no `__local` args, no SLM sizing |
| `barrier()` | **none** → G3's hardest risk doesn't apply at this rung |
| Long vectors (`<64 x float>`) | none → the G3 scalarizer isn't exercised |
| Thread positions | `<3 x i32>` params, same shape as G2/G3 |
| **New `@air.*` intrinsic** | **`@air.max.s.i64(x, 0)`** — the K-loop trip-count guard, in 7 of the 14 |

**Only blocker: `air.max.s.i64`.** `air2spir.py` exits 2 on any unknown `@air.*` call, so 7 of the
14 kernels were refused. Fixed by inlining it rather than importing OpenCL `_Z3maxll`:

```llvm
%.airminmax0 = icmp sgt i64 %19, 0
%21 = select i1 %.airminmax0, i64 %19, i64 0
```

Inlining keeps the module free of external symbols (nothing for IGC to resolve) and preserves
value numbering. The rewrite covers `@air.{max,min}.{s,u}.iN` generally.

**Verification (Mac):** all 12 staged kernels convert and pass `clang -x ir -c -emit-llvm`
(Apple clang 21). Regression: vecadd and all four G3 matmuls still convert, with zero min/max
sites — the patch is inert for the earlier rungs.

**Kernel identity.** The Metal sidecar filenames carry a mangling hash, not a role. Each of the
repeated generics was matched to its 04b launch by the layout strides baked into its body
(N=256, D=2, H=16) — `fwd1` has A stride 2 / B stride 16 where `fwd2` has 16 / 1, and so on.
The mapping table lives in `pre_process/README.md` and must be re-derived if `04b` changes.

**New host work at this rung: `const` arguments.** G3's kernels had their sizes as comptime
constants, so every argument was a buffer or `__local`. 04b passes `Int32` shapes and `Float32`
scalars (`two_over_N`, `LR`), which Metal delivers as constant-buffer pointers — manifest kind
`const`. The host binds each as a 4-byte USM cell holding the value itself (as opposed to a
`buffer` arg, which is an 8-byte holder containing a device pointer). Every argument is fixed
for the whole run, so `g4_train_mlp_lz.mojo` binds once and the epoch loop is pure launches.

**Kit:** `gpu/g4_*.air.ll` (12), `gpu/g4_build_kernels.sh`, `gpu/g4_train_mlp_lz.mojo`,
`gpu/g4_run.pbs`. Host compile-checked on the Mac (Mojo 1.0.0, no warnings); not yet run.

**Open risks for the run:**
- `fmul contract` / `fadd contract` in the AIR: if IGC fuses where Apple/NVIDIA did not, the
  loss curve could drift in the last digits. The bar is 04b's curve 2.178 → 0.000561.
- 12 module loads (~25 ms each in G2) before the first epoch.
- 4,500 launches through the immediate command list — launch overhead dominates at N=256, H=16,
  exactly as 04b's own header warns for the Apple GPU.

## ⚠️ G4 run 1 — wrong loss curve, diverged to NaN (job 8828385, 2026-09-15)

Epoch 1 loss **2.5238** against a reference of 2.178, then 788 → inf → NaN. Epoch 1 only exercises
the upload plus `fwd1`/`biasrelu`/`fwd2`/`addb2`, so the fault was in the forward path, not backprop.
A CPU reference written alongside (NumPy, float32, same LCG) reproduced 04b's documented endpoints
exactly — 2.178356 and 0.000561476 — which made the observed value provably wrong rather than
merely suspicious, and gave signatures to compare against: z2 all-zero would give 2.5615, skipping
the ReLU 2.6684.

## 🔬 G4 run 2 — stage-by-stage forward check localizes it (job 8828430, 2026-09-15)

Added an upload readback plus a per-stage comparison of `z1`, `a1`, `z2` against the CPU reference,
reporting wrong-count, **exact-zero count** and max relative error — the zero count separates a
coverage/write fault from an arithmetic one.

| Stage | Result |
|---|---|
| upload check (X, Y, W1, W2) | OK |
| `z1` | **0 / 4096 wrong** — `fwd1` correct |
| `a1` | 1625 / 4096 wrong, **3733 exact zeros** (≈2048 expected from ReLU) |
| `z2` | 224 / 256 wrong, **224 exact zeros** — only 32 rows (2 thread groups) written |

No wrong *values* anywhere — only missing writes. That is the signature of kernels reading buffers
their producer had not filled yet: `biasrelu` computes `v = z1 + b1` and stores `relu(v)`, so an
unwritten `z1` (memset zero) yields `a1 = 0`; `fwd2` then produces exactly 0.0 for those rows.
`z1` still verified clean because `b1 = 0` makes `biasrelu`'s write-back of `Z` a no-op.

**Root cause: a Level Zero immediate command list is not in-order.** `mojo_intel_gpu` creates it
with `mode = ASYNCHRONOUS` and no ordering guarantee, so appended kernels may execute concurrently.
G1–G3 could not have caught this: each launched a single kernel repeatedly, where overlap is
harmless and is in fact what makes the timing method work. **The first multi-kernel pipeline is
where it surfaces.**

**Fix (AURORA PATCH 6):** `zeCommandListAppendBarrier` binding + `IntelGPUContext.barrier()`, called
after each of the 15 launches. It orders execution device-side with no host round-trip.

**Lesson for G5-lite:** `DeviceContext` presents an *in-order stream*; a Level Zero backend must
supply that ordering itself. CUDA streams give it for free, which is why the Polaris path never
needed it. This belongs in the host layer, not in user code.

*(Two host bugs of ours were also fixed on the way: all four uploads shared one staging buffer with
no wait between asynchronous copies, and the loss readback read `h_z2` before its `memcpy_dtoh`
completed. Neither turned out to be the cause — the upload check passed — but both were real races.)*

## ✅ G4 PASS — 04b MLP training reproduces its loss curve on a PVC tile (job 8828472, 2026-09-15)

Node `x4001c1s4b0n0`, FLAT + `ZE_AFFINITY_MASK=0`. Kernel sources are **unchanged**
`MOJO_CURRICULUM/04b_train_mlp_gpu.mojo` (Metal AIR → `air2spir` → SPIR-V → Level Zero).
12 distinct kernels, 15 launches per epoch, 300 epochs. Output: `gpu/mojo_g4.o8828472`.

Forward verification: `z1` 0/4096 wrong, `a1` 0/4096 wrong (2108 exact zeros = the genuine ReLU
zeros), `z2` 0/256 wrong, max relative error **0.0** at every stage.

| epoch | PVC tile (Mojo kernels) | CPU float32 reference | A100 / Apple (04b documented) |
|---:|---|---|---|
| 1 | 2.1783555 | 2.17835617 | 2.178 |
| 50 | 0.016860535 | 0.016860541 | — |
| 100 | 0.0056486796 | 0.00564867724 | — |
| 150 | 0.0029576812 | 0.00295768236 | — |
| 200 | 0.0016037121 | 0.00160371314 | — |
| 250 | 0.0008951729 | 0.000895173696 | — |
| 300 | **0.0005614754** | 0.000561475754 | 0.000561 |

Every printed epoch agrees to ~8 significant figures — the full float32 print precision. **The
`fmul contract` / `fadd contract` risk flagged in G4 prep did not materialize:** IGC's fusion
choices produce no measurable divergence from the Apple/NVIDIA path.

| | PVC tile | A100 (Polaris P3a) |
|---|---|---|
| train time | **0.366 ms/epoch** | 0.116 ms/epoch |

**Caveat on that ratio (3.2×): it is host overhead, not kernel speed.** At N=256, H=16 the kernels
are tiny and the epoch is dominated by 15 launches; our path additionally pays an explicit
`zeCommandListAppendBarrier` after every launch, while Polaris got in-order execution free from CUDA
stream semantics. 04b's own header warns the GPU is not faster at this size on any backend. A
like-for-like host comparison waits for G5-lite.

**G4 COMPLETE.** Workload parity (level A) now holds for all three Polaris workloads: vector add,
matmul, and MLP training.

## ✅ G5.0 — Aurora emits the kernel IR itself; the Mac step is gone (uan-0007, 2026-09-15)

`bash gpu/g5_check_air_on_aurora.sh` → `gpu/g5_air_check.txt`. The G5-lite question was whether the
Metal-AIR step must happen on a Mac. It does not.

| Check | Result |
|---|---|
| Apple accelerator targets on the **Linux** build | ✅ `apple-m1` … `apple-m5` (+ `-metal4` variants) all listed |
| `max` in `gpu/.venv` | ❌ → ✅ after `uv add "max==26.5.0"`; **Mojo stayed 1.0.0 (ed45d567)** |
| AIR sidecar for `02_vecadd_gpu.mojo` emitted on uan-0007 | ✅ 6,563 B, triple `air64-apple-macosx15.0.0` |
| vs the Mac IR that **G2 actually ran** (`g2_vecadd.air.ll`) | ✅ **IDENTICAL** (modulo the mangled symbol name) |
| `04b_train_mlp_gpu.mojo` | ✅ **14 sidecars**, matching the Mac exactly |

**Why it works:** Mojo's Metal backend is a *code generator*, not a runtime. Nothing about emitting
AIR needs an Apple GPU — or macOS — to be present, and Modular ships the generator in the Linux
build. The only thing missing was the `max` package, which `gpu/.venv` lacked because it was created
with `uv add "mojo==1.0.0"` alone while the curriculum sources import `max.gpu.host` and `layout`.

**What this changes:**
- The pipeline is now **single-machine**: Aurora goes from unchanged `.mojo` source to running PVC
  kernels with no other computer involved.
- The "kernel IR was generated on a Mac" caveat comes off the claims list.
- Because the IR is byte-identical to what G2 ran, **no re-validation of G1–G4 is required** — this
  is the same input, produced locally, not a new build to re-qualify.
- G5-lite's "one-command build" is now actually possible: source → AIR → `air2spir` → SPIR-V → run.

**Version-pinning note:** `uv add "max==26.5.0"` was pinned deliberately (that is the version the Mac
pairs with Mojo 1.0.0). An unpinned add could move the compiler and invalidate the G1–G4 setup;
`mojo --version` was checked immediately afterwards and was unchanged.

## ✅ G5-lite (2 of 3) — one-command build from Mojo source on Aurora (uan-0007, 2026-09-15)

`bash build_kernels.sh curriculum/04b_train_mlp_gpu.mojo g4new g4_roles.map` → `gpu/g5_build.txt`.
**12 kernels built from the unchanged `.mojo` source in one command, all `ocloc`-validated for PVC**,
with no Mac and no hand-staged `.air.ll` files:

```
.mojo -> mojo build --emit asm --target-accelerator apple-m4   (14 AIR sidecars)
      -> air_roles.py    (hash-named sidecars -> 12 stable roles, verified against g4_roles.map)
      -> air2spir.py     (AIR -> SPIR + .spv.args manifest)
      -> clang -target spir64 -> llvm-spirv -> ocloc -device pvc
```

**The role problem, and why it needed solving.** Mojo names each sidecar with a mangling hash, and a
generic kernel instantiated per `TensorLayout` produces several sidecars sharing a base name — in
04b, `matmul_kernel` is both `K_fwd1` (X @ W1) and `K_fwd2` (a1 @ W2). Binding the wrong one computes
the wrong thing *silently*. `air_roles.py` identifies each kernel by a signature — base name,
argument counts, and the layout strides compiled into the body — which is stable across rebuilds
where the hash is not guaranteed to be. A source change that alters the kernel set makes the build
**fail with an error naming the offending kernels** instead of mismapping (failure path tested).

**Independent confirmation of the G4 mapping:** the resolver reproduced, from signatures alone, the
exact 12-way mapping derived by hand during G4 prep — `fwd1`→`1c8acd…`, `fwd2`→`c39ada…`,
`dW1`→`a7e596…`, `dW2`→`e260f8…`, `db1`→`eae5be…`, `db2`→`c5034b…`. Argument manifests match the
Mac-built `g4_*.spv.args` exactly; the `.spv` files differ by 4 bytes only because the trial used a
longer entry-point name (`g4new_` vs `g4_`).

**G5-lite status:** (1) AIR on Aurora ✅ G5.0 · (2) one-command build ✅ this section ·
(3) `DeviceContext`-compatible host layer — the remaining design work, whose hardest constraint is
the G4 lesson: it must supply in-order stream semantics itself.

## ✅ G5-lite regression — G4 re-run from source-built kernels (job 8828588, 2026-09-15)

`g4_*.spv` rebuilt by `build_kernels.sh` from `curriculum/04b_train_mlp_gpu.mojo` on uan-0007
(replacing the Mac-derived set), then `qsub g4_run.pbs`. Results are **identical**:

| | Mac-built kernels (job 8828472) | Source-built on Aurora (job 8828588) |
|---|---|---|
| forward check z1 / a1 / z2 | 0 / 0 / 0 wrong | 0 / 0 / 0 wrong |
| epoch 1 | 2.1783555 | 2.1783555 |
| epoch 300 / final | 0.0005614754 | **0.0005614754** |
| train time | 0.3659 ms/epoch | 0.3655 ms/epoch |
| `.spv` sizes | 4744, 4704, 4088, … | identical |

**Nothing in the working pipeline traces back to the Mac.** The `.spv` binaries are byte-for-byte the
same size as the Mac-built set, so the path is reproducible rather than merely equivalent: the same
Mojo source produces the same PVC kernels on Aurora alone.

# 🛠️ G5 — the `spirv64` backend (the project)

## ✅ G5-scope — open compiler built from source; GPU lowering mostly open (Mac, 2026-09-15)

**Main line — first step of the backend.** No Aurora time used.

### 🔨 Build (task 1)

- Clone: `github.com/modular/modular` @ `6417db28` (2026-09-15) →
  `/Users/rbutler/Desktop/LLMs/MOJO_STUFF/MOJO_POLARIS_AURORA/modular` (outside `AURORA/` so the sync
  script doesn't ship it). Compiler reports **`Mojo 1.2.0.dev0 (deadbeef)`**, ahead of the 1.0.0
  (`ed45d567`) we run on Aurora. LLVM = **24.0** (commit `4154b56c`), Bazel via `./bazelw`.
- **SPIR-V target enabled** with one line in `MODULE.bazel` using the repo's own hook
  (`bazel/public-patches/llvm_project.bzl` `extra_targets`):
  `llvm_configure.configure(extra_targets = ["SPIRV"])` → generated `llvm/targets.bzl` =
  `["AArch64", "RISCV", "SPIRV", "X86"]`.
- Attempt 1 ❌ (4 min): link errors in host tools (`strlen`, `operator new` undefined). Cause: the
  hermetic `ld64.lld` rejects the **macOS 27 SDK**'s `.tbd` stubs — `unknown architecture arm64e.x1`.
- Fix: `bazel/internal/cc-toolchain/macos_sysroot_repository.bzl` honours a new `MOJO_MACOS_SDK_PATH`;
  `local.bazelrc` = `build --config=build-mojo` + `common --repo_env=MOJO_MACOS_SDK_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.
  Both edits marked `AURORA PATCH`. Repo diff: 2 files, +18/−9.
- Attempt 2 ✅: `./bazelw build //Mojo:mojo` — **15.5 min, 8,718 actions** (M4 Max, 16 jobs) →
  `bazel-bin/Mojo/tools/mojo/mojo-full` (175 MB). `bazelw run //Mojo:mojo -- run hello.mojo` prints the
  right answer.
- Target probe with the built compiler:
  - `--target-triple nvptx64-nvidia-cuda` → `No available targets are compatible` (LLVM has no NVPTX here).
  - `--target-triple spirv64-unknown-unknown` → **LLVM accepts it**; Mojo stops at
    `target 'spirv64-unknown-unknown' is not supported by this build` (`Mojo/lib/Target/TargetTraits.cpp:183`).
    That registry is exactly where a `spirv64` backend registers.

### 🗺️ Where a backend plugs in (task 2)

Three peer interfaces, each dispatched by triple through a static registry
(`RegisterTargetTraits<>` / `RegisterTargetLowering<>` / `RegisterTargetBackend<>`):

| Layer | Interface | Host example | GPU-relevant hooks |
|---|---|---|---|
| metadata | `Mojo/lib/Target/TargetTraits.h:49` | `lib/Target/Host/HostTraits.{h,cpp}` (93 lines) | `isGPU`, `requiredStackAllocationAddressSpace`, `codegenTriple`, emission kinds, accelerator archs |
| MLIR → LLVM dialect | `Mojo/lib/Target/TargetLowering.h:73` | `lib/KGENToLLVM/Target/Host/HostLowering.cpp` (38 lines) | `populateLowerPOPToLLVMPatterns`, `isConvergentOp` (barrier), `markExportedKernel` / `isExportedKernel`, kernel-arg ABI (`lowerKernelArgToMemory`, `getKernelArgIndirectionType`, `getKernelByValArgAttrName`), `verifyOp` |
| LLVM → artifact | `Mojo/include/Mojo/Compiler/Target/TargetBackend.h:109` | `lib/Compiler/ObjectCompiler/Target/Host/HostBackend.{h,cpp}` (139 lines) | `isOffload`, `sharedMemoryAddressSpace`, `buildLLVMPipeline`, `linkRuntimeLibraries`, `attachCodegenAttributes`, `emitAssembly/emitObject/createArchive` |

The Host backend is a thin example — it overrides almost nothing. The **header comments name the GPU
cases** (NVPTX, barrier, threadgroup memory), so the interface was shaped by the closed GPU backends.
A target with `isBaseTarget() = true` skips the "please install MAX" gate
(`TargetTraits.cpp:26`, `:185`).

### 🔍 Open vs closed (task 3 — the sizing question)

| Piece | Status | Where |
|---|---|---|
| `thread_idx` / `block_idx` / `block_dim` / `global_idx` | **OPEN, Mojo source** — compile-time `is_nvidia_gpu/is_amd_gpu/is_apple_gpu` branches emit vendor intrinsic names (`llvm.air.thread_position_in_threadgroup.x`, …); `global_idx` = `b_idx*b_dim + t_idx` for all vendors | `Mojo/stdlib/std/_gpu/primitives/id.mojo:203,248,292,352,413` |
| `barrier()` | **OPEN, Mojo source** — same pattern (`llvm.air.wg.barrier`) | `max/mojo/max/gpu/sync/sync.mojo:116` |
| GPU target predicates | **OPEN** — `is_triple["air64-apple-macosx"]` etc.; `is_gpu()` is a fixed OR of the three | `Mojo/stdlib/std/sys/info.mojo:966,990,1245` |
| `stack_allocation[SHARED]` → shared global | **OPEN, generic** — `pop.global_alloc<gpu_shared>` lowered target-independently | `stack_allocation.mojo:88`, `Mojo/lib/KGENToLLVM/LowerPOPToLLVM.cpp:2221`, `POPOps.cpp:494` |
| Convergent (barrier) function attribute | **OPEN**, via `isConvergentOp` hook | `Mojo/lib/Transforms/InferFunctionAttrs.cpp:86` |
| Kernel outlining, kernel IDs, per-kernel split + sidecars | **OPEN, generic** | `Mojo/lib/Compiler/KGENCompiler.cpp:257,444`; `ObjectCompiler.cpp:1805`; `LowerKGENToLLVM.cpp:741` |
| Shared-memory globals kept out of split anchors | **OPEN**, via `isSharedMemoryGlobal` | `ObjectCompiler/LLVMIRUtils.cpp:332` |
| Stdlib vendor plugin selection | **OPEN** — `cuda`, `hip`, `metal` plugins chosen by target field `stdlib_plugin` | `Mojo/stdlib/std/_plugin/selector.mojo`, `KGENDialect/KGENAttrs.cpp:3803` |
| NVIDIA / AMD / Metal **backend implementations** (their Traits/Lowering/Backend, kernel ABI, entry marking) | **CLOSED** — linked into the prebuilt `mojo` binary (strings `air64-apple-macosx`, `nvptx64-nvidia-cuda`, `amdgcnspirv`); not in the tree | prebuilt `.venv/…/modular/bin/mojo` |
| **GPU runtime behind `DeviceContext`** | **CLOSED** — the Mojo side (`device_context.mojo`, 7,158 lines) is open but calls **116 `AsyncRT_*` functions** (DeviceContext 108 refs, DeviceGraphBuilder 33, DeviceBuffer 30, DeviceFunction 20, DeviceStream 20, …) whose implementations are not in the open `AsyncRT/` (CPU only) and not exported by any prebuilt dylib | `max/mojo/max/gpu/host/*.mojo` |

**Answer: the lowering is open and parameterized by target — much better than the "closed" case.** A
`spirv64` target means: (a) new Traits/Lowering/Backend classes (C++, modelled on the hooks above);
(b) an `is_intel_gpu()` branch in about five open Mojo files, emitting SPIR-V/OpenCL builtins
instead of AIR intrinsics — the mapping `air2spir.py` already uses; (c) an `intel` stdlib plugin. What
must be **written from scratch** is not lowering but the **runtime**: a Level Zero implementation of the
`AsyncRT_Device*` surface that `DeviceContext` calls, with in-order streams (the G4 lesson).

### 🧪 LLVM's in-tree SPIR-V backend at LLVM 24 (task 4)

- Handles OpenCL kernel builtins directly: `get_global_id`/`get_local_id`/… tests
  (`llvm/test/CodeGen/SPIRV/opencl/get_global_id.ll`), `barrier`/`work_group_barrier` →
  `OpControlBarrier` (`SPIRVBuiltins.td:686`), Intel split-barrier extension (`:705`), address spaces →
  `Workgroup`/`CrossWorkgroup` storage classes (`SPIRVUtils.cpp:385`). 242 codegen tests.
- A GPU vendor already uses it for compute: `spirv64-amd-amdhsa` triple with AMD decorations
  (`amdgcnspirv-atomic-metadata-decoration.ll`); the prebuilt Mojo binary contains `amdgcnspirv`.
- ⇒ The backend's last stage can plausibly be LLVM's SPIR-V `TargetMachine` inside Mojo, with no
  `llvm-spirv` and no LLVM-22/24 version gap. **Not yet proven on our kernels** — first thing to test.

### 🔧 Fork burden (task 5)

- The 50-commit shallow clone is **all from 2026-09-15** — a monorepo sync with heavy churn. 5 of those
  touch `Mojo/lib`/`include` in one day.
- Our compiler changes are additive (new target files + small registrations), plus ~5 stdlib/`max` Mojo
  files that need an `elif is_intel_gpu()` branch. Rebase conflicts should be local.
- Risk: `Mojo/docs/compiler/WorkingInOSRepo.md` — *"You can't use a locally built compiler to build any
  of the MAX targets."* `DeviceContext` lives in `max`. Whether the `max` Mojo package builds with our
  fork is **open item #1** for G5.

### 📐 Revised estimate and build order

Earlier: open → 4–6 months; closed → 9–15 months (one experienced LLVM/MLIR developer, full-time).
**Revised: ≈ 5–8 months.** The compiler side lands near the "open" case. The added cost is the closed
runtime (a Level Zero `AsyncRT_Device*` implementation) and the risk of building `max` with a fork.

Proposed order (each step is checked against the known-good `.spv` and G1–G4 results):
1. **`spirv64` target skeleton** — Traits/Lowering/Backend registered, `isGPU`, shared addrspace, emit
   via LLVM's SPIRV `TargetMachine`. Exit: `mojo build --target-triple spirv64…` emits a kernel sidecar.
2. **Stdlib intrinsics** — `is_intel_gpu()`, branches in `id.mojo`, `sync.barrier`, `is_gpu()`; kernel
   entry marking (`spir_kernel`) and argument ABI. Exit: `02` vecadd `.spv` validates with `ocloc`, matches
   `g2_vecadd.spv` behaviour.
3. **Run 02 / 03c / 04b kernels on PVC** — the existing Level Zero host serves only as a test harness here.
4. **Build `max` with the fork** (open item #1), then the **Level Zero `AsyncRT_Device*` runtime** with
   in-order streams, covering the subset 02/03c/04b use.
5. **Definition of done:** 02, 03c, 04b unmodified, host code and all, via `DeviceContext`.

🚦 **Gate:** Question 4 — who does the C++ compiler work? **Answered (Ralph, 2026-09-15): Claude writes
it, Ralph reviews** and runs the Aurora steps.

## 🧱 G5 step 1 — `spirv64` target skeleton emits a valid SPIR-V kernel (Mac, 2026-09-15)

**Main line.** New files in the clone (each marked `AURORA PATCH (G5)`; the existing globs in
`Mojo/BUILD.bazel` pick them up, so no BUILD edits were needed):

| File | What |
|---|---|
| `Mojo/lib/Target/IntelGPU/IntelGPUTraits.{h,cpp}` | name `intel`, matches `Triple::spirv64`, `isGPU`, extensions `.spvasm/.spirv.ll/.spv/.spirv.bc`, accelerator arch `intel-pvc`, `isBaseTarget` (no MAX gate) |
| `Mojo/lib/KGENToLLVM/Target/IntelGPU/IntelGPULowering.cpp` | exported offload functions → `spir_kernel` calling convention |
| `Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUBackend.cpp` | `isOffload`, per-exported split, shared-memory addrspace 3, `asm` = SPIR-V text / `object` = SPIR-V binary through LLVM's SPIR-V `TargetMachine`; drops CPU name/features for the TM |

Incremental rebuild: 1 min 19 s. `mojo build --print-supported-accelerators` now lists
**`Intel GPU (SPIR-V / Level Zero): intel-pvc`**.

Test: `gpu/g5_skel_compile.mojo` uses the open stdlib's `std.compile.compile_info` with an inline
`#kgen.target<triple = "spirv64-unknown-unknown", arch = "intel-pvc", …>` (LLVM's generic spirv64 data
layout). **No MAX package needed.** Kernel: `add_one(p: Pointer[Float32], n: Int32)` — no GPU builtins yet.

- Run 1 ❌ `Calling convention requires void return type` — the test function returned a value; OpenCL
  kernels return `void` (correct behaviour, test fixed). Also `'intel-pvc' is not a recognized processor`
  → fixed with `adjustOptionsForTargetMachine`.
- Run 2 ✅ LLVM IR: `define dso_local spir_kernel void @"…add_one…"(ptr, i32)`, triple `spirv64-unknown-unknown`.
  SPIR-V: `OpCapability Kernel`, `OpMemoryModel Physical64 OpenCL`, `OpEntryPoint Kernel`, generator
  **"LLVM SPIR-V Backend; 24"**, version 1.4.
- Binary (`emission_kind="object"`, 1,168 bytes) → `gpu/g5_add_one.spv`: **`spirv-val --target-env spv1.4`
  passes** (SPIRV-Tools 1.4.357 via Homebrew). `--target-env opencl2.2` rejects it only for being version
  1.4 rather than 1.2.

**Noted for step 2:**
- Pointer parameters come out as `OpTypePointer Function`, because Mojo's `AddressSpace.GENERIC` = 0.
  OpenCL kernel buffers normally live in `CrossWorkgroup` (addrspace 1); Metal's AIR also uses
  `addrspace(1)` for buffers. The kernel argument ABI has to match the TileTensor holders.
- The entry-point name is the full mangled Mojo name, which the runtime must look up verbatim.
- `IntelGPUTraits` does not yet set `requiredStackAllocationAddressSpace`, and there is no stdlib target
  entry, `is_intel_gpu()`, or intrinsic branches yet.

**✅ IGC accepts it (uan-0007, 2026-09-15):** `ocloc compile -spirv_input -device pvc` on `g5_add_one.spv`
(1,168 bytes) → `Build succeeded`, `g5_add_one_pvc_pvc.bin` 4,616 bytes. **The first SPIR-V emitted by
our Mojo backend compiles for PVC with Intel's GPU compiler.** It also shows IGC *compiles* kernel pointer
parameters in `Function` storage (Mojo addrspace 0); whether they *run* correctly is still open.
(The script's `Error: Device name missing.` came from a wrong `ocloc -v` version probe, now
`ocloc query OCL_DRIVER_VERSION`. It was harmless.)

## 🧵 G5 step 2 (part 1) — thread indexing, shared memory and barrier lower to SPIR-V (Mac, 2026-09-15)

**Main line.** Stdlib patches in the clone (each marked `AURORA PATCH (G5)`):

| File | Change |
|---|---|
| `Mojo/stdlib/std/sys/info.mojo`, `sys/__init__.mojo` | `is_intel_gpu()` = triple `spirv64-unknown-unknown`; added to `is_gpu()`; exported |
| `Mojo/stdlib/std/_gpu/primitives/id.mojo` | Intel branches: `thread_idx` → `llvm.spv.thread.id.in.group`, `block_idx` → `llvm.spv.group.id`, `block_dim` → `llvm.spv.workgroup.size`, `grid_dim` → `llvm.spv.num.workgroups` (`.i64`, dim index arg); `global_idx` reuses the generic formula |
| `max/mojo/max/gpu/sync/sync.mojo` | `barrier()` Intel branch → `llvm.spv.group.memory.barrier.with.group.sync` (**untested**: `max` isn't built with the fork yet) |

Why `llvm.spv.*`: LLVM 24 defines these as real intrinsics (`llvm/include/llvm/IR/IntrinsicsSPIRV.td:82–99,173`).
The SPIR-V instruction selector lowers them to vec3 `BuiltIn` input variables and `OpControlBarrier`
(`SPIRVInstructionSelector.cpp:5498–5540, 3649`). That fits the stdlib's existing
`llvm_intrinsic[...]` vendor pattern exactly.

Tests (each compiles two variants, A = patched stdlib path, B = OpenCL builtin calls as `air2spir` produced):
- `gpu/g5_step2_ids.mojo`: vecadd with `global_idx.x`. IR:
  `group.id * workgroup.size + thread.id.in.group`. A and B both lower to
  `BuiltIn LocalInvocationId / WorkgroupId / WorkgroupSize`; B also marks them
  `LinkageAttributes … Import` (the Intel-tools form).
- `gpu/g5_step2_barrier.mojo`: shared-memory block sum with `stack_allocation[SHARED]` + barrier. IR: an
  `internal addrspace(3) global [16 x float]`, and the kernel is marked **`convergent`** (the open
  `InferFunctionAttrs` propagated it). SPIR-V: module-scope **`OpVariable … Workgroup`**, and
  `OpControlBarrier 2 2 264` (A) / `272` (B).
- Run 1 ❌: `name + ".i64"` in the helper became a runtime `String` concatenation inside the kernel
  (`String._add` → `alloc`). Fixed by passing literal intrinsic names.
- All four `.spv` (`g5_ids_{a,b}`, `g5_barrier_{a,b}`) **pass `spirv-val --target-env spv1.4`**.

**Open questions for IGC (the `ocloc` check answers them):**
1. Pointer kernel parameters in `Function` storage (Mojo `GENERIC` = addrspace 0) — accepted, or must they be `CrossWorkgroup` (1)?
2. Module-scope `Workgroup` variable for shared memory. G3 used `__local` kernel args instead.
3. Barrier memory semantics 264 vs 272.

**Version finding:** the compiler source exists only on `main` (`1.2.0.dev`). Tag `mojo/v1.0.0`
(`b4497b7`, 2026-08-11, "Pin lockfiles to Mojo 1.0.0, MAX 26.5.0") has the stdlib but no `Mojo/lib`. In
`1.2.0.dev`, `global_idx` is public from `max.gpu`, while the curriculum (1.0.0) imports `std.gpu`.
→ "unmodified" needs either a curriculum migration or compatibility re-exports in the fork.

**✅ IGC verdict (uan-0007, driver 25.18.33578):** all 5 `.spv` build for PVC — `g5_add_one` 4,616 B,
`g5_ids_a` 7,912, `g5_ids_b` 8,088, `g5_barrier_a` 8,392, `g5_barrier_b` 8,392. IGC accepts the
`llvm.spv.*` lowering (A), module-scope `Workgroup` shared memory, and both barrier semantics.
**Stdlib keeps variant A.**

## 🧩 G5 step 2 (part 2) — 1.0 compatibility, the unchanged curriculum kernel, kernel ABI (Mac, 2026-09-15)

**Decision (Ralph):** option (a), compatibility re-exports in the fork.
- New `Mojo/stdlib/std/gpu/__init__.mojo` (re-exports `std._gpu`: `global_idx`, `thread_idx`,
  `block_idx`, `block_dim`, `grid_dim`, …) and `std/gpu/host/__init__.mojo` (`get_gpu_target`). Both match the
  `mojo/v1.0.0` tag's export lists. `g5_step2_ids.mojo` now imports `from std.gpu import global_idx`, and
  its `g5_ids_a.spv` is **byte-identical**.

**Open item #1, first answer: the fork compiles `max` and `layout` *source*.** The unchanged
`MOJO_CURRICULUM/02_vecadd_gpu.mojo` (imported through a symlink `curriculum_02.mojo`, with
`-I modular/max/mojo -I modular/max/kernels/src`) elaborates, including `TileTensor` and
`max.gpu.host.DeviceContext`. There were 35 deprecation warnings (`bitcast`, `load`, `store`) and no errors. `compile_info`
on its `vecadd_kernel` for `spirv64` → `spir_kernel`. (Building `max` as a *package* is still untested.)

**Kernel argument ABI.** With no hooks the kernel took `{ ptr, {{{}},{{}}} }` by value ×3 plus `i32`. SPIR-V can't
pass pointer-holding or empty structs by value. Reading `DeviceContext._call_with_pack_checked`
(`max/mojo/max/gpu/host/device_context.mojo:3369`) showed the non-Metal path packs each argument's device
bytes into its own 8-byte-aligned slot and hands the runtime `dense_args_addrs` (one pointer per argument) and `argc`.
**Chosen ABI: every kernel argument by pointer, `ptr addrspace(1)`**, so the Level Zero runtime copies
each slot into device memory and binds it without knowing any layout. The alternative, flattening
structs into separate params, needs a per-kernel layout manifest and stays a later optimization.
Implemented in `IntelGPULowering.cpp`:
- `lowerKernelArgToMemory`: non-pack `KGEN::StructType` → `!kgen.pointer<T, 1>`.
- `getKernelArgIndirectionType`: `ImmReg`/`OwnedReg` `SIMDType`/`PointerType` → `!kgen.pointer<T, 1>`.
- `getKernelByValArgAttrName` = `llvm.byref`, which carries the pointee type without copy semantics.
- Run 1 ❌ `cast<Ty>() argument of incompatible type` (abort): returning a pointer for *every* type was too
  broad → restricted to structs. Run 2 ❌ `expected valid attribute name`: exported kernels with memory
  args need the by-value attribute name → `llvm.byref`. Run 3 ✅.

Result for unchanged `vecadd_kernel`:
`spir_kernel void (ptr addrspace(1) byref({ptr,…}) ×3, ptr addrspace(1) byref(i32))`. That is **exactly the G2 host's
holder layout.** `gpu/g5_emit_kernel.mojo` → `gpu/g5_vecadd.spv` (9,264 B, passes `spirv-val`). SPIR-V params
are `CrossWorkgroup` pointers. Entry point = the full Mojo linkage name, **3,889 chars** (`g5_vecadd.spv.name`).

**✅ IGC builds it (uan-0007):** `ocloc -device pvc` on `g5_vecadd.spv` (9,264 B) → `Build succeeded`,
`g5_vecadd_pvc_pvc.bin` 33,168 B.

## ⚠️ G5 step 3 run 1 — kernel runs, writes nothing (job 8829357, 2026-09-15)

`qsub -v MOJOFILE=g5_vecadd_lz.mojo g1_run.pbs` on x4010c1s7b0n0:
- The 3,889-char entry point was **accepted**. `zeModuleCreate` + `zeKernelCreate` took 47.5 ms.
- The launch ran with **no fault**, at 0.00468 ms/pass.
- **999,999 / 1,000,000 mismatches.** Every `c[i]` is 0.0; the one "match" is `c[2]`, whose expected value is 0.0.

Diagnosis from `spirv-dis`: the logic is right (global id, `size` check, loads, add, store), but the data pointers
loaded from the holders were `OpTypePointer Function`, i.e. thread-private storage. A store through
private memory has no visible effect, so IGC may drop it: "no writes, no fault". In G2's working Metal-derived kernel,
the same pointers were `addrspace(1)`.

**Fix — `Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUAddressSpaces.{h,cpp}`.**
`moveGenericPointersToGlobalAddressSpace` runs in `IntelGPUBackend::emitAssembly/emitObject` just before
SPIR-V codegen. It rebuilds the module with `CloneFunctionInto` and a type remapper, so every addrspace-0 pointer
(signatures, instructions, aggregate fields, globals, `byref`/`byval` attribute types) becomes
addrspace 1. It also keeps surviving allocas in addrspace 0 behind an `addrspacecast`, re-mangles pointer-overloaded
intrinsics, and runs `verifyModule`. Metal's closed backend has to do the equivalent to produce its AIR.
Result: `g5_vecadd.spv` data pointers are now `CrossWorkgroup`, and it passes `spirv-val`. The step-2 kernels
re-emit and validate, with shared memory still `Workgroup`.

## ✅ G5 step 3 run 2 PASS — our `spirv64` backend's kernel runs correctly on PVC (job 8829448, 2026-09-15)

```
=== G5: Mojo spirv64 backend kernel -> Level Zero -> Intel GPU ===
  device      : Intel(R) Data Center GPU Max 1550      (x4100c1s3b0n0, FLAT, ZE_AFFINITY_MASK=0)
  entry point : 3889 chars
  kernel src  : MOJO_CURRICULUM/02_vecadd_gpu.mojo (vecadd_kernel, unchanged)
  compiler    : source-built Mojo, spirv64 target (LLVM SPIR-V codegen)
  N 1,000,000 · block 256 · 3907 groups
  mismatches  : 0 / 1000000        RESULT : PASS
  module load : 39.75 ms (zeModuleCreate + zeKernelCreate)
  GPU time    : 0.00633 ms/pass (kernel only)
  c[1], c[999999] : 1.0 -999997.0
```

**The first Mojo GPU kernel compiled by our Intel backend in the open-source Mojo compiler runs correctly on
an Aurora PVC tile.** The toolchain was unchanged curriculum source → source-built Mojo (`1.2.0.dev`) with our
`IntelGPU` target → LLVM's in-tree SPIR-V codegen → Level Zero. There was no Metal backend, `air2spir.py`, clang or `llvm-spirv`.
Timing matches the Metal-route G2 (0.00664 ms/pass) and G1 OpenCL C (0.00561). The host is still the
G2 Level Zero harness, **not** `DeviceContext`.

Run 2 differed from run 1 only by the address-space remap: `ocloc` passes, and the PVC binary is now 33,208 B,
up from 33,168 B.

## 🧰 G5 step 3 prep — 03c/03d coarse matmul (shared memory + barrier) through our backend (Mac, 2026-09-15)

**Main line.** `gpu/g5_emit_matmul.mojo` compiles the unchanged `03d_matmul_check.mojo` (N=256) and
`03c_matmul_coarse.mojo` (N=2048) kernels; the curriculum files are imported via symlinks `curriculum_03{c,d}.mojo`.

- Run 1 ❌ `function instantiation failed`: the kernel reached `layout`'s `row_major(shape)` →
  `String.__init__` → `alloc` → `abort` → `print`, host-only code inside a GPU kernel. Cause: the
  stdlib compiles assertion, abort and poison messages and heap-`String` refcounting **out** on Apple GPU
  (Metal has no kernel printf). Intel fell into the generic-GPU print path, written for CUDA/ROCm printf,
  or the host path.
- Fix (stdlib, `AURORA PATCH (G5)`): **treat Intel like Apple** in
  `builtin/debug_assert.mojo` (assertions off in kernels), `os/os.mojo` (`_abort_report` and the `abort`
  message), `memory/_poison.mojo` (uninit-read report), and `collections/string/string.mojo` (`_add_ref`/`_drop_ref`
  no-ops). An OpenCL `printf` path could re-enable messages later. Regression: `g5_vecadd.spv` is
  **byte-identical** after the change.
- Run 2 ✅ `g5_check.spv` 86,104 B, `g5_coarse.spv` 86,328 B; both pass `spirv-val`. Each has
  2 `OpControlBarrier`, 2 module-scope `Workgroup` `float[1024]` arrays (BM×BK and BK×BN), and no function
  calls or imports. Vector types are `v3ulong` and `v16float` only: LLVM split the kernel's `SIMD[float32, 64]` register tile
  into legal 16-wide vectors (G3 lesson 5 does not recur).
- Harness `gpu/g5_matmul_lz.mojo` = `g3_matmul_lz.mojo` with a two-variant table and entry-point names from
  `.spv.name`. The emitter's `.spv.args` lists 3 buffers and no `local` args, since shared memory is a
  Workgroup variable. It compile-checks with Mojo 1.0.0.

**✅ IGC builds both (uan-0007):** `g5_coarse_pvc_pvc.bin` is 154,504 B. For both kernels IGC warns
`compiled SIMD32 allocated 128 regs and spilled around 248` and `[RetryManager] Start recompilation of the kernel`.
That is the **same register-spill warning the Metal-route coarse kernel got in G3**: a performance issue (the Apple/NVIDIA-tuned
8×8 register tile), not correctness. The 3,889+-char mangled entry points make IGC's messages unreadable;
short entry-point names are a later cleanup.

## ✅ G5 step 3 — 03c/03d (shared memory + barrier) PASS via our backend (job 8829553, 2026-09-15)

`qsub -v MOJOFILE=g5_matmul_lz.mojo g1_run.pbs` on x4112c2s7b0n0:

| Variant | N | Correctness | Avg time | GFLOP/s | Metal route, G3 (job 8827590) |
|---|---|---|---|---|---|
| check (03d) | 256 | **FULL 0 / 65,536**, max rel err 0.0 ✅ | 0.101 ms | 332 (too small to mean anything) | 0/65,536, 0.10 ms, 333 |
| coarse (03c) | 2048 | SAMPLED 0 / 260, max rel err 0.0 ✅ | 9.26 ms | **1,856** | 0/260, 9.03 ms, **1,903** |

Warmup (JIT) was 0.65 ms (check) and 11.7 ms (coarse). The entry points are 4,827 and 4,863 chars; both were accepted.

**Our backend's SPIR-V for barrier (`OpControlBarrier`) and module-scope `Workgroup` shared memory
synchronizes correctly on PVC.** The full element-by-element check at N=256 spans a 2×2 grid, with 32 K-slabs × 2
barriers. Speed matches the Metal-route kernel within 2.5% (one run each, so that is within noise). The same IGC
SIMD32 register-spill warning applies to both routes, so tuning is a kernel-geometry question, not a backend one.
The host is still the G3 Level Zero harness.

## 🧰 G5 step 3 prep — 04b's 12 kernels through our backend (Mac, 2026-09-15)

**Main line.** 04b names its kernels with comptime locals inside `main()`, which can't be imported. So
`gpu/g5_emit_mlp.mojo` imports 04b's generic kernels and module-level layouts (via symlink
`curriculum_04b.mojo`) and instantiates them exactly as 04b's `main()` does. As in G4, the four SGD launches share
one kernel. Run 1 ❌: two Mojo errors in my emitter (`out` is a reserved argument name; reusing one variable for
differently-typed `compile_info` results). Fixed with a generic `_emit[kernel]` helper. Run 2 ✅:

| role | bytes | args (buffers + consts) | | role | bytes | args |
|---|---|---|---|---|---|---|
| fwd1 | 8,020 | 3 + 3 | | dW2 | 8,096 | 3 + 3 |
| fwd2 | 8,084 | 3 + 3 | | db2 | 5,612 | 2 + 2 |
| biasrelu | 7,496 | 3 + 2 | | da1 | 8,116 | 3 + 3 |
| addb2 | 5,016 | 2 + 1 | | relugrad | 7,412 | 3 + 2 |
| dz2 | 7,080 | 3 + 2 | | dW1 | 8,024 | 3 + 3 |
| sgd | 3,440 | 2 + 2 | | db1 | 5,692 | 2 + 2 |

All 12 pass `spirv-val`. SPIR-V parameter counts equal the manifests, and there are no calls or imports and no `Function`-storage
pointers. Scalars arrive as `CrossWorkgroup` `uint`/`float` pointers, which is **exactly G4's binding** (holders +
4-byte cells). Harness `gpu/g5_train_mlp_lz.mojo` is `g4_train_mlp_lz.mojo` with `g5_<role>.spv` and
entry-point names from `.spv.name`. It compile-checks with Mojo 1.0.0.

**✅ IGC builds all 12 (uan-0007):** `ocloc -device pvc` passes for every kernel.

## ✅ G5 step 3 — 04b MLP training PASS via our backend (job 8829859, 2026-09-15)

`qsub -v MOJOFILE=g5_train_mlp_lz.mojo g1_run.pbs` on x4020c0s7b0n0. The upload check was OK, and all 15 launches bound over 12
kernels with no manifest errors.

Forward verification: `z1` 0/4096 wrong, `a1` 0/4096 wrong (2108 exact zeros = genuine ReLU zeros), `z2`
0/256 wrong, max rel err **0.0** at every stage.

| epoch | **our spirv64 backend** (job 8829859) | Metal route, G4 (job 8828472) | CPU float32 reference |
|---|---|---|---|
| 1 | 2.1783555 | 2.1783555 | 2.17835617 |
| 50 | 0.016860535 | 0.016860535 | 0.016860541 |
| 100 | 0.0056486796 | 0.0056486796 | 0.00564867724 |
| 150 | 0.0029576812 | 0.0029576812 | 0.00295768236 |
| 200 | 0.0016037121 | 0.0016037121 | 0.00160371314 |
| 250 | 0.0008951729 | 0.0008951729 | 0.000895173696 |
| 300 | **0.0005614754** | 0.0005614754 | 0.000561475754 |
| ms/epoch | 0.370 | 0.366 | — |

**Digit-for-digit identical to the Metal-route run at every printed epoch**, and it matches the NumPy float32 reference to print
precision. **All three `MOJO_CURRICULUM` GPU workloads (02 vecadd, 03c coarse matmul with shared memory + barrier,
04b MLP training) now run correctly on PVC from kernels compiled by our Intel backend in the open-source
Mojo compiler.** Their host sides are still the G2/G3/G4 Level Zero test harnesses, not `DeviceContext`. Step 3 is
closed.

## 🧭 G5 step 4 — what remains for "unmodified, host code and all" (analysis, 2026-09-15)

1. **The `DeviceContext` runtime.** `max/mojo/max/gpu/host/device_context.mojo` (open Mojo) talks to the device only
   through closed `AsyncRT_*` C functions: `AsyncRT_DeviceContext_create`, `createBuffer_*`, `HtoD/DtoH_async`,
   `createStream`, `loadFunction`, `enqueueFunctionDirect`, `synchronize`, … (116 names in `max/gpu/host`).
   The Metal branch uses the same `AsyncRT_*` calls; only its argument packing differs
   (`_is_apple_gpu` at `device_context.mojo:3257,3469`). Options:
   - (i) **A Level Zero library that implements the subset of `AsyncRT_*` 02/03c/04b use** (C ABI, in-order streams
     via a barrier per launch as in G4, arguments as slot copies per our backend ABI). `max` Mojo stays nearly unchanged.
   - (ii) An Intel branch in every `DeviceContext` method, calling `mojo_intel_gpu` directly. That is invasive: `create`,
     buffers, copies, streams and launch each need their own branch.
2. **The fork must run on Aurora.** User programs are compiled by the forked compiler on the machine that runs
   them. It currently builds only on the Mac (arm64 macOS). A Linux x86_64 build is needed on a UAN or a compute node;
   Bazel downloads go through `proxy.alcf.anl.gov`.
3. **`--target-accelerator intel-pvc` end to end:** the stdlib GPU target entry (`_builtin_targets.mojo`), an `intel`
   stdlib plugin, `has_accelerator()` detecting a Level Zero GPU, and `DeviceContext.compile_function` producing
   our `.spv` at run time.
4. **`max` built with the fork** (open item #1). Its source compiles; packaging it is untested.

**Decisions (Ralph, 2026-09-15), all as recommended:**
- (1) runtime approach (i): implement the needed `AsyncRT_*` subset over Level Zero
- (2) in **C++**
- (3) build the fork on a **`next-eval` compute node**
- (4) start the stdlib target entry and `has_accelerator()` on the Mac meanwhile

**Fork-on-Aurora kit:**
- `gpu/g5_make_overlay.sh` (Mac) packs our 18 patched or new repo files into `gpu/g5_fork_overlay.tar.gz`
  (59,662 B, sha256 `49a26d51…`).
- `gpu/g5_prepare_fork.sh` (uan-0007) clones `modular` at `6417db28` into `MOJO_WORK/modular`, applies
  the overlay, and writes a Linux `local.bazelrc` (`--config=build-mojo`; output root, disk cache and repo cache under
  `MOJO_WORK/bazel`).
- `gpu/g5_build_fork.pbs` (2 h) builds with the ALCF proxy, runs a hello smoke test, lists accelerators, and re-emits
  `g5_vecadd.spv` on Aurora to `cmp` it with the Mac build.
- ⚠️ **Lesson:** the repo's `.gitignore` entry `target/` is matched case-insensitively on macOS, so
  **git ignores `Mojo/lib/**/Target/IntelGPU/`.** `git status`/`git diff` don't show the backend's C++ files, which is why the overlay uses an
  explicit list.

On uan-0007, `g5_prepare_fork.sh` ✅ cloned at `6417db28` and applied all 18 files.

**✅ 4a — the fork builds on Aurora (job 8829889, 2026-09-16).** x4112c7s5b0n0, 208 cores / 1,134 GB:
- `./bazelw build //Mojo:mojo` → **3,062 s (51 min)**, 11,324 actions, cold caches (53 disk-cache hits),
  downloads through the ALCF proxy. No Linux-specific patches were needed beyond the overlay.
- Smoke: `Mojo 1.2.0.dev0 (deadbeef)` runs a hello program on the compute node;
  `--print-supported-accelerators` lists **`Intel GPU (SPIR-V / Level Zero): intel-pvc`**.
- **Cross-platform check: the `g5_vecadd.spv` emitted on Aurora is byte-identical to the Mac's** (9,264 B).
  Our backend produces the same SPIR-V on macOS/arm64 and Linux/x86_64.
- That build used overlay v1 (18 files); the 4b patches need the newer overlay (21 files, `a86f02e7…`) and
  an incremental rebuild.

## ✅ G5 step 4b — `--target-accelerator intel-pvc` end to end on the Mac (2026-09-15)

**Main line.** Patches, each marked `AURORA PATCH (G5)`:
- `std/_gpu/host/_builtin_targets.mojo`: `TargetAccelerator[IntelPVC, _intel_pvc_target, ["intel-pvc"]]`.
  `IntelXeHPCFamily` has warp 16, max work group 1024, and ~128 KB shared memory per group. `_intel_pvc_target` is the `spirv64` `#kgen.target`.
  `IntelPVC` has `api="level_zero"`, `arch_name="intel-pvc"`.
- `std/sys/info.mojo` / `sys/__init__.mojo`: `Vendor.INTEL_GPU`; `_vendor_from_arch` classifies `"intel"`;
  `has_intel_gpu_accelerator()`. (`has_accelerator()` was already true for any non-empty accelerator arch.)
- `Mojo/lib/Target/TargetTraits.{h,cpp}`: `requireMaxForAcceleratorRequest` — which aborted **every**
  `--target-accelerator` with "please install MAX" — now exempts an arch owned by a base (open) target.

**Result — the normal toolchain command on the unchanged curriculum programs:**
```
mojo build --target-accelerator intel-pvc --emit asm -I modular/max/mojo -I modular/max/kernels/src \
    MOJO_CURRICULUM/{02_vecadd_gpu,03c_matmul_coarse,04b_train_mlp_gpu}.mojo
```
- All three compile, host code and all, and emit kernel sidecars `host__<module>_<kernel>_<hash>.spvasm`:
  1 for 02, 1 for 03c, **14 for 04b** (the same count as the Metal route). All 16 assemble and pass `spirv-val`.
- The 02 sidecar's kernel body is **identical, instruction for instruction (ids aside), to `g5_vecadd.spv`**, which passed on PVC in job 8829448.
- **No emitter scripts are needed any more;** the compiler's own offload path produces our kernels.

**The runtime surface, measured.** The three host programs (`host.s`) reference just **21 `AsyncRT_*` symbols**.
Three are `CPUDevice` calls from the stdlib startup code (shipped with the compiler). **18 are the Level Zero runtime's job (4d):**

| Group | Functions (C prototypes from `max/mojo/max/gpu/host/device_context.mojo` comments) |
|---|---|
| context | `const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *api, int id)` · `_retain/_release(ctx)` · `int64_t _id(ctx)` · `void _deviceApi(llvm::StringRef *result, ctx)` · `void _strfree(const char*)` · `const char *_synchronize(ctx)` |
| buffers | `const char *_createBuffer_async(const DeviceBuffer **result, void **device_ptr, ctx, size_t len, size_t elem_size)` · `_createHostBuffer(same)` · `_DtoD_async(ctx, dst, src)` · `_setMemory_async(ctx, dst, uint64_t val, size_t val_size)` · `AsyncRT_DeviceBuffer_bytesize/_context/_retain/_release(buf)` |
| functions | `const char *_loadFunction(const DeviceFunction **result, ctx, const char *moduleName, const char *functionName, const char *data, size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel, int32_t optimizationLevel)` · `_enqueueFunctionDirect(ctx, fn, uint gx, gy, gz, bx, by, bz, uint sharedMemBytes, attrs, uint nAttrs, void **args, uint argCount, uint64_t *argSizes)` · `AsyncRT_DeviceFunction_release(fn)` |

- `DeviceFunction` requires `emission_kind == "object"`, so `loadFunction`'s `data` is our backend's **binary
  SPIR-V**, which goes straight to `zeModuleCreate`.
- `DeviceContext()` defaults `api` to `GPUInfo.api` = `"level_zero"`.
- Kernel arguments arrive exactly as analysed for our ABI: `args[i]` points to each argument's packed device bytes. The runtime copies each one into
  a USM cell and binds it with `zeKernelSetArgumentValue(i, 8, &cell)`.
- Overlay regenerated with the 3 new files: 21 files, 71,067 B, sha256 `a86f02e7…`.

## 🧱 G5 step 4d — the Level Zero runtime, written (Mac, 2026-09-15)

**Main line.** `gpu/g5_runtime/mojo_level_zero_rt.cpp` (~560 lines) implements the 18 `AsyncRT_*` entry
points over Level Zero. It compile-checks with `clang++ -std=c++17 -fsyntax-only` against the Level Zero
header LLVM bundles for its OpenMP offload plugin (there is no `level-zero` Homebrew formula); the real
build uses Aurora's `/usr/include/level_zero`.

Design, all of it from the G1–G4 lessons:
- **In-order stream:** an immediate command list is not in-order, so a `zeCommandListAppendBarrier` follows
  every launch, copy and fill. That is the G4 bug, fixed once inside the runtime instead of in user code.
- **Kernel arguments:** `DeviceContext` hands over one host pointer per argument and **no sizes** on this
  path, so `loadFunction` reads the sizes **out of the SPIR-V module** (a ~120-line type walker: each kernel
  parameter is a pointer, and its pointee's size is what to copy). Each argument is copied into a
  device-visible cell from a 1 MiB bump arena, and the cell address is bound with
  `zeKernelSetArgumentValue`. The arena resets on `synchronize`, so cells outlive the launches that read them.
- **Indirect access:** every kernel gets `zeKernelSetIndirectAccess(HOST|DEVICE|SHARED)`, since buffers are
  reached through the argument holders (the G2 lesson).
- **Buffers:** `createBuffer_async` → `zeMemAllocDevice`; `createHostBuffer` → `zeMemAllocHost` (this is what
  `map_to_host` uses); `DtoD_async` → `zeCommandListAppendMemoryCopy`; `setMemory_async` → `zeCommandListAppendMemoryFill`
  with the pattern kept in the arena. Frees synchronize first, since commands in flight may still read the allocation.
- **Refcounts** on context, buffer and function mirror the retain/release pairs `DeviceContext` calls.
- Errors are `malloc`'d strings, as `AsyncRT_DeviceContext_strfree` expects; `zeModuleCreate` failures carry the IGC build log.

**Wiring:** no change to `max` is needed. `mojo run` resolves JIT symbols through its linker passthrough:
`-Xlinker -L<dir> -Xlinker -lmojo_level_zero_rt` makes the JIT `dlopen` the library
(`Mojo/tools/mojo/Common/XlinkerResolution.h`).

New files: `gpu/g5_build_runtime.sh` (uan-0007 build) and `gpu/g5_e2e_run.pbs` — the end-to-end job that runs
the **unmodified** 02/03c/04b on a PVC tile through the fork plus this runtime.

**✅ Builds on Aurora (uan-0007, 2026-09-16):** `g5_build_runtime.sh` with **icpx 2026.1.0** against
`/usr/include/level_zero/ze_api.h` and `/usr/lib64/libze_loader.so.1` → `libmojo_level_zero_rt.so`
(52,232 B), exporting **all 18** `AsyncRT_*` symbols, with no warnings.

## ✅⚠️ G5 step 4 — end-to-end run 1: all three unmodified programs PASS, but slow (job 8830xxx, 2026-09-16)

`qsub g5_e2e_run.pbs` — unmodified `02_vecadd_gpu`, `03c_matmul_coarse`, `04b_train_mlp_gpu`, compiled by the
fork on Aurora and run through our Level Zero runtime (`e2eout1.txt`, 1,256 lines):

| program | correctness | this run | harness runs (G5 step 3) |
|---|---|---|---|
| 02 vecadd | **0 / 1,000,000 mismatches, PASS** | 15.22 ms/pass | 0.0063 |
| 03c matmul | **C[0,0] = 4096.0 as expected** | 285 ms, 60 GFLOP/s | 9.26 ms, 1,856 |
| 04b training | **loss 2.1783555 → 0.0005614754**, every printed epoch identical to G4 | 220 ms/epoch | 0.370 |

**Correctness is complete: the whole `DeviceContext` path — context, buffers, host mapping, fills, kernel
loading, launches and ordering — works.** Three unmodified Mojo GPU programs, host code and all, ran on a
PVC tile through our compiler backend and our runtime.

**The performance gap is one bug.** Per launch: 15.2 ms (02), 14.7 ms (04b: 220/15), 285 ms (03c). Those are
IGC *compile* times, not execution times, and they scale with kernel size.
`DeviceContext.enqueue_function` constructs a **fresh `DeviceFunction` on every call**
(`device_context.mojo:4792`), so it calls `AsyncRT_DeviceContext_loadFunction` per launch — and our
runtime rebuilt the SPIR-V module every time. The closed runtime must cache compiled modules.

**Fix:** a compiled-kernel cache in `loadFunction`, keyed by entry-point name + module bytes, holding one
reference per entry and destroyed with the context (`mojo_level_zero_rt.cpp`). Compile-checked; needs a
rerun. Expect ~0.006 ms/pass (02), ~9 ms (03c) and ~0.4 ms/epoch (04b), i.e. the harness numbers plus a
one-time compile.

**Runtime behaviour verified by run 1; risks that are now settled:** Level Zero *does* snapshot kernel
arguments at append (04b relaunches one kernel with different arguments and the loss curve is exact); the
`deviceApi` `{pointer, length}` out-parameter is right; and host↔device `DtoD_async` copies correctly
(`map_to_host` round-trips).

## 🏁 G5 — SOURCE PARITY: end-to-end run 2 PASS, at harness speed (2026-09-16, `e2eout2.txt`)

Same job after the compiled-kernel cache. **The definition of done under "Goal" is met:**

| program (unmodified) | correctness | run 2 | run 1 (no cache) | our LZ harness (step 3) | Metal route (G2/G3/G4) |
|---|---|---|---|---|---|
| 02 vecadd | **0 / 1,000,000, PASS** | **0.0403 ms/pass** | 15.22 | 0.0063 | 0.0066 |
| 03c matmul (N=2048) | **C[0,0] = 4096.0** | **11.54 ms, 1,489 GFLOP/s** | 285 ms, 60 | 9.26 ms, 1,856 | 9.03 ms, 1,903 |
| 04b training (300 epochs) | **2.1783555 → 0.0005614754**, every printed epoch exact | **0.354 ms/epoch** | 220 | 0.370 | 0.366 |

- The cache turned a per-launch IGC compile into a one-time cost: 378× on 02, 25× on 03c, 622× on 04b.
- **04b is faster end-to-end than our hand-written Level Zero harness** (0.354 vs 0.370 ms/epoch) and than
  the Metal route (0.366). Its 15 launches per epoch dominate, and `DeviceContext` + our runtime handle them
  slightly better than the harness did.
- 02 and 03c retain a per-launch host overhead versus the harness (0.040 vs 0.0063 ms; 11.5 vs 9.3 ms):
  `enqueue_function` builds a `DeviceFunction` per call (encode args, cache lookup, handle setup) where the
  harness bound arguments once. That is Mojo-side work, not kernel time; reducing it is a tuning task.
- Versus the vendor library, the coarse matmul now runs at **~7% of oneMKL** (1,489 of 21,148 GFLOP/s),
  against 9% for the same kernel through the harness. Untuned, and the kernel's geometry is the known limit.

**What this establishes:** `MOJO_CURRICULUM` 02, 03c and 04b — host code and all — compile with our
`spirv64` backend in the open-source Mojo compiler and run correctly on an Aurora PVC tile through our
Level Zero implementation of `DeviceContext`. **Parity level B (source parity) in `AURORA_PLAN.md` is
reached.** Caveats that remain: our fork rather than an official Mojo release; `mojo run` (JIT) proven and
`mojo build` untested; `-I` flags still needed for `max`; one tile; float32; the 18-function runtime slice.

## 🔩 G5 hardening (Mac, 2026-09-16)

**✅ Open item #1 is resolved, and the answer is the opposite of the documented one.**
`Mojo/docs/compiler/WorkingInOSRepo.md` says "You can't use a locally built compiler to build any of the
MAX targets", but with our fork:
- `./bazelw build //max/mojo/max:max` → `max.mojoc` in 40 s, and
  `//max/kernels/src/layout:layout` → `layout.mojoc` in 7 s.
- Compiling the unchanged `02_vecadd_gpu.mojo` against those **compiled packages**
  (`-I bazel-bin/max/mojo/max -I bazel-bin/max/kernels/src/layout`) succeeds with **zero errors**, and
  much faster than re-parsing the `max` sources.
→ the `-I` flags now point at two package files rather than source trees. Dropping them entirely needs the
packages on the compiler's default import path: `MODULAR_MOJO_MAX_IMPORT_PATH` is not honoured by the fork,
and a `MODULAR_HOME`/`modular.cfg` with `import_path` did not take either (the config's section/search-path
layout needs more digging). Cosmetic, not a blocker.

**⏳ Ahead-of-time path:** `gpu/g5_e2e_aot.pbs` builds the packages with the fork on the node, compiles the
unmodified programs with `mojo build` (linking `-lmojo_level_zero_rt` with an rpath), and runs the resulting
executables. Untested until it is submitted.

**Earlier risk list (from before run 1):** (1) Level Zero must snapshot kernel arguments at
append time (04b relaunches one kernel with different arguments); (2) the `deviceApi` out-parameter is
written as a `{pointer, length}` pair, matching Mojo's `StaticString`; (3) `DtoD_async` between a host and a
device buffer is assumed to be a plain copy. Pass = the forward
check has 0 wrong in every stage, and the loss curve matches 2.17835617 → 0.000561475754 at every printed epoch
(G4, job 8828472: 2.1783555 → 0.0005614754, 0.366 ms/epoch).

**Earlier pending note (now done):**
`qsub -v MOJOFILE=g5_matmul_lz.mojo g1_run.pbs`. Pass = check FULL 0 mismatches / 65,536, coarse sampled 0 / 260.
For comparison, the Metal route (job 8827590) gave coarse **1,903 GFLOP/s**. `ocloc -device pvc` passes. The PVC binary is now
**33,208 B**, up from 33,168 B, so IGC compiled the remapped kernel differently (consistent with stores no longer being dropped).

**Step 3 test, original notes:** `gpu/g5_vecadd_lz.mojo` is the G2 host with `g5_vecadd.spv` and the
entry-point name read from file. It compile-checks with Mojo 1.0.0. Risks this run settles:
1. Data pointers loaded from the holders are typed `Function` storage (G2's were addrspace 1).
2. Level Zero / IGC acceptance of a 3,889-char kernel name.


## 🔧 G6 — PVC tuning sweep, prepared on the Mac (2026-09-17)

Opened after Ralph sent the source-parity announcement. The audience is scientists who need good
performance and who, with one or two exceptions, do not write kernels themselves. That splits the
performance question in two, and only one half is about kernels.

### The finding that opened the rung: we have never passed IGC any build flags

`gpu/g5_runtime/mojo_level_zero_rt.cpp` built every kernel module with

```cpp
moduleDesc.pBuildFlags = "";     // through all of G1-G5
```

`zeModuleCreate` hands that string straight to Intel's GPU compiler, so IGC ran on its defaults
throughout: **128 GRF per thread**, where PVC also offers a 256-GRF mode. The G3 warning was
`compiled SIMD32 allocated 128 regs and spilled around 247` — that is IGC telling us the kernel
needed roughly twice the register budget it was given, and we had never asked for the larger one.

This is a **runtime-side** lever, not a kernel-side one: it applies to every Mojo kernel that runs
through our path, including those written by people who never tune anything. The geometry sweep
would never have found it.

**Change:** `pBuildFlags` now comes from `MOJO_LZ_BUILD_FLAGS` (default `""`, so the G5 numbers
remain reproducible), and `MOJO_LZ_BUILD_LOG=1` prints IGC's build log — the only place the
register and spill counts are visible at run time — instead of discarding it on success. Helper
functions compile-checked standalone on the Mac; the full file still builds only on Aurora
(no Level Zero headers here).

### The kernel-side lever: geometry, with a real control

`gpu/g5_matmul_pvc.mojo` is `MOJO_CURRICULUM/03c_matmul_coarse.mojo` with its five geometry
constants rewritable in place by `g6_set_geometry.sh`. `MOJO_CURRICULUM` is untouched.

**A false start worth recording.** The geometry was first parameterized with `-D` defines
(`get_defined_int`). The emitted SPIR-V was instruction-for-instruction identical — `diff` of the
disassembly with quoted strings normalized was empty — but the mangled entry-point name grew from
**4,891 to 9,747 characters**, because a define stays an unfolded expression inside the mangled
type. Kernel-name length is a known risk on this path (the longest we have run through IGC and
Level Zero is 3,889 characters), so a sweep built that way could have failed in all sixteen cells
for a reason unrelated to tuning. Literals instead of defines: the name is back to **4,887**.

**Control check (Mac, our `spirv64` backend).** With the baseline geometry, `g5_matmul_pvc.mojo`
compiles to a kernel whose disassembly is **identical to the curriculum `03c` kernel's**, opcode
for opcode (86,312 vs 86,328 bytes, the 16 bytes being the module and function names). Re-emitting
`03c` with today's fork reproduces the stored `g5_coarse.spv` size exactly, so there is no fork
drift. The `base` cell of the sweep must therefore reproduce G5's 1,489 GFLOP/s; if it does not,
the rest of the table is not to be trusted.

### Pre-flight: all eight geometries compile and validate, before any queue time

`bash gpu/g6_sweep_check.sh` on the Mac — compile with our backend, then `spirv-val`:

| tag | BM | BN | BK | TM | TN | threads/block | acc/thread | SPIR-V bytes | status |
|---|---|---|---|---|---|---|---|---|---|
| `base` | 128 | 128 | 8 | 8 | 8 | 256 | 64 | 86,312 | ok (= curriculum 03c) |
| `t8x8_bk16` | 128 | 128 | 16 | 8 | 8 | 256 | 64 | 121,372 | ok |
| `t8x4` | 128 | 64 | 8 | 8 | 4 | 256 | 32 | 57,544 | ok |
| `t4x8` | 64 | 128 | 8 | 4 | 8 | 256 | 32 | 58,588 | ok |
| `t4x4` | 64 | 64 | 8 | 4 | 4 | 256 | 16 | 37,444 | ok |
| `t4x4_bk16` | 64 | 64 | 16 | 4 | 4 | 256 | 16 | 49,732 | ok |
| `t4x4_big` | 128 | 128 | 8 | 4 | 4 | 1024 | 16 | 36,996 | ok |
| `t2x2` | 32 | 32 | 8 | 2 | 2 | 256 | 4 | 27,912 | ok |

Geometry constraints (`N % BM`, `BM % STRIDE_A`, `BK % STRIDE_B`, the 1024-thread workgroup limit)
are `comptime assert`s, so an invalid combination fails to compile rather than producing wrong
numbers. The kernel also now does a **full** N×N correctness check with an exact-zero count, not
the sampled 260 cells of G3 — the zero count is what distinguishes a missing write from an
arithmetic error (the G4 lesson).

### What is not yet measured

Everything on the GPU. `gpu/g6_sweep.pbs` crosses the eight geometries with the IGC flag sets
(`default` = 128 GRF as in G1–G5, `largegrf` = 256 GRF), 16 cells, and prints IGC's register and
spill report per cell. **Prerequisite: `bash g5_build_runtime.sh` on uan-0007** so the runtime
actually reads `MOJO_LZ_BUILD_FLAGS`; the job refuses to run against an older `.so` rather than
silently producing two identical halves.

Expectations to test, not claims:
- `base` + `default` must reproduce ≈1,489 GFLOP/s, or the sweep is invalid.
- If the spill is the binding constraint, `largegrf` should help `base`/`t8x8_bk16` most (64
  accumulators/thread) and the small-tile geometries least.
- If instead the geometry is wrong for PVC's EU width, the small-tile cells win under both flag
  sets, and the A100/Apple tiling simply does not transfer.
- Those two outcomes imply different advice to users, which is the point of running it.

### Host-side launch overhead — instrumented, with a named suspect (Ralph: cover it, 2026-09-17)

Independent of kernel speed: through `DeviceContext` vecadd costs 0.0403 ms/pass against 0.0063
for our own Level Zero harness, and matmul 11.54 ms against 9.26. Same kernels both ways, so this
is host work, and it is the number that decides performance for programs with many small launches —
most real scientific code, and most of the audience.

**The suspect, found by reading the launch path rather than guessing.** `enqueue_function` builds a
fresh `DeviceFunction` per call, so `AsyncRT_DeviceContext_loadFunction` runs **once per launch**,
not once per kernel. Its first act:

```cpp
std::string cacheKey(functionName);   // ~4,887 bytes of mangled name
cacheKey.push_back('\0');
cacheKey.append(data, dataLength);    // + the ENTIRE SPIR-V module
functionCache.find(cacheKey);         // ...then hash all of it
```

For the coarse matmul that is ~91 KB allocated, copied and hashed **on every launch**. The G5
compiled-kernel cache removed the IGC recompile but left this in the hot path.

**Instrumented, not assumed.** `MOJO_LZ_PROFILE=1` now reports, per call: total, cache-key build
(with bytes copied), cache lookup, module build, and argument encoding. Off by default, behind one
predicted branch when off. The whole runtime file compiles clean under `-Wall -Wextra` on the Mac
against stub Level Zero headers (`gpu/g5_ze_stub/`, `bash gpu/g5_syntax_check.sh`) — new tooling,
because until now a typo in this file cost a full sync/ssh/build round trip.

**The experiment.** Two programs spanning 22× in module size with near-identical launch semantics:

| program | kernel | SPIR-V module | role |
|---|---|---|---|
| `gpu/g6_launch_overhead.mojo` | `bump_kernel` | **3,900 bytes** | per-launch floor |
| `gpu/g5_matmul_pvc.mojo` | `matmul_pvc` (baseline) | **86,312 bytes** | the G5 case |

`g6_launch_overhead.mojo` holds the buffer fixed at 1M float32 and varies only the grid (1 / 16 /
256 / 4096 blocks), 2,000 launches each: host cost per launch is constant in the grid, device time
is not, so the grid=1 row is the host floor. Compile-checked on the Mac — it reaches the linker and
fails only for want of the Aurora runtime `.so`, and the kernel passes `spirv-val`.

**Prediction, recorded before the run:** if the cache key is the cost, `bump_kernel`'s key-build
line should be roughly 22× cheaper than the matmul's, and the fix is a cheap key (module pointer +
length, content-verified) rather than anything architectural. If the two are equal, the cost is
elsewhere and the profile says where. `g6_sweep.pbs` also runs with `MOJO_LZ_PROFILE=1`, so its
eight geometries (27,912 → 121,372 bytes) give the same relationship across 4.3× for free.

Neither has been run. `qsub g6_launch.pbs`.

## ✅ G6 — PVC tuning sweep RESULTS: 1,452 → 7,737 GFLOP/s (job 8836109, 2026-09-17)

All 16 cells ran in ~11 minutes (walltime 01:30 requested, far more than needed). **Every cell is
numerically exact: 0 / 4,194,304 wrong, 0 exact zeros** — the full N×N check, not G3's sampled 260.

Node yardstick unchanged: oneMKL SGEMM on this tile = **21,148 GFLOP/s**.

### The table

GFLOP/s at N=2048, one PVC tile, one run per cell (no repeat statistics yet).

| tag | BM×BN, BK | TM×TN | acc/thread | default (128 GRF) | `-ze-opt-large-register-file` | flag effect |
|---|---|---|---|---|---|---|
| `base` | 128×128, 8 | 8×8 | 64 | **1,452** | **4,412** | **×3.04** |
| `t8x8_bk16` | 128×128, 16 | 8×8 | 64 | 1,086 | 4,579 | ×4.22 |
| `t8x4` | 128×64, 8 | 8×4 | 32 | 4,278 | 6,327 | ×1.48 |
| `t4x8` | 64×128, 8 | 4×8 | 32 | 4,221 | 7,273 | ×1.72 |
| `t4x4` | 64×64, 8 | 4×4 | 16 | 6,740 | 7,421 | ×1.10 |
| `t4x4_bk16` | 64×64, 16 | 4×4 | 16 | 6,082 | **7,737** | ×1.27 |
| `t4x4_big` | 128×128, 8 (1024 thr) | 4×4 | 16 | 5,903 | 5,165 | **×0.87** |
| `t2x2` | 32×32, 8 | 2×2 | 4 | 6,415 | 3,834 | **×0.60** |

**The control holds.** `base`/`default` is the geometry that compiles to a kernel identical to
curriculum `03c`: **1,452 GFLOP/s against G5's 1,489**, −2.5% on single runs. The table is trustworthy.

### Headline

**Best cell: `t4x4_bk16` + large GRF = 7,737 GFLOP/s = 36.6% of oneMKL**, against the 1,489 (7.0%)
we shipped and announced. **A factor of 5.2 on the same hardware, the same compiler backend and the
same runtime** — the kernel source differs only in five geometry constants, and the flag is one
string passed to `zeModuleCreate`.

For scale: the same coarse kernel on an A100 reached 8,966 GFLOP/s. PVC now reaches **86% of the
A100's absolute number** (it was 17%), though still 36.6% of its own vendor library against the
A100's 64.8% of cuBLAS.

### Two levers, both real, and they do not multiply

- **Geometry alone** (default flags): 1,452 → 6,740, **×4.64**.
- **Flag alone** (`base` geometry): 1,452 → 4,412, **×3.04**.
- **Both:** 7,737, **×5.33** — not ×14. They are two ways of relieving the *same* constraint
  (register pressure), so they overlap almost completely. Either one alone captures most of the win.

### The spill report explains the whole default column

IGC emits `compiled SIMD32 allocated 128 regs and spilled around N` only when it spills. Under
default flags it fired on exactly the register-hungry geometries, and performance tracks it inversely:

| geometry | acc/thread | spilled | GFLOP/s |
|---|---|---|---|
| `t8x8_bk16` | 64 | 254 | 1,086 |
| `base` | 64 | 248 | 1,452 |
| `t8x4` | 32 | 106 | 4,278 |
| `t4x8` | 32 | 96 | 4,221 |
| 16- and 4-accumulator geometries | 16 / 4 | *(no warning)* | 5,903 – 6,740 |

**Under `-ze-opt-large-register-file` the warning disappears from every cell.** The flag does what
it was reached for. The G3-era reading — "the Apple/NVIDIA tiling goes *backwards* on PVC" — was a
correct observation of a spill, not a property of the hardware.

### ⚠️ The flag is a trade, not free — this corrects a prediction

The prediction recorded before the run was that large GRF "helps every Mojo kernel on our path,
including users who never tune". **That is wrong, and the sweep says so plainly:**

- `t2x2` (4 accumulators): 6,415 → **3,834**, a 40% *loss*.
- `t4x4_big` (16 accumulators, 1024 threads/block): 5,903 → **5,165**, a 12% loss.

256-GRF mode halves the hardware threads resident per EU. A kernel that was never register-bound
pays that in occupancy and gets nothing back. **So the flag must not be switched on by default.**
The honest rule is narrower and more useful: *turn it on when IGC reports a spill, leave it off
otherwise* — and we can now read that report at run time (`MOJO_LZ_BUILD_LOG=1`).

### What this overturns

`AURORA_RESULTS.md` (G3) and our earlier note to colleagues both say the optimized kernels run backwards on
PVC — naive 2,449 > tiled 2,177 > coarse 1,903 — and that the untuned example reaches ~7% of oneMKL.
The first half is now explained (register spill, fixable) and the second is superseded: the coarse
kernel with PVC-appropriate geometry reaches **7,737**, which is **3.2× the naive kernel**, not below
it. Tiling does transfer to PVC; it just cannot keep Apple's constants.

### Launch-overhead evidence from the same job (free, via `MOJO_LZ_PROFILE=1`)

Steady-state host cost per launch, from the eight module sizes the sweep happens to span:

| module bytes | cache-key build | cache lookup | enqueue total | **host µs/launch** |
|---|---|---|---|---|
| 32,772 (`t2x2`) | 1.43 | 0.74 | 4.63 | **6.80** |
| 42,304 (`t4x4`) | 1.63 | 0.99 | 4.91 | 7.53 |
| 62,404 (`t8x4`) | 2.16 | 1.22 | 5.26 | 8.64 |
| 91,172 (`base`) | 2.83 | 2.01 | 4.89 | 9.73 |
| 126,232 (`t8x8_bk16`) | 3.74 | 2.72 | 4.98 | **11.44** |

**Mechanism confirmed, magnitude refuted.** Cache-key build does scale linearly with module size
(~29 ns/KB), exactly as predicted — but it is 1.4–3.7 µs, not the dominant term. Total measured host
cost is 6.8–11.4 µs/launch, and argument encoding is a flat 0.42 µs. So fixing the cache key is worth
at most ~5 µs/launch on the largest module: real, cheap, and **not** the explanation for the
0.0403 vs 0.0063 ms/pass vecadd gap (34 µs). The remainder is elsewhere — Mojo-side `DeviceFunction`
construction above our runtime, which our counters cannot see. `g6_launch.pbs` (job 8836110) measures
the floor directly.

## ✅ G6 — launch overhead RESULTS: the per-launch barrier is the cost (job 8836110, 2026-09-17)

`g6_launch_overhead.mojo`: one trivial kernel (`buf[tid] += 1`, 3,900-byte module), buffer fixed at
1M float32, only the grid varied, 2,000 launches per row. `buf[0]` came back **8004** — exactly
4 rows × (2,000 + 1 warmup), so every launch landed and none were lost.

| grid | threads | **µs/launch** | measured host cost |
|---|---|---|---|
| 1 block | 256 | **3.91** | ~3.7 |
| 16 blocks | 4,096 | 12.28 | ~3.7 |
| 256 blocks | 65,536 | 38.19 | ~3.7 |
| 4,096 blocks | 1,048,576 | 38.37 | ~3.7 |

**Host cost is flat and small, and it is not the problem.** The runtime's own counters, over 8,004
calls: cache-key build 0.27 µs (5,235 bytes), cache lookup 0.09 µs, `enqueueFunction` 3.36 µs,
argument encoding 0.24 µs → **3.72 µs/launch**, which matches the 1-workgroup row (3.91) almost
exactly. The grid does not change any of it.

**So the rise from 3.9 to 38.4 µs is device-side, and it is not the kernel.** 256 and 4,096
workgroups cost the same 38 µs despite 16× the work and 16× the memory traffic — that rules out both
compute and bandwidth. What is left is `zeCommandListAppendBarrier`, which we append after **every**
launch to supply the in-order stream `DeviceContext` promises (the G4 fix). It drains the device each
time, and the drain cost grows with the number of workgroups in flight until it saturates.

This is also most of the gap we have been quoting since G5: 40.3 µs/pass through `DeviceContext`
against 6.3 through our own harness — **the harness launched independent kernels with no barrier at
all**, so it was never paying for ordering. Comparing them as though they did the same work was not
apples to apples, and the G5 write-up should be read with that in mind.

### The cache-key prediction: mechanism right, magnitude wrong

Predicted before the run: "~22× apart on the key-build line → the fix is a cheap cache key". Measured
across a 17.4× span in module size, key-build went 0.27 µs (5,235 B) → 2.70 µs (91,172 B), a 10×
rise — linear in size at ~29 ns/KB, so **the mechanism is confirmed**. But it is 7% of per-launch
host cost for a small kernel and 27% for a large one, of a host cost that is itself only ~10% of the
38 µs total. **Fixing the cache key is worth ~2.4 µs/launch on the matmul and is not the main event.**
Worth doing, cheap, but the barrier is where the time is.

Control in this job: the baseline-geometry matmul gave **1,488.63 GFLOP/s** against G5's 1,489 —
tighter than the sweep's 1,452, so that 2.5% was ordinary run-to-run variance.

### Next: replace the barrier with an in-order command list (written, not yet run)

`zeCommandListCreateImmediate` was being given `queueDesc.flags = 0`. Level Zero can provide in-order
semantics in the driver via `ZE_COMMAND_QUEUE_FLAG_IN_ORDER`, without a full drain per launch.
`MOJO_LZ_ORDERING` now selects `barrier` (default, what G5 shipped), `inorder`, or `none`
(diagnostic, no ordering — expected to be wrong).

**A bug caught while writing it:** the first version guarded the flag with `#ifdef`. That
enumerator is an *enum member*, not a macro, so the test is always false and `inorder` would have
silently reported "your header has no in-order support" on a machine whose header has it. Used
unconditionally now; a too-old header fails the build loudly instead.

`gpu/g6_ordering.pbs` runs all three modes against **both** a speed test and a correctness test.
The correctness test is `04b` — 15 dependent launches per epoch, the program the G4 ordering bug was
found in. Under `none` its loss curve must go **wrong** (that is the G4 bug deliberately
reintroduced, proving ordering is genuinely under test); under `barrier` and `inorder` it must
reproduce 2.1783555 → 0.0005614754 at every printed epoch. Fast *and* exact under `inorder` is the
result worth having; fast but wrong means the flag does not mean what we think it does.

## 🔬 G6 — ordering experiment: the launch gap is SEMANTICS, not overhead (job 8836165, 2026-09-17)

Three ordering modes, each against a speed test and a correctness test. **This refutes the actionable
half of the barrier hypothesis recorded above, and the correction matters more than the original
guess did.**

| mode | 1 blk | 16 blk | 256 blk | 4096 blk | host `enqueue` | 04b (15 dependent launches/epoch) |
|---|---|---|---|---|---|---|
| `barrier` (G5 shipped) | 3.92 | 12.33 | **38.32** | **38.41** | 3.41 µs | ✅ exact, 0.3557 ms/epoch |
| `inorder` (driver) | 2.90 | 12.14 | **37.74** | **37.91** | 2.31 µs | ✅ exact, 0.3670 ms/epoch |
| `none` (diagnostic) | 2.79 | 2.79 | **2.76** | **10.69** | 2.24 µs | 💥 **GPU segfault** |

### What was predicted, and what actually happened

Predicted: the per-launch `zeCommandListAppendBarrier` is the cost, and Level Zero's in-order command
list would remove it. **Half right.**

- **`none` collapses to ~2.8 µs at every grid up to 4,096 workgroups** — a 14× drop. So the 38 µs is
  unambiguously the cost of *ordering*, exactly as diagnosed.
- **But `inorder` is no cheaper than `barrier` on the device**: 37.74 vs 38.32 µs, a 1.5% difference.
  `ZE_COMMAND_QUEUE_FLAG_IN_ORDER` does not buy a cheaper serialization — it buys the *same*
  serialization, expressed through the driver instead of an explicit command.

**Therefore the 38 µs is not barrier overhead. It is the kernel's own completion latency, exposed
because in-order execution forbids the next launch from overlapping it.** Under `none` the launches
pipeline and the per-launch cost falls to the host floor.

### The consequence for what we claim

The gap we have quoted since G5 — 40.3 µs/pass through `DeviceContext` against 6.3 through our own
Level Zero harness — **is the price of the in-order stream `DeviceContext` guarantees, not a defect
in our runtime.** Our harness launched *independent* kernels with no ordering at all, so it was
measuring a different thing; the comparison was never apples to apples. Any correct implementation of
`DeviceContext` pays this, on any vendor. It is not an Intel problem and not a fork problem.

What `inorder` *does* buy is host-side: `enqueueFunction` drops 3.41 → 2.31 µs/launch (**−32%**) by
not appending a barrier command every time. Real, small, and free.

**And the realistic program is unaffected either way**: 04b runs 0.3557 (`barrier`) vs 0.3670
(`inorder`) ms/epoch, both inside the 0.354–0.370 band we have seen across runs all week. Its kernels
are small (N=256, H=16), so per-kernel latency — and therefore the cost of ordering — is small. **The
38 µs penalty only bites programs that launch large kernels they could have overlapped.**

### The `none` run vindicates the G4 fix more strongly than expected

Predicted: 04b's loss curve goes wrong. Actual: `buf[0]` came back **7,725 instead of 8,004** in the
launch benchmark — 279 read-modify-write updates lost to genuine overlap, direct proof the launches
really were concurrent — and **04b did not merely compute wrong numbers, it took a GPU page fault**
(`Segmentation fault from GPU at 0x0 ... level: 3 (PML4), access: 0 (Read)`) right after epoch 1.

So AURORA PATCH 6 was not only a numerical-correctness fix. Without ordering this program *crashes
the device*. Worth stating plainly in the assessment document.

### Where this leaves launch overhead

Host cost is ~3–4 µs/launch and mostly irreducible: `enqueue` 2.31 (with `inorder`), cache-key build
0.29–2.70 depending on module size, lookup ~0.1–2.0. The cache-key fix is worth ~2.4 µs on the
largest module and remains cheap to do, but **host overhead was never the big number.** The big number
is ordering, and ordering is a semantic guarantee, not waste.

**The real gap to name is in Mojo/MAX, not in our runtime:** `DeviceContext` offers no way for a
program to say "these launches are independent, overlap them". CUDA streams and Level Zero queues both
express that; Mojo's API currently does not. A program with large independent kernels cannot get that
performance back on *any* backend. That belongs in the assessment document as a limitation of the API,
observed from the inside.

**Default unchanged for now.** `inorder` is semantically cleaner and cheaper on the host, but 04b
measured 3% slower and we have one run of each — not enough to move the default. Repeat both a few
times before deciding.

### Cosmetic, for the record

`grep: warning: stray \ before =`, twice per compile, comes from Modular's own `tools/bazel` wrapper
(line 34: `grep -E 'config\=prebuilt-mojo|...'`), not from our scripts. Harmless.

## 🧪 G6 — stencil benchmark added (built 2026-09-17, not yet run)

**Why, in Ralph's framing.** His colleagues are not asking us to support them. They use machines from
many vendors, they expect Modular to do the supporting, and what they want from us is *evidence that
Mojo will follow them onto new hardware* — with the corollary that if two people can add a vendor to
the open compiler in a few days, they can credibly ask Modular to support whatever interesting chip
their science needs next.

**Performance is part of that claim, not supporting evidence for it.** Ralph corrected an earlier
framing of this section: *"if it does not run well that is almost equivalent to not running."*
Compiling and producing exact results only demonstrates that a toy can be bootstrapped; what is
actually in question is whether a community-built backend reaches **usable** speed. The tuning work is
therefore load-bearing. What to avoid is the *phrasing* "36.6% of oneMKL" — dismissible by the very
reader worth convincing, since nobody writes their own SGEMM — not the measurement behind it.

**Why a stencil.** Everything measured so far is vector add, matmul or a tiny MLP: trivial or
GEMM-shaped. Quoting "36.6% of oneMKL" invites the reply that nobody writes their own SGEMM — they
call the library, from Mojo too — so being 3× off it says nothing about the code they write.
`gpu/g6_stencil.mojo` is a 5-point Jacobi stencil: memory-bound, halo-shaped, **no vendor library
exists to call**, and it is the shape of a large share of real HPC kernels.

**It carries its own yardstick.** The same run measures a pure streaming copy on the same tile — the
achievable-bandwidth ceiling for a kernel touching each element once — and reports the stencil as a
percentage of it, using identical 8-bytes-per-point accounting. No outside number is needed and no
vendor claim is being leaned on.

**Correctness is exact, not a tolerance.** The field is `f(i,j) = 1 + i/2 + j/4`, which is linear and
therefore an exact fixed point of the 5-point average. Every value is a multiple of 0.25 below 2²³, so
float32 represents all intermediates exactly and the grid after any number of sweeps must equal the
initial grid bit for bit. A *constant* field would also be a fixed point but would hide indexing bugs;
a linear one does not. Both buffers are checked: 0 wrong out of 8,388,608.

**Verified on the Mac (no Aurora needed):** compiles for `intel-pvc` with zero errors and zero
warnings, failing only at link for want of the runtime `.so`. Both kernels emit and pass `spirv-val`
— `stencil_kernel` 8,636 bytes, `copy_kernel` 8,008 bytes.

**It also tests the open question from the sweep.** `g6_stencil.pbs` runs it 3× under each IGC flag
setting. A stencil holds almost nothing in registers, so this is precisely the case where
`-ze-opt-large-register-file` should *hurt*. If it does, the advice to colleagues becomes "enable it
when IGC reports a spill", not "enable it".

**The portability point is the real deliverable.** `g6_stencil.mojo` is plain Mojo with no Intel in
it: the same file runs on Polaris and on an Apple GPU with **stock, unmodified Mojo**. Three rows from
one unmodified source is the argument; the Aurora row alone is not.

*Note: our fork is built with SPIR-V but not Metal (`target 'air64-apple-macosx' is not supported by
this build`), so the Mac row needs a stock Mojo install rather than the fork.*

## ✅ G6 — confirmation runs: error bars, the flag question settled, ordering settled (jobs 8836259 / 8836260, 2026-09-17)

### A. Error bars — the measurements are stable

Matmul GFLOP/s, 3 runs per cell, all exact (0 / 4,194,304):

| cell | mean | spread | |
|---|---|---|---|
| `base` / default (control) | **1,480.3** | 0.41 | **0.03%** |
| `base` / largegrf | 4,417.4 | 7.28 | 0.16% |
| `t4x4` / default | 6,754.8 | 5.89 | 0.09% |
| `t4x4_bk16` / largegrf | **7,782.0** | 27.57 | **0.35%** |

**Headline with its error bar: 7,782 ± 28 GFLOP/s, a 5.26× speedup** over the shipped configuration.
36.8% of oneMKL; **86.8% of the identical kernel on an A100**. Geometry alone ×4.56, flag alone ×2.98.

⚠️ **The honest error bar is node-to-node, not run-to-run.** Within one job the control repeats to
0.03%. Across the three jobs that measured it, it read 1,452 / 1,488.6 / 1,480.3 — a spread of ~2.5%.
So quote **±2.5%**, not ±0.03%; the within-job precision is real but flatters us.

### B. ⭐ The flag question is settled, and the answer is "not by default"

The 256-GRF flag against the three real programs, and against the stencil:

| program | default | largegrf | effect |
|---|---|---|---|
| 03c matmul | 1,480.2 GFLOP/s | 4,408.7 | **×2.98 — large win** |
| 02 vecadd | 0.04005 ms/pass | 0.03885 | +3.1% faster |
| 04b training | 0.34436 ms/epoch | 0.35531 | **−3.2% slower** |
| **stencil (streaming copy ceiling)** | **708.4 GB/s** | **459.0** | **−35.2%** |
| **stencil** | **630.8 GB/s** | **420.1** | **−33.4%** |

**A memory-bound kernel loses a third of its bandwidth under large-GRF.** The mechanism is the one
predicted: 256-GRF mode halves the hardware threads resident per execution unit, and memory-bound code
needs those threads in flight to keep the memory system busy. Register-hungry compute code wins big;
everything else pays.

**So the advice to users is: enable it when Intel's compiler reports a spill, never blanket-on.** That
report is now readable at run time (`MOJO_LZ_BUILD_LOG=1`). This is the recommendation the sweep could
not have produced on its own — it took a memory-bound kernel to expose the cost.

### C. ⭐ Stencil: 89% of achievable streaming bandwidth

5-point Jacobi, 2048², 100 sweeps, **0 wrong of 8,388,608 every run**:

| | copy (ceiling) | stencil | **stencil / copy** |
|---|---|---|---|
| default | 708.4 GB/s (±8.2%) | 630.8 GB/s (±7.3%) | **89.06% (±0.85%)** |
| largegrf | 459.0 | 420.1 | 91.53% |

**A hand-written stencil reaches 89% of what the same tile achieves on a pure streaming copy.** For
the claim the assessment document actually makes — *when you must write a kernel yourself, do you get
good use of the hardware* — this is the evidence, and it is a good answer.

**The self-normalising design paid off.** Absolute bandwidth wandered 8% across repeats (745 / 687 /
693 GB/s on the copy), but the *ratio* held to ±0.85% because the stencil tracked it. Measuring the
yardstick in the same run was the right call.

⚠️ **State the ceiling honestly.** Our "achievable" is a simple scalar copy kernel at ~708 GB/s, which
is itself well short of the tile's HBM peak — both kernels use scalar float32 loads with no
vectorisation. So **89% means 89% of a simple copy, not 89% of the hardware.** That is the right
comparison for "code a user writes", since both sides are ordinary unvectorised code, but it must not
be read as 89% of peak. Follow-up worth doing: a vectorised copy to find the real ceiling.

### D. Ordering default: keep `barrier`

04b, ms/epoch, 3 runs each — `barrier` 0.3531, `inorder` 0.3722. **`inorder` is 5.4% slower and every
one of its runs is slower than every `barrier` run**, so the difference is outside the spread. The
in-order command list is cheaper on the host (enqueue 3.41 → 2.31 µs) but that does not survive to
end-to-end time on a real program. **Default stays `barrier`** — which is also the path proven since
G5. `MOJO_LZ_ORDERING` remains available for experiments.

## ⚠️ G6 — ahead-of-time `mojo build`: compiles, does not yet launch (job 8836261, 2026-09-17)

The last untested half of the pipeline, owed since 2026-09-16. **The compiler side passes; the
packaging side does not.**

**What worked.** The fork built the `max` and `layout` packages on the Aurora node in 97 s
(`max.mojoc` 1,121,629 B, `layout.mojoc` 1,194,384 B), and `mojo build --target-accelerator intel-pvc`
produced a standalone executable for **all three unmodified programs** — 02 vecadd (141,144 B),
03c matmul (230,624 B), 04b training (375,616 B), every build exit 0. **So our backend compiles
ahead-of-time exactly as it compiles just-in-time**; nothing here is a backend defect.

**What failed.** All three executables die at launch:

```
error while loading shared libraries: libKGENCompilerRTShared.so: cannot open shared object file
```

That is **Mojo's own compiler-runtime library**, not ours. `mojo run` finds it through Bazel's
runfiles tree; a standalone binary has no such environment. It lives at
`$REPO/bazel-bin/Mojo/libKGENCompilerRTShared.so`. The job put our Level Zero runtime on
`LD_LIBRARY_PATH` and gave it an rpath, but never did the same for Mojo's.

**Fix applied to `gpu/g5_e2e_aot.pbs`** (not yet rerun): add `$REPO/bazel-bin/Mojo` to
`LD_LIBRARY_PATH` *and* pass a second `-Xlinker -rpath` for it, so the executables are self-contained
rather than dependent on the job's environment; print `ldd` before running so any remaining unresolved
library names itself; check the library exists up front and search the fork if not; and add
`$IGNORE_DEPRECATED`, which this job predated.

**What to claim meanwhile:** ahead-of-time compilation works; ahead-of-time *deployment* needs the
Mojo runtime library shipped alongside, which is ordinary packaging and not yet demonstrated end to
end. Rerun `qsub g5_e2e_aot.pbs`.

## ✅ G6 — ahead-of-time `mojo build` PASSES end to end (job 8836303, 2026-09-17)

The rpath fix worked. **All three unmodified programs now build to standalone executables with our
backend and run correctly on a PVC tile — the last untested half of the pipeline, owed since
2026-09-16, is closed.**

```
libKGENCompilerRTShared.so => MOJO_WORK/modular/bazel-bin/Mojo/libKGENCompilerRTShared.so
libmojo_level_zero_rt.so   => MOJO_WORK/runtime/lib/libmojo_level_zero_rt.so
```

| program (unmodified, `mojo build`) | result | speed |
|---|---|---|
| 02 vecadd | **0 / 1,000,000, PASS** | 0.0392 ms/pass |
| 03c matmul (N=2048) | **C[0,0] = 4096.0** | 11.75 ms, 1,462.3 GFLOP/s |
| 04b training (300 epochs) | **2.1783555 → 0.0005614754**, every printed epoch exact | 0.358 ms/epoch |

Matmul at **1,462.3** against the JIT path's 1,480.3 is −1.2%, inside the ±2.5% node-to-node band
established in job 8836259 — so **ahead-of-time and just-in-time produce the same performance**, as
they should: the same backend emits the same kernel either way.

**Both claims about the pipeline are now proven:** `mojo run` (JIT) *and* `mojo build` (AOT).

### ⚠️ A silent-failure trap caught in this job's output, now fixed

The run printed, and ignored:

```
cp: cannot create regular file '.../aot/pkgs/max.mojoc': Permission denied
```

Bazel writes its outputs **read-only** (`r-xr-xr-x`), so copying over an existing copy fails. The
build then succeeded anyway — **because it used the `max`/`layout` packages left behind by the
previous job.** This run happened to be harmless (the packages were unchanged), but the failure mode
is nasty: after any change to the fork, this job would silently compile against stale packages and
report success. `g5_e2e_aot.pbs` now removes the old copies first, makes the new ones writable,
**and exits with an error rather than building against packages of unknown vintage.**

## 🖥️ G6 — the two-machine portability table (Mac, 2026-09-17)

Ralph's call: skip Polaris for now (its Mojo predates the `std.gpu.*` → `max.gpu.*` module move, so
the current sources would not compile there without an upgrade), reference the earlier Polaris report,
and present **Mac → Aurora** — which is also the truer story, since all of this was developed on the
laptop and ported to the supercomputer.

**The versions line up exactly.** `MOJO_CURRICULUM/.venv` runs **Mojo 1.0.0 (ed45d567)** with
`max==26.5.0` — the *same build* as Aurora's venv. So "the same file on both machines" is literal.
(Note `/Users/rbutler/VENVS/BASE/bin/mojo` is 1.0.0b2 and has no `max` package; use the curriculum
venv.)

Same unmodified sources, stock Mojo on the Mac, our fork on Aurora:

| program | Mac — M4 Max, stock Mojo | Aurora — PVC tile, our backend |
|---|---|---|
| 02 vecadd (1M) | 0 / 1,000,000 · **0.1023 ms/pass** | 0 / 1,000,000 · **0.0392 ms/pass** (2.6× faster) |
| 03c matmul, N=2048, *Apple-tuned geometry* | `C[0,0]` exact · **4,356.4 GFLOP/s** | `C[0,0]` exact · **1,480.3** (2.9× slower) |
| 03c matmul, *retuned for PVC* | — | **7,782** (1.8× faster than the Mac) |
| 04b training, 300 epochs | curve agrees to 7 s.f. · **0.2727 ms/epoch** | · **0.344 ms/epoch** |

**This is the tuning argument made bidirectional, which is much stronger than making it one way.** The
curriculum's matmul geometry was tuned on the Mac; it gets 4,356 GFLOP/s there and only 1,480 on
Aurora. Retuned for Aurora it reaches 7,782 — 1.8× the Mac. Neither machine is "the fast one": the
kernel constants belong to the hardware, and moving them is ordinary work in both directions.

The Mac wins the tiny training loop (0.273 vs 0.344 ms/epoch) because at N=256, H=16 the epoch is 15
kernel launches and the Mac's unified memory has less launch latency — not a throughput result.

### ⚠️ Correction: the cross-machine loss curves agree to 7 s.f., not 8, and are not identical

Measured directly rather than assumed:

| epoch | Aurora | Mac |
|---|---|---|
| 1 | 2.1783555 | 2.1783555 *(identical)* |
| 50 | 0.016860535 | 0.016860532 |
| 100 | 0.0056486796 | 0.005648679 |
| 150 | 0.0029576812 | 0.0029576812 *(identical)* |
| 200 | 0.0016037121 | 0.0016037134 |
| 250 | 0.0008951729 | 0.0008951736 |
| 300 | 0.0005614754 | 0.00056147523 |

**They agree to about 7 significant figures and differ in the 7th–8th digit** — textbook float32
non-associativity, different accumulation order on different hardware. Two epochs happen to match
exactly. The claim "the same numbers as on NVIDIA and Apple" is therefore **too strong** and has been
corrected in `ASSESSMENT.md` to "agrees to seven significant figures". The *within-machine* claim is
untouched and remains exact: on Aurora, our backend reproduces our harness's curve digit for digit.

### ❌ RETRACTED — there is no Modular Apple-backend bug; the Mac is missing an Xcode component

**An earlier draft of this section claimed we had found a bug in Modular's Metal backend, because
`g6_stencil.mojo` failed there with `Metal Compiler failed to compile metallib` while compiling and
running through our Intel backend. That claim was wrong and is withdrawn.**

The isolation was chasing the wrong variable. The decisive test was not a code mutation at all:
a **byte-identical copy of `03a_matmul_naive.mojo`, under a different filename in the same
directory, also fails** — while the original works. So it never depended on the kernel's contents.
Mojo caches the compiled metallib per source file; the curriculum programs only run because their
metallibs were cached by earlier runs. Anything new must compile fresh, and fresh Metal compilation
is broken on this machine:

```
$ xcrun -sdk macosx metal --version
error: cannot execute tool 'metal' due to missing Metal Toolchain;
       use: xcodebuild -downloadComponent MetalToolchain
```

**Root cause: macOS 26.6.2 / Xcode 26 ships the Metal Toolchain as a separately downloaded component,
and it is not installed.** Nothing to do with Mojo, Modular, or our backend. Fix:
`xcodebuild -downloadComponent MetalToolchain`, then re-run.

**Lesson worth keeping:** four consecutive code-mutation experiments all "confirmed" a hypothesis
about kernel structure, and all four were misreading a cache hit as a property of the code. The test
that broke it open was changing *nothing* — copying a working file to a new name. When several
mutations all fail and the original works, suspect the harness, not the code.

**Consequences for what is recorded above:**
- The Mac numbers for 02 / 03c / 04b are real GPU measurements from validly cached kernels, and the
  portability table stands. Worth re-running once the toolchain is installed, as confirmation.
- **The stencil should be expected to run on the Mac after the install** — its absence from the table
  is an environment gap on one laptop, not a portability limit. Re-run it then.

⚠️ **`gpu/g6_stencil.mojo` was restored byte-for-byte** to the version that produced job 8836260's
numbers, after the `thread_idx`/`block_idx` experiment above. Do not let debugging edits drift into a
file whose measurements are already published.

## ⚠️ G6 — the stencil's "% of achievable bandwidth" metric does not hold up (Mac, 2026-09-17)

After `xcodebuild -downloadComponent MetalToolchain`, `g6_stencil.mojo` compiles and runs on the Mac
— confirming the retraction above. **But the Mac result invalidates the metric's design.**

| | copy ("ceiling") | stencil | ratio |
|---|---|---|---|
| Aurora PVC | 708 GB/s | 631 GB/s | 89% *(sensible)* |
| **Mac M4 Max** | **343–401 GB/s** | **579–759 GB/s** | **167–209%** ❌ |

**A ratio above 100% is prima facie proof the yardstick is wrong.** A kernel that reads four
neighbours and writes one cannot exceed a kernel that reads one and writes one, if the latter really
measures the bandwidth ceiling.

**Diagnosis.** Re-timing the copy *after* the stencil raises it from 401 to 499 GB/s (+24%), so part
of it is a first-timing/residency artifact that the single warmup launch does not absorb on Apple's
unified memory. **But that does not close the gap** — the stencil still runs at 675 GB/s against the
re-timed copy's 499. On this hardware the naive scalar copy is simply a worse kernel than the stencil.

**Consequence — what we may and may not say:**
- ❌ **Drop "89% of achievable bandwidth".** The copy is not a ceiling; it is just another
  unvectorised kernel, and on at least one GPU it is the *slower* of the two. Where the ratio is
  under 100% it is an **upper bound** on efficiency, not a measurement of it: if the true ceiling is
  higher than our copy reached, the real efficiency is lower than the ratio suggests.
- ✅ **Absolute bandwidth is fine, and is a good portability result on its own:** the same stencil
  reaches **631 GB/s on an Aurora tile and a median 666 GB/s on an M4 Max** — comparable, from one
  unmodified source, on two very different memory systems. **Nine Mac runs span 436–759 GB/s**
  (median 666, two low outliers), against Aurora's ±7% over three. A laptop sharing its GPU with a
  desktop is not a measurement instrument; quote the median and say so.
- ✅ Correctness is unaffected: 0 wrong of 8,388,608 on both machines.

**Honest note on how this was designed.** Measuring the yardstick in the same run was the right
instinct — it caught its own flaw, which a spec-sheet number never would have. But it was recorded
earlier tonight as "the self-normalising design paid off" on the strength of Aurora alone, where the
ratio looked sensible. One platform agreeing with a hypothesis is not evidence that the metric is
sound. **A Mac-side vectorised copy is the follow-up worth doing**; until there is a yardstick that is
actually a ceiling on both machines, report absolute GB/s.
