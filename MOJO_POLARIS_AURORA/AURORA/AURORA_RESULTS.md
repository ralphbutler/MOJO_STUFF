# 🌌 Aurora Mojo Port — Results & Experience Log

Running, chronological log of the Aurora (Argonne, Intel CPU) Mojo bring-up, kept so the
full experience is reportable later. Companion to `AURORA_PLAN.md` (checklist) and
`AURORA_PORT.md` (rationale). Polaris analog: `../POLARIS/REPORT_RESULTS.md`.

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
