# 🛠️ REPORT — Reproducing Mojo on ALCF Aurora (from zero)

**Purpose:** an ordered, copy-pasteable procedure to stand up the Mojo toolchain on Aurora
(Intel CPU target) and run CPU workloads, capturing every gotcha so the next person doesn't
rediscover them. Aurora's defining twist vs the Polaris sibling: **Mojo will not install on the
bare host** — its wheels need a newer glibc than Aurora ships — so everything runs inside an
**apptainer container**.
**Status:** verified end-to-end 2026-07-08 — toolchain (in container), L0 smoke, and the CPU
ladder L2/L3/L4 + a NUMA experiment all working. GPU is out of scope (no Intel-GPU backend).
**Companion docs:** `REPORT_RESULTS.md` (numbers), `AURORA_RESULTS.md` (detailed running log),
`AURORA_PLAN.md` (ladder), `FROM_POLARIS.md` (carryover from the A100 port).

---

## 0. 📋 Prerequisites

- An ALCF Aurora account with an **active allocation** to charge jobs to (ours: `ModCon`) and
  membership in its unix group. A storage directory alone is not enough.
- Layout we used: a working dir on live project storage holding the sources, the uv project
  (`pyproject.toml`), the container image, and the `.venv`, all in one place:
  `/lus/flare/projects/ModCon/$USER/MOJO_WORK/AURORA` (referred to below as `$PROJROOT`).

> ⚠️ **Machine profile (verify on first login):** 2× Intel **Xeon CPU Max 9470 (Sapphire
> Rapids)** per node — **102 physical / 204 logical cores** usable (of 104/208; ~2 physical
> reserved by core-specialization), **AVX-512**, on-package **HBM** (no DDR). OS **SLES 15-SP4,
> glibc 2.31**, kernel 5.14. **PBS Pro 2022.1.7.** The 6× Intel Data Center GPU Max per node are
> **unusable from Mojo** (no Intel-Xe/Level-Zero backend) — this is a **CPU-only** experiment.
> The glibc version drives the single most important divergence from Polaris (Step 4).

---

## 1. 🔑 Login environment

Unlike Polaris (conda module), Aurora needs no module stack for Mojo — just install `uv`
standalone. The login node has **direct outbound internet (no proxy)**; system `python3` is an
unusable 3.6, but `uv` fetches its own.

```bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

`which mojo` returns nothing — that is expected throughout; `mojo` lives inside the venv and is
always invoked through the container (Steps 5–6). The `aurora_setup.sh` script automates this
login prep (uv install + storage write-test + proxy autodetect).

---

## 2. 💾 Storage — avoid the dead-allocation trap

Home is tiny and, critically, **a retired project's Lustre space can be quota-zeroed** — you can
read it but cannot write a byte. On Aurora the expired `candle_aesp_CNDA` allocation showed
`Used 60.67T* / Quota 1M` (effectively zero) on `/lus/flare`; the live `ModCon` allocation shows
`54.1T / 1000T`.

**Do this:**
- Put `$PROJROOT` + the `.venv` + the uv cache + the container image on the **live** project FS
  (`/lus/flare/projects/ModCon/...`), never HOME and never the dead allocation.
- The default `~/.cache` symlink pointed into the dead allocation — repoint it to a live ModCon
  dir (or override just uv: `export UV_CACHE_DIR=$PROJROOT/.uv_cache`). Check with
  `readlink -f ~/.cache`.
- Quota sanity: `myquota` (a project row with a tiny/`1M` quota = dead, don't use it).

---

## 3. 📤 Get the code onto Aurora

Upload sources + PBS scripts (not a Mac-built `.venv` — wrong platform). From your workstation:

```bash
scp *.pbs *.mojo aurora_setup.sh \
    <user>@aurora.alcf.anl.gov:/lus/flare/projects/ModCon/<user>/MOJO_WORK/AURORA/
```

---

## 4. 🧱 THE key divergence — glibc 2.31 blocks the bare-host install

`uv add mojo` on the login node **fails**:

```
Distribution `mojo==1.0.0b2` ... doesn't have a wheel for the current platform.
You're on Linux (manylinux_2_31_x86_64), but mojo only has wheels for:
manylinux_2_34_aarch64, manylinux_2_34_x86_64, macosx_13_0_arm64
```

Every Modular wheel — surveyed across **all** versions of `mojo`, `mojo-compiler`, and `max` on
PyPI — is **`manylinux_2_34_x86_64`**, requiring **glibc ≥ 2.34**. Aurora has **glibc 2.31**.
There is no older/more-portable wheel to pin to. (Polaris ran the *same* `mojo==1.0.0b2` fine
because its login glibc is ≥ 2.34; Aurora is the older userspace.)

> ⛔ Do **not** follow uv's `required-environments` hint — it would install the 2.34 wheel that
> then dies at runtime with `GLIBC_2.34 not found`. The fix is a **container**, not a marker.

The Aurora kernel (5.14) is new enough; only the userspace glibc is short — the textbook case
for a container. We use **Ubuntu 24.04 (glibc 2.39)** via apptainer. Performance is unaffected
(native CPU; only libs come from the image).

> 💡 **The container is not a hack.** Functionally it gives Mojo exactly what a host with
> glibc ≥ 2.34 would — the same real CPU/cores/HBM, just a newer userspace. And ALCF documents
> apptainer as a **recommended, first-class way to run on Aurora** (for reproducibility and
> environment control), so a well-run project would plausibly containerize *anyway*; here it
> simply went from optional-but-sensible to mandatory.

---

## 5. 🐳 Build the container + Mojo venv (feasibility gate) — on a COMPUTE node

Container build/pull needs **user namespaces** (blocked on Aurora *login* nodes, allowed on
*compute*), and `uv add mojo` needs **internet** (compute reaches it only via the proxy). Both
coexist only on a compute node, so the build is a batch job. `aurora_build.pbs`:

```bash
#PBS -N mojo_build
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -j oe
set -e
export http_proxy=http://proxy.alcf.anl.gov:3128        # compute-node internet
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
module load apptainer/1.2.5
PROJROOT=/lus/flare/projects/ModCon/$USER/MOJO_WORK/AURORA
IMG="$PROJROOT/ubuntu2404.sif"
cd "$PROJROOT"
# fresh slate: drop any py3.6 pyproject/.python-version left by bare-host attempts
rm -rf .venv uv.lock pyproject.toml .python-version main.py hello.py
[ -f "$IMG" ] || apptainer pull "$IMG" docker://ubuntu:24.04     # glibc 2.39
apptainer exec --bind /lus \
  --env http_proxy="$http_proxy" --env https_proxy="$https_proxy" \
  "$IMG" bash -lc '
    export PATH="$HOME/.local/bin:$PATH"
    export UV_CACHE_DIR="'"$PROJROOT"'/.uv_cache"
    cd "'"$PROJROOT"'"
    uv python install 3.12
    uv init --name mojo-aurora --python 3.12
    uv add mojo --prerelease allow --python 3.12
    uv lock && uv sync
    uv run mojo --version
'
```

```bash
qsub aurora_build.pbs        # → mojo_build.o<jobid>
qstat -u $USER               # confirm Q/R (see gotcha below)
```

Success = the log ends with **`Mojo 1.0.0b2 (2cf4d08a)`**. The host `uv` binary (installed in
Step 1) is reused inside the container — `$HOME` is auto-bound, and because the container's
glibc is 2.39, `uv` now sees `manylinux_2_39` and the 2.34 wheel resolves. uv also fetches a
managed **CPython 3.12.13** (Aurora's system 3.6 is too old, and any stale `.python-version`
pinned to 3.6 must be removed — see Troubleshooting).

---

## 6. 🚀 Running Mojo jobs (the proven recipe)

Every Mojo invocation goes **through the container**, reading the pre-built `.venv` on flare
(no network needed at run time). Two scripts:

- `aurora_l0.pbs` — runs the L0 smoke test.
- `aurora_run.pbs` — **generic runner**, takes the file via `-v`:

```bash
cd /lus/flare/projects/ModCon/$USER/MOJO_WORK/AURORA
qsub aurora_l0.pbs                                          # → mojo_l0.o<jobid>
qsub -v MOJOFILE=00_simd_type.mojo   aurora_run.pbs         # L2
qsub -v MOJOFILE=02_vecadd_parallel.mojo aurora_run.pbs     # L3
qsub -v MOJOFILE=03e_matmul_cpu.mojo aurora_run.pbs         # L4
```

The core of `aurora_run.pbs`:
```bash
module load apptainer/1.2.5
PROJROOT=/lus/flare/projects/ModCon/$USER/MOJO_WORK/AURORA
IMG="$PROJROOT/ubuntu2404.sif"
: "${MOJOFILE:?set via qsub -v MOJOFILE=<file>.mojo}"
apptainer exec --bind /lus "$IMG" "$PROJROOT/.venv/bin/mojo" run "$PROJROOT/$MOJOFILE"
```

Notes:
- **Use absolute paths** to `.venv/bin/mojo` and the source (Polaris lesson).
- **`--bind /lus`** is required — flare is not auto-mounted; `$HOME` is.
- **Batch beat interactive** — `qsub -I` works but the interactive queue was backed up; batch
  was smoother (also the Polaris preference). `qsub -I -l select=1,walltime=01:00:00 -q debug
  -A ModCon -l filesystems=flare:home` if you want a shell.
- Harmless noise in every log: `Failed to initialize Crashpad…` and `gocryptfs not found`.

---

## 7. 🧯 Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `uv add mojo` → "only has wheels for manylinux_2_34" | Aurora glibc 2.31 < 2.34 | Install/run inside a glibc-2.39 container (Steps 4–6) — **not** the `required-environments` hint |
| `Failed to create user namespace: … max_user_namespaces exceeded` | login nodes block unprivileged userns | Do all apptainer build/pull/exec on a **compute** node (Step 5) |
| `uv add` → "No interpreter found for Python 3.6" | stale `.python-version` (pinned to system 3.6) from a bare-host `uv init` | `rm -f .python-version main.py hello.py`; `uv python install 3.12` + `--python 3.12` (Step 5) |
| `qsub: directive error: -A ModCon # ...` | Aurora PBS rejects inline comments after `#PBS` values | keep directive lines bare; put comments on their own lines |
| `Disk quota exceeded` on write | cache/dir on the dead `candle_aesp_CNDA` allocation | point `$PROJROOT` / `~/.cache` / `UV_CACHE_DIR` at live `ModCon` (Step 2) |
| container can't reach the internet in the build | compute nodes are offline except via proxy | `export http_proxy=https_proxy=http://proxy.alcf.anl.gov:3128` in the job (Step 5) |
| `qsub` ran but produced no job/output | submission didn't register | always `qstat -u $USER` right after; confirm Q/R |
| `which mojo` empty | mojo is venv-local, container-only | run via `apptainer exec … .venv/bin/mojo` (Step 6) |

---

## 8. 🧭 What transfers from Polaris — and what's different

- **Transfers:** live-storage discipline + the dead-allocation trap, login-vs-compute split, the
  PBS batch pattern with absolute venv paths, `-A ModCon`, and "verify the submit with `qstat`".
- **Different on Aurora:**
  - **No conda/frameworks module** — `uv` via `curl` instead (login has direct internet).
  - **`-l filesystems=flare:home`** (Polaris used `home:eagle`).
  - **The whole toolchain lives in a container** (glibc gate) — there is no bare-host `uv run
    mojo`; everything is `apptainer exec`.
  - **No `ptxas`/CUDA driver workaround** — that is NVIDIA-only and meaningless here (no CUDA,
    no NVIDIA driver, no GPU backend).
  - Container build must be on a **compute** node (userns); Polaris built on login.

---

## 📎 Appendix — key facts at a glance

| | |
|---|---|
| Login prep | `curl -LsSf https://astral.sh/uv/install.sh \| sh` (no module stack; direct internet) |
| Workdir (`$PROJROOT`) | `/lus/flare/projects/ModCon/<user>/MOJO_WORK/AURORA` |
| Allocation | `-A ModCon`, `-q debug`, `-l filesystems=flare:home` |
| OS / libc | SLES 15-SP4, **glibc 2.31** (too old for Mojo wheels → container) |
| Container | `ubuntu2404.sif` (glibc 2.39), `module load apptainer/1.2.5`, built on a **compute** node |
| Compute proxy | `http_proxy=https_proxy=http://proxy.alcf.anl.gov:3128` |
| Mojo | `1.0.0b2`, CPython 3.12.13, venv `$PROJROOT/.venv` |
| Run pattern | `apptainer exec --bind /lus ubuntu2404.sif .venv/bin/mojo run <file>` |
| CPU | 2× Xeon Max (Sapphire Rapids), 102 phys / 204 log cores, AVX-512 (f32 W=16), HBM |
| GPU | 6× Intel GPU Max — **not usable** from Mojo (no backend) |
