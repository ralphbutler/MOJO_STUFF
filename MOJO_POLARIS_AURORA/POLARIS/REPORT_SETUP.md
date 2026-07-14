# 🛠️ REPORT — Reproducing Mojo/MAX on ALCF Polaris (from zero)

**Purpose:** an ordered, copy-pasteable procedure to stand up the Mojo/MAX toolchain on
Polaris and run GPU workloads on the A100s, capturing every gotcha we hit so the next person
(or an Aurora port) doesn't rediscover them.
**Status:** verified end-to-end 2026-07-08 — Mojo toolchain, A100 GPU kernels, training, and MAX
LLM inference (CLI + in-process) all working. Optional `max serve` endpoint not exercised.
**Companion docs:** `REPORT_RESULTS.md` (numbers), `POLARIS_PLAN.md` (ladder), `RESULTS01.md`
(matmul deep-dive).

---

## 0. 📋 Prerequisites

- An ALCF Polaris account with an **active allocation** to charge jobs to (ours: `ModCon`).
  Verify with your PI — a *storage* directory alone is not enough; you need a live compute
  allocation and to be in its unix group.
- Basic layout we used: a project dir `MOJO_WORK/` holding a `pyproject.toml`
  (`mojo>=1.0.0b2`, `requires-python>=3.12`) at its root, with GPU source files in a
  `POLARIS/` subdir. The venv is built **at the root** and shared by the subdir.

> ⚠️ **Machine profile (verify on first login):** 4× NVIDIA **A100-SXM4-40GB** per node,
> AMD EPYC (Milan) CPUs, **PBS Pro** scheduler. NVIDIA driver **570.124.06 (CUDA 12.8)** —
> this exact version drives the single most important workaround (Step 5).

---

## 1. 🔑 Login environment (every fresh shell)

A bare login shell has no usable `python`/`uv`. Load the conda module stack first:

```bash
module use /soft/modulefiles && module load conda && conda activate base
```

This gives `uv` + CPython **3.12.11** (from the ALCF conda `2025-09-25` mconda3). Loading
conda **rewrites `MODULEPATH`** (you'll see Lmod auto-swap `PrgEnv-nvidia`→`PrgEnv-gnu`); this
is normal but has a side effect noted in Step 5.

---

## 2. 💾 Storage — avoid the dead-allocation trap

Home directories are tiny (50 GB) and, critically, **an expired/retired project's eagle space
can have a zero quota** — you can read it but cannot write a single byte. We lost time to this:
the old `candle_aesp` eagle allocation showed `quota 0k / limit 0k`, so `uv sync` failed with
`Disk quota exceeded` while extracting.

**Do this:**
- Put the working dir + venv on a **live** project filesystem, e.g.
  `/lus/eagle/projects/<PROJECT>/<user>/MOJO_WORK`.
- Make sure `~/.cache` resolves to live storage too (uv/pip/HF all write there). Check:
  ```bash
  readlink -f ~/.cache          # must NOT point at a dead/zero-quota allocation
  ```
  If it points somewhere dead, repoint the symlink to a live project dir, **or** override just
  uv's cache: `export UV_CACHE_DIR=/lus/eagle/projects/<PROJECT>/<user>/.uv_cache`.
- Quick quota sanity check: `myquota` (a project row showing `quota 0k` = dead, don't use it).

---

## 3. 📤 Get the code onto Polaris

Upload only what's needed — **not** a Mac-built `.venv` (wrong platform; it just gets deleted
and rebuilt). From your workstation:

```bash
rsync -av --exclude '.venv' --exclude '.claude' \
  /path/to/MOJO_WORK/ <polaris>:/lus/eagle/projects/<PROJECT>/<user>/MOJO_WORK/
```

You need the root `pyproject.toml` + `uv.lock` + `.python-version` (they drive `uv sync`) and
your source subdir.

---

## 4. 🧱 Build the Mojo venv (P0 feasibility gate)

On the **login** node (it has an HTTP proxy — `proxy.alcf.anl.gov:3128` — for downloads;
compute nodes are offline):

```bash
cd /lus/eagle/projects/<PROJECT>/<user>/MOJO_WORK
uv sync
```

Success = `Installed N packages` including `mojo==1.0.0b2` (+ `mojo-compiler`,
`mojo-lldb-libs`, …). The venv lands at `MOJO_WORK/.venv`.

**Smoke test (login/CPU, no GPU needed):**
```bash
uv run mojo --version          # → Mojo 1.0.0b2 (2cf4d08a)
uv run mojo hello.mojo
uv run mojo 00_simd_type.mojo
```
If these run, the wheel is sound on Polaris — the core feasibility gate is cleared.

> 💡 `which mojo` returns nothing — that's expected. `mojo` lives inside the venv. Use
> `uv run mojo …` on login, or the **absolute** `.venv/bin/mojo …` inside jobs.

---

## 5. 🎯 THE key workaround — Mojo GPU needs a system `ptxas`

Mojo 1.0.0b2 requires NVIDIA driver **≥ 580 (CUDA ≥ 13.0)**. Polaris runs **570 / CUDA 12.8**,
so a GPU kernel fails at launch:

```
Your current NVIDIA GPU driver version is not supported.
  Required: driver version >= 580 (CUDA >= 13.0)
  Detected: 570.124.06 (CUDA 12.8)
```

**Fix:** AOT-compile through a *system* `ptxas` whose CUDA version is **≤ the driver's**:

```bash
export MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas
```

- Use **12.8** — it matches the driver exactly. **Do NOT** use a `cuda-13.x` ptxas: its cubins
  won't load on the 12.8 driver.
- The `cudatoolkit-standalone` **module** is *not* visible after `module load conda` (the
  MODULEPATH swap hides it). Don't fight it — point at the binary under
  `/soft/compilers/cudatoolkit/cuda-*/bin/ptxas` directly (list them: `ls -d
  /soft/compilers/cudatoolkit/cuda-*`).

---

## 6. 🚀 Running GPU jobs (the proven PBS recipe)

Compute nodes have the A100s (`nvidia-smi` works there, not on login). Use a batch job — the
`debug` queue caps you at **one queued job at a time**, so put multiple steps in one script.

```bash
cat > job.pbs <<'EOF'
#!/bin/bash
#PBS -A ModCon
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:eagle
ROOT=/lus/eagle/projects/ModCon/rbutler/MOJO_WORK
cd "$ROOT/POLARIS"
module use /soft/modulefiles && module load conda && conda activate base
export MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas
echo "=== node: $(hostname) ==="
nvidia-smi -L
"$ROOT/.venv/bin/mojo" 02_vecadd_gpu.mojo
EOF
qsub job.pbs                       # prints a job ID; output → job.pbs.o<ID>, errors → job.pbs.e<ID>
```

Notes:
- **Absolute paths matter.** The venv is at `MOJO_WORK/.venv` (root), source in
  `MOJO_WORK/POLARIS`. A relative `.venv/bin/mojo` from the subdir won't resolve — always use
  `$ROOT/.venv/bin/mojo`.
- For interactive iteration instead of batch: `qsub -I -A ModCon -q debug -l
  select=1:system=polaris -l walltime=00:30:00 -l filesystems=home:eagle` (drops you onto a
  compute node; then re-run the module + export lines).
- Harmless noise in output: `Failed to initialize Crashpad…` and the Lmod PrgEnv swap lines.

---

## 7. 🔥 PyTorch comparison (no install needed)

The ALCF conda base already ships **PyTorch 2.8.0** (CUDA 12.9 build; runs fine on the 12.8
driver). Torch scripts run under conda `python` — a **separate process** from the mojo venv:

```bash
python train_torch_mlp.py          # auto-selects device=cuda on Polaris (no MPS)
python bench_torch.py 2048         # cuBLAS matmul; our patched version benches cuda w/ TF32 OFF
```

For a fair fp32 comparison against a hand-written kernel, force true fp32:
`torch.backends.cuda.matmul.allow_tf32 = False`.

---

## 8. 🤖 MAX (serving / inference) — install it separately

**MAX is versioned independently of Mojo.** `mojo` is on a bleeding-edge `1.0.0b2` line;
`max`/`modular` use CalVer (latest **26.4.0**). There is **no `max` matching `mojo==1.0.0b2`**
(`uv add "max==1.0.0b2"` → unsatisfiable). MAX serving doesn't use the Mojo compiler, so give
it its own env and use the **`modular`** umbrella (the bare `max` wheel under-declares deps and
dies on `ModuleNotFoundError: tqdm`):

```bash
cd /lus/eagle/projects/ModCon/rbutler/MOJO_WORK
uv venv .venv-max --python 3.12
uv pip install --python .venv-max "modular==26.4.0"
.venv-max/bin/max --version        # → MAX 26.4.0
```

**Pre-download model weights on the LOGIN node** (compute nodes are offline):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
```

Then in a GPU job, invoke `.venv-max/bin/max generate …` (one-shot) or `.venv-max/bin/max
serve …` (endpoint). **MAX needs the SAME driver workaround as the Mojo compiler** — it hits
the ≥580 driver wall and is fixed by the *same* `MODULAR_NVPTX_COMPILER_PATH` export (Step 5).
Also set `export HF_HUB_OFFLINE=1` in the job so MAX uses the pre-cached weights. Verified
working: `max generate` loaded Qwen2.5-0.5B on an A100 at ~310 tok/s (one-time ~110 s model
compile). Example job:

```bash
#PBS -A ModCon -q debug -l select=1:system=polaris -l walltime=00:20:00 -l filesystems=home:eagle
ROOT=/lus/eagle/projects/ModCon/rbutler/MOJO_WORK
cd "$ROOT/POLARIS"
module use /soft/modulefiles && module load conda && conda activate base
export MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas
export HF_HUB_OFFLINE=1
"$ROOT/.venv-max/bin/max" generate --model-path Qwen/Qwen2.5-0.5B-Instruct --prompt "..."
```

**In-process use (the primary HPC pattern — no server).** To embed inference in your own
program, use the Python API under `.venv-max/bin/python` with the same env vars:

```python
from max.entrypoints.llm import LLM
from max.pipelines import PipelineConfig          # fallback: max.entrypoints.pipelines

def main():                                        # REQUIRED guard, see below
    llm = LLM(PipelineConfig(model_path="Qwen/Qwen2.5-0.5B-Instruct"))   # load once
    for out in llm.generate(["prompt A", "prompt B"], max_new_tokens=100, use_tqdm=False):
        print(out)

if __name__ == "__main__":                         # MAX's LLM spawns a telemetry worker via
    main()                                         # multiprocessing(spawn) that re-imports this
                                                   # file — without the guard it re-runs and
                                                   # raises "start a new process before
                                                   # bootstrapping finished".
```

Verified: loaded Qwen2.5-0.5B **once (~45 s)**, then answered 3 prompts in **0.58 s** (load
amortized). Harmless warnings you'll see: `Penalties ... ignoring`, a leaked-semaphore shutdown
notice, and a graph-capture batch cap — none affect output.

The `max serve` + litellm path (a persistent OpenAI-compatible endpoint) is optional and only
needed for a shared/multi-client service; if you use it, install litellm into `.venv-max` on the
login node first (`uv pip install --python .venv-max litellm`) since `uv run --with litellm`
can't fetch on an offline compute node.

---

## 9. 🧯 Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `uv sync` → `Disk quota exceeded` | cache/home on a dead (0k) allocation | Point `~/.cache` / `UV_CACHE_DIR` at a live project dir (Step 2) |
| `.venv/bin/mojo: No such file` in a job | relative path; venv is at repo root not subdir | Use absolute `$ROOT/.venv/bin/mojo` (Step 6) |
| `driver version not supported (>=580)` | Mojo needs newer driver than Polaris has | `MODULAR_NVPTX_COMPILER_PATH=…/cuda-12.8.1/bin/ptxas` (Step 5) |
| `which mojo` empty | mojo is venv-local | `uv run mojo` or `.venv/bin/mojo` (Step 4) |
| `module load cudatoolkit-standalone` → unknown | `conda` swapped MODULEPATH | Point at `/soft/compilers/cudatoolkit/cuda-*/bin/ptxas` directly (Step 5) |
| `qsub` → `would exceed per-user limit of jobs in 'Q'` | debug queue = 1 queued job/user | Wait for the job to start (Q→R), or bundle steps into one `.pbs` |
| `uv add "max==1.0.0b2"` unsatisfiable | MAX uses CalVer, not Mojo's version | `uv pip install "modular==26.4.0"` in a separate venv (Step 8) |
| `max` → `ModuleNotFoundError: tqdm` | installed bare `max` wheel | Use the `modular` umbrella instead (Step 8) |
| MAX/HF can't fetch on compute node | compute nodes are offline | Pre-download weights on the login node (Step 8) |

---

## 10. 🧭 What transfers to Aurora (Intel GPU)

- **Transfers:** everything about *environment & workflow* — conda-module login, live-storage
  discipline, `uv` venv build, login-vs-compute split, the PBS batch pattern, and the
  "install MAX in its own env" approach.
- **Does NOT transfer:** the `ptxas`/driver-580 workaround and all A100 numbers are
  **NVIDIA-specific**. Per our plan, Aurora Mojo is presently **CPU-only** (no Intel-GPU
  backend), so `has_accelerator()` GPU execution won't replicate — Aurora is a
  toolchain-runs-and-CPU-numbers experiment, not a GPU re-run. **Do not apply the CUDA driver
  fix on Intel hardware.**

---

## 📎 Appendix — key facts at a glance

| | |
|---|---|
| Login | `module use /soft/modulefiles && module load conda && conda activate base` |
| Workdir | `/lus/eagle/projects/ModCon/rbutler/MOJO_WORK` |
| Allocation | `-A ModCon`, `-l filesystems=home:eagle`, queue `debug` |
| Mojo | `1.0.0b2`, venv `MOJO_WORK/.venv`, run via `.venv/bin/mojo` |
| GPU env var | `MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas` |
| PyTorch | `2.8.0` in conda base, run via conda `python` |
| MAX | `modular==26.4.0` in `MOJO_WORK/.venv-max`, run via `.venv-max/bin/max` |
| Driver | `570.124.06` / CUDA `12.8` |
