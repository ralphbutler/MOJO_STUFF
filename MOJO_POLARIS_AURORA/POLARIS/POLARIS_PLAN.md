# 🚀 Mojo on Polaris (Argonne) — Plan

> **Malleable, not gospel.** This is a checklist we revise as we learn. If reality
> contradicts a line here, change the line. Check boxes off as we go.

## 🧭 Fresh session? Start here (this dir is memory-blind)

A session rooted in this subdir does **not** auto-load the `MOJO_CURRICULUM` project
memory — this plan is the source of truth. Context you'd otherwise get from memory:

- **Why:** part of an evaluation of switching Argonne's stack to the Mojo/MAX ecosystem.
  Polaris (NVIDIA) is the "should just work" target; get real A100 numbers + a MAX
  serving endpoint as evidence.
- **Canonical teaching source:** `../../MOJO_CURRICULUM` — files here are working copies;
  don't edit teaching copies there. Sibling `../AURORA/` is the CPU-only Intel-GPU target.
- **Run:** `uv run mojo <file>` (venv lives at the `MOJO_WORK/` root, one level up).
  The `mojo-syntax` skill is available and **must** be used when writing Mojo.
- **First action once the ALCF account is live:** P0 below (toolchain smoke).
- Note: on Polaris itself there's no venv yet — build one per the logistics section;
  `uv run` working locally just means the kit is sound.

## 📌 Where we are (updated 2026-07-07 evening)

**P0 is essentially DONE — toolchain proven on Polaris.** Only `00_simd_type` left to run.

- **Login incantation (every fresh shell):**
  `module use /soft/modulefiles && module load conda && conda activate base`
  → gives `uv` + CPython 3.12.11. No `UV_CACHE_DIR` override needed (see storage).
- **Storage (resolved the hard way):** old project `candle_aesp` eagle quota is **0k = dead**
  (can't write a byte). Now on **`ModCon`**: working dir
  `/lus/eagle/projects/ModCon/rbutler/MOJO_WORK`, and `~/.cache` → ModCon dotcache
  (real quota). `#PBS -A ModCon`, `-l filesystems=home:eagle`.
- **Done:** `uv sync` ✅ (installed `mojo==1.0.0b2` + compiler/lldb libs) ·
  `uv run mojo --version` → `Mojo 1.0.0b2 (2cf4d08a)` ✅ · `uv run mojo hello.mojo` ✅.
- **Access pattern:** `uv run mojo <file>` on login; `.venv/bin/mojo <file>` inside jobs
  (`which mojo` returns nothing — it's venv-local, that's expected).
- **P0 ✅ P1 ✅ P2 ✅ P3 ✅ (2026-07-07); P4a ✅ (2026-07-08).** All numbers in `POLARIS_RESULTS.md`.
  - **MAX installed** (separate env): `uv venv .venv-max --python 3.12 &&
    uv pip install --python .venv-max "modular==26.4.0"` → `.venv-max/bin/max` = **MAX 26.4.0**.
    `mojo==1.0.0b2` has no matching `max`; max/modular use CalVer — use the `modular` umbrella.
  - **P4a done:** `max generate` ran Qwen2.5-0.5B on A100 (310 tok/s). MAX needs the **same**
    `MODULAR_NVPTX_COMPILER_PATH` ptxas fix as Mojo, plus `HF_HUB_OFFLINE=1`. Job = `p4_generate.pbs`.
  - **NEXT = P4b (in-process Python API, now the PRIMARY deliverable):** `from max.entrypoints.llm
    import LLM` in a script run by `.venv-max/bin/python` on a compute node — the real HPC usage
    pattern (embed inference in a program), vs the serve/endpoint case which is optional (P4c).
    First step: confirm the `LLM` API surface on login (`inspect.signature`, no GPU needed), then
    a batch job with the same ptxas + HF_OFFLINE exports.
  - **P4c (`max serve` + litellm) = OPTIONAL**, only for a shared/multi-client endpoint. Needs
    litellm installed into `.venv-max` on login first (`uv run --with litellm` fails offline).
- **Polaris GPU job recipe (batch, proven):**
  ```bash
  #PBS -A ModCon -q debug -l select=1:system=polaris -l walltime=00:10:00 -l filesystems=home:eagle
  ROOT=/lus/eagle/projects/ModCon/rbutler/MOJO_WORK
  cd "$ROOT/POLARIS"
  module use /soft/modulefiles && module load conda && conda activate base
  export MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas
  "$ROOT/.venv/bin/mojo" <file.mojo>
  ```
- **File CPU/GPU map:** CPU/login = `hello`, `00_simd_type`. GPU/compute-node =
  `02_vecadd_gpu`, `03a/b/c/d_matmul_*`, `04b_train_mlp_gpu`.

## 🖥️ Machine profile (verify on first login)

- **GPUs:** 4× NVIDIA A100 (40 GB) per node — Mojo's **CUDA backend applies**, so our
  GPU kernels should run essentially unchanged. This is the home-turf target.
- **CPU:** AMD EPYC (Milan). **Scheduler:** PBS Pro (`qsub`, `-l filesystems=...`),
  same family as Aurora — the `aurora_*.pbs` script is a good template.
- **Contrast with Aurora:** here `has_accelerator()` should return **True** and the GPU
  matmuls should actually execute. That's the whole reason Polaris comes first.

## 🎯 Goal

Prove the Mojo/MAX toolchain on Argonne NVIDIA iron, get **real A100 numbers**, and stand
up a **MAX serving endpoint** — the concrete evidence for the Mojo-switch evaluation.

## 🪜 The ladder (check off as accomplished)

- [x] **P0 — Toolchain smoke (login node).** DONE 2026-07-07: `uv sync` (mojo==1.0.0b2),
      `mojo --version`, `hello.mojo`, and `00_simd_type` all ✅. Toolchain proven on Polaris.
      CORRECTION: original said "CPU matmul `03a`" — wrong, `03a` is GPU → moved to P2.
- [x] **P1 — GPU is live (compute node).** DONE 2026-07-07: `02_vecadd_gpu.mojo` on an A100
      → `RESULT PASS`, 0/1000000 mismatches, 0.00766 ms/pass. `has_accelerator()==True`,
      kernel launched. **Required for ALL GPU runs:**
      `export MODULAR_NVPTX_COMPILER_PATH=/soft/compilers/cudatoolkit/cuda-12.8.1/bin/ptxas`
      (Polaris driver 570/CUDA 12.8 < Mojo's required driver 580/CUDA 13, so AOT-compile via
      system ptxas). Used a batch job, not `qsub -I` — worked fine.
- [x] **P2 — Real workload numbers.** DONE 2026-07-07 (N=2048): naive 2,659 · tiled 4,137 ·
      coarse **8,966 GFLOP/s** (~46% of A100 fp32 peak); `03d` PASS 0/65536. A100 section +
      Apple-vs-NVIDIA "tiling flip" table added to `RESULTS01.md`. Key finding: simple 16×16
      tiling LOSES on Apple (0.80×) but WINS on A100 (1.56×) — same source. Portability proven.
- [x] **P3 — The money shot.** DONE 2026-07-07. (a) `04b` MLP training on A100 reproduced its
      loss curve bit-for-bit (2.178→0.00056), 0.116 ms/epoch — full train loop ports with zero
      numeric drift. (b) Matmul vs cuBLAS (N=2048, true fp32): Mojo coarse **8,966 GFLOP/s =
      ~65% of cuBLAS's 13,839**. See POLARIS_RESULTS.md. Optional TODO: matched-dim timed
      training race (04b dims == Torch + time Torch loop).
- [~] **P4 — Inference deliverable.** *(Reframed 2026-07-08: HPC uses inference embedded in a
      program on a compute node, so in-process load-and-use is the representative pattern; serving
      is an optional shared-endpoint case, more of a workstation/production scenario.)*
      - [x] **P4a — `max generate`** (CLI one-shot, no server): MAX 26.4.0 ran Qwen2.5-0.5B on an
        A100, 310 tok/s. MAX needs the **same** `MODULAR_NVPTX_COMPILER_PATH` ptxas workaround as
        Mojo (+ `HF_HUB_OFFLINE=1`).
      - [x] **P4b — in-process Python API (PRIMARY): DONE 2026-07-08.** `from max.entrypoints.llm
        import LLM; LLM(PipelineConfig(model_path=...)).generate([...])` under `.venv-max/bin/python`
        on A100. Loaded Qwen2.5-0.5B once (44.9 s), 3 prompts in 0.58 s (load amortized). Script
        MUST use `if __name__ == "__main__":` (LLM spawns a multiprocessing telemetry worker).
        File: `p4b_inprocess.py`. This is how MAX would actually be used on Polaris/Aurora.
      - [ ] **P4c — `max serve` + litellm (OPTIONAL):** persistent OpenAI-compatible endpoint; only
        needed for a shared/multi-client service or cross-request continuous batching. Documented,
        not required. (`max_serve_litellm.sh` is the Mac template.)

## 📦 Files staged here

`hello.mojo` · `00_simd_type.mojo` · `02_vecadd_gpu.mojo` · `03a/b/c/d_matmul_*.mojo` ·
`04b_train_mlp_gpu.mojo` · `bench_torch.py` · `train_torch_mlp.py` ·
`max_serve_litellm.sh` · `max_generate.sh` · `RESULTS01.md` (Mac baseline for comparison).

## ⚠️ Logistics unknowns (the risk is the environment, not the code)

- Will the Mojo pip wheel install/run on Polaris (HPE Cray / SLES-ish, glibc)? → P0 is the gate.
- **Login/compute split (observed 2026-07-06):** `nvcc` works on login (toolkit module
  is present) but `nvidia-smi` errors — no GPU exposed there. So P0 runs entirely on
  login; P1+ needs a compute node (`qsub -I` for interactive iteration).
- Login-node internet via proxy; compute nodes likely offline → install + `uv sync` on
  login, run `.venv/bin/mojo` directly in the job (mirror the Aurora setup approach).
- CUDA/driver module: which `module load` gives the toolkit MAX/Mojo expects?
- A `polaris_check.sh` analog of `mac_local_check.sh` — write once we see the env.

## ❓ Unresolved (confirm once account is fixed — later today)

1. ~~Account/allocation name for `#PBS -A`? queue? `-l filesystems=` values?~~
   ANSWERED: `-A ModCon`, interactive queue `debug` (2 nodes, 1h), `-l filesystems=home:eagle`.
2. ~~`uv` allowed, or module/conda only?~~ ANSWERED: `uv` available via `module load conda`.
3. A100 count/MIG per node as scheduled?
4. Target serve models actually available on Polaris storage (Gemma4 / Qwen3.6)?
5. P-day scope: just "it runs + numbers" (P0–P2), or push to serving (P4)?
