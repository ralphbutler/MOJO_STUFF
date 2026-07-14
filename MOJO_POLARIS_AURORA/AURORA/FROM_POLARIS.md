# 🔗 Carryover from the Polaris port (read this before starting Aurora)

Written 2026-07-08 at the end of the **Polaris (NVIDIA A100)** bring-up, while the lessons
were fresh, so the Aurora port doesn't rediscover them. Aurora is a *different* machine (Intel
CPU, no GPU backend), so treat this as **methodology to adapt**, not values to copy verbatim.

**Reference docs (siblings — relative links work):**
- `../POLARIS/REPORT_SETUP.md` — the full from-zero reproduction guide (the model for what an
  Aurora setup doc should become). Its §1–4, §7, §9 are largely machine-agnostic.
- `../POLARIS/REPORT_RESULTS.md` / `../POLARIS/POLARIS_RESULTS.md` — results + environment findings.

---

## ✅ TRANSFERS to Aurora (methodology — adapt the values, keep the shape)

1. **Storage discipline (the biggest lesson).** On Polaris we lost real time because:
   - HOME is tiny (50 GB) and a **retired allocation's project space can be quota-zeroed**
     (writes fail with `Disk quota exceeded` even though you can read it).
   - `uv sync` extracts a large toolchain; its **cache** (`~/.cache/uv`) and the **venv** must
     live on **live project storage**, not HOME and not a dead allocation.
   - → On Aurora: put the venv + uv cache on a **live Aurora project filesystem**, verify with
     the site's quota tool first, and check `readlink -f ~/.cache`. **This directly contradicts
     the current `aurora_setup.sh`, which installs into `~/mojo-aurora` (HOME) — fix that.**
2. **Login vs compute split.** Install + `uv sync` on the **login** node (has the proxy for
   downloads); run under `qsub` on compute nodes (usually offline). Pre-download anything the
   compute job needs (model weights, etc.) on login first.
3. **Proven PBS recipe *structure*** (values change per site):
   ```bash
   #PBS -A <alloc> -q <queue> -l select=1:system=aurora -l walltime=... -l filesystems=<aurora-fs>
   cd "$WORKDIR"
   <load the site module stack>        # Aurora: NOT the Polaris conda line
   "$VENV/bin/mojo" <file.mojo>         # absolute path to the venv binary
   ```
   - Use **absolute paths** to the venv binary inside jobs (relative paths bit us when the venv
     was one dir up from the sources).
   - **Verify the submit registered:** run `qstat -u $USER` right after `qsub` — we hit a case
     where `qsub` appeared to run but produced no job/output. Confirm you see Q/R.
   - Harmless job-log noise to expect: Lmod PrgEnv auto-swaps, "Failed to initialize Crashpad".
4. **`uv run mojo` on login; `.venv/bin/mojo` in jobs.** `which mojo` returns nothing — it's
   venv-local; that's expected, not an error.
5. **If you attempt MAX (CPU inference) on Aurora:** install it in a **separate venv** via the
   **`modular`** umbrella (`uv pip install --python .venv-max "modular==<ver>"`), NOT the bare
   `max` wheel (under-declares deps). For the in-process Python API, the driver script **must**
   guard its body with `if __name__ == "__main__":` (MAX's `LLM` spawns a multiprocessing
   telemetry worker). Set `HF_HUB_OFFLINE=1` in the job; pre-download weights on login.

---

## ⛔ NVIDIA-ONLY — do NOT copy to Aurora

- **The `ptxas` / driver-580 workaround** (`MODULAR_NVPTX_COMPILER_PATH=…/cuda-12.8.1/bin/ptxas`).
  That fixes an NVIDIA-driver/CUDA mismatch. **Aurora has no CUDA and no NVIDIA driver — this
  line is meaningless (and wrong) there.**
- **All A100 GPU numbers** and any `has_accelerator() == True` expectation. On Aurora
  `has_accelerator()` is **False by design** (no Intel-Xe/Level-Zero backend in Mojo). The GPU
  matmul/train kernels (`02/03a-d/04b`) will not run; Aurora is the **CPU-only** experiment
  (SIMD + `parallelize` on Sapphire Rapids + HBM). This is already the AURORA_PLAN thesis.

---

## ❓ Aurora-specific unknowns to resolve IN the Aurora session (login-node recon first)

- **Module stack:** how do you get `uv`/Python on an Aurora login node? Polaris used
  `module use /soft/modulefiles && module load conda`. Aurora is different (oneAPI /
  `frameworks` module, or a `curl | sh` uv install as `aurora_setup.sh` currently does). Confirm.
- **Filesystems:** the `-l filesystems=` value (Polaris was `home:eagle`; Aurora uses its own —
  e.g. Flare/DAOS/Lustre). Get the right names + your live project path.
- **Allocation & queue names** for `#PBS -A` / `-q` on Aurora (Polaris used `ModCon` / `debug`).
- **Proxy:** confirm Aurora's login-node proxy values (the script has commented placeholders).
- **Does the Mojo x86-64 Linux wheel run on Aurora's SLES/Cray env?** — that's L0, the real gate,
  same role P0 played on Polaris.

---

## 🧭 Recommended first moves in the Aurora session
1. Login-node recon (module stack, `uv`, quota tool, filesystems, allocation) — mirror the
   Polaris "Step 1 recon" but for Aurora's environment.
2. Fix the storage target (live project FS, not HOME) and adjust `aurora_setup.sh` accordingly.
3. L0 smoke (`aurora_l0.pbs`) — the feasibility gate.
4. Then the CPU ladder (L1/L2) per `AURORA_PLAN.md`.
