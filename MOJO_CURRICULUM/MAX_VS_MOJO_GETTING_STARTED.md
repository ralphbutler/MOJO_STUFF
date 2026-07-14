# 🎓 Getting Students Started: MAX vs Mojo

Notes on how to onboard a class into the Modular ecosystem, and where to run it.
(Draft for later thinking — not decided.)

## 🍴 Should students start with "MAX high-level"?

Depends what "MAX high-level" means, because it splits two ways:

- **MAX as a serving platform** — deploy a model, hit an OpenAI-compatible
  endpoint. Genuinely easy and impressive on day one, but teaches almost **no Mojo
  and no GPU programming**. It's an MLOps / deployment skill. Great if the goal is
  "use models productively"; nearly useless if the goal is "port kernels to
  accelerators."
- **MAX graph API in Python** — build / compile model graphs. A middle layer: some
  performance intuition, still not the language.

So the real question is *what are these students for?* For kernel porting to HPC
GPUs, the durable skill is **Mojo the language**; MAX-serving is a motivating demo,
not the curriculum.

### Suggested arc
1. **Day 1 hook:** run a model on MAX → "one command, faster than vLLM." Builds
   excitement.
2. **Then drop to Mojo:** CPU basics → GPU kernels (vector add, matmul, tiling).
   This is the transferable part.
3. **Later:** custom MAX ops in Mojo — connects the two, shows *why* the language
   exists.

> MAX-first as *hook*, Mojo as *substance*. Don't let students camp in MAX serving
> thinking they're learning the ecosystem — that's learning to drive, not to build
> the car.

## 🌐 Where to run it

> **✅ CORRECTION (verified 2026-07-05, M4 Max):** The old "not Macs" claim was
> wrong for **MAX inference**. `uv pip install "max[serve]"` from **plain PyPI**
> (max 26.4.0) installs on macOS arm64, and `max generate` ran Qwen2.5-0.5B on the
> **Apple Silicon GPU** (`devices: gpu[0]`, Metal) at ~14.9 tok/s / 323ms TTFT.
> No pixi, no Modular index, no NVIDIA required. So a Mac is a *fine* box for the
> Day-1 "run a model on MAX" hook. Two caveats remain: (a) **Mojo GPU-kernel
> programming** (matmul/tiling) is the part that's still NVIDIA/AMD-centric — the
> Metal backend for hand-written kernels is less complete than for MAX's shipped
> ops; (b) shared-pool/cloud reasons below still apply for a *class*.

For heavier / shared GPU work, off-Mac options:

- **Google Colab** — free NVIDIA T4, zero setup, install Mojo/MAX via pixi. *The*
  student answer for GPU kernels. Verify the install path before committing a class.
  **(Currently the leading candidate.)**
- **GPU cloud / partners** (Spheron, Lambda, RunPod, Vast.ai) — cheap hourly
  NVIDIA, good for a shared class pool.
- **University / HPC allocation** — **Polaris (NVIDIA A100)** is the natural
  learning box before Aurora. *(Possible, but undecided.)*

## 🍎 macOS gotcha: REPL dies in Desktop/Documents/Downloads

If a student does local Mojo on a Mac and the project lives under **`~/Desktop`,
`~/Documents`, or `~/Downloads`**, the REPL (`mojo` / `mojo repl`) crashes on the
first expression with a wall of `dyld: Library not loaded: libMSupportGlobals.dylib`
+ `file system sandbox blocked open()` and a **SIGKILL** (exit status 9). It looks
like a broken install, but it is **not** — those three folders are macOS
privacy-protected (TCC), and macOS blocks the REPL's runtime `dlopen`. A clean
reinstall does **not** fix it; the compiler path (`mojo file.mojo`) is unaffected,
which makes it extra confusing.

**Fixes (any one):**
1. Grant the terminal (iTerm → `/Applications/iTerm.app`, or Terminal) **Full Disk
   Access**: System Settings ▸ Privacy & Security ▸ Full Disk Access ▸ **+** ▸ add
   the app ▸ toggle ON ▸ **⌘Q and relaunch** (closing the window is not enough).
   **✅ CONFIRMED WORKING (2026-07-05):** after granting iTerm2 FDA, `from max import
   engine` and a full `max generate` run both succeed with the project under
   `~/Desktop` — no SIGKILL. So Fix #1 resolves it for MAX, not just the Mojo REPL.
2. Keep the project **outside** Desktop/Documents/Downloads (e.g. `~/Dev/…`).
3. Skip the REPL; run scratch files with `mojo file.mojo` (works even under Desktop).

> Tell students this up front — a fresh clone onto the Desktop hits it immediately
> and reads as "Mojo is broken."

## 🧪 Colab install path (checked 2026-07-05)

**Compatibility: VERIFIED against Modular docs — Colab qualifies.**

| Requirement | Modular needs | Colab provides | ✓ |
|---|---|---|---|
| OS | Linux, glibc 2.34+ (Ubuntu 22.04+) | Ubuntu 22.04 | ✓ |
| Arch | x86-64-v3 (Haswell+) | modern x86_64 | ✓ |
| Python | 3.10–3.14 | 3.11 / 3.12 | ✓ |
| GPU | NVIDIA (`cuda` backend) | T4 (free tier) | ✓ |
| Driver | NVIDIA 580+ | Colab keeps current | ✓ (confirm at runtime) |

**Install command: NEEDS ONE TEST RUN.** Official installers are `pixi` and `uv`,
which make their own env (awkward in a notebook). The Colab-friendly pip path below
is *inferred* from the uv nightly index URL — docs show `pip` only for uninstall —
so run it once to confirm before relying on it.

```python
# Colab cell — MAX + Mojo (nightly). Set Runtime > Change runtime type > T4 GPU first.
!pip install modular --index-url https://whl.modular.com/nightly/simple/ --pre -q
!mojo --version
!nvidia-smi --query-gpu=driver_version,name --format=csv
```

Then in a Python cell: `from max import engine` (or run a kernel via `!mojo run`).
**Fallback if pip balks:** use `uv` inside Colab (clunkier but documented/guaranteed).

## ⚠️ The Aurora landmine

If the endgame is Aurora specifically: **Mojo has no Intel-GPU backend today**
(only `cuda` / `hip` / `metal`). Aurora's Intel Max GPUs aren't a Mojo target yet.
Students would learn Mojo on **NVIDIA/AMD**, and the Aurora port depends on Modular
shipping an Intel / Level-Zero (or SPIR-V) backend that doesn't publicly exist yet.
Worth knowing before building a curriculum aimed at Aurora.

## 📌 Open items / to revisit
- Decide platform: **Colab (leading)** vs **Polaris** vs cloud pool. *Very unsure.*
  Note the Day-1 MAX hook now also works **locally on Mac GPU** (verified), so a Mac
  is viable for the demo even if kernel work goes to Colab/Polaris.
- ~~I have never used MAX myself.~~ First hands-on done 2026-07-05: installed
  `max[serve]` on the M4 Max, ran Qwen2.5-0.5B on the Metal GPU. It's real and easy.
- ~~Verify Mojo/MAX install path on Colab.~~ Compatibility verified; pip command
  still needs one real notebook run (see "Colab install path" section).
- On **macOS the install is just `uv pip install "max[serve]"` from PyPI** — simpler
  than the Colab/Modular-index path documented above. Re-check whether the same plain
  PyPI path works on Colab/Linux (may avoid the nightly-index dance entirely).
- Watch for any public signal on Mojo → Intel-GPU (Aurora) support.
