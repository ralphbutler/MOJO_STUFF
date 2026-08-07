# 🌍 MOJO_WORLD_MODELS_SHARE

A **world model** over the `HIDE_SEEK_LLM` gridworld — learn the environment's transition function from traces, then throw the real simulator away and run agents inside the learned model — with the two compute-bound halves written in **Mojo**.

The deliverable is a measurement, not a framework:

> **Where does Mojo actually beat the Python/PyTorch stack, and by how much, on an honest baseline?**

## 🏁 Key Findings & Benchmark Summary

**All milestones complete — M0 through M8.**

*M1 — Parity.* The Mojo simulator reproduces the Python reference **byte-for-byte across 1,100 episodes** — 200 × 40 turns on grid 12, plus 300 × 100 turns each on grids 8, 16, and 24 — with all 11 tracked code paths exercised.

*M2 — Benchmark.* CPU-to-CPU, 400 episodes × 100 turns:

| | µs/turn | vs Python |
|---|---:|---:|
| Python (simulation only) | 3.948 | 1× |
| **Mojo, 1 thread** | **0.128** | **30.9×** |
| **Mojo, 12 threads** | **0.013** | **299.6×** |

81% parallel efficiency. 75 M turns/s means M3's data generation is effectively free.

*M3 — Dataset.* `data/train`: 100,000 episodes × 100 turns = **10.1 M state vectors, generated in 1.16 s**. 100% of free cells visited, 96.8% unique states, 13.5% sealed-fort frames. Mojo generates maps and actions itself, via an LCG shared with Python; `gen_gate.py` verifies 60/60 episodes element-identical.

*M4 — Dynamics Model.* An MLP with 24 softmax heads (one per state field) reaches **93.0% exact-match** next-state accuracy — every field correct — on episode-held-out data, trained in 160 s on MPS.

*M5 — The Dream.* Mojo's closed-loop rollout matches PyTorch **100.0000%** over 20k steps. Dreams survive a **median of 49 steps** before diverging from reality (6% last the full 100) — nearly 5× better than compounding per-step accuracy predicted, because errors cluster rather than accumulate independently.

**The crossover measured:**

| concurrent dreams | PyTorch µs/step | Mojo µs/step | ratio |
|---:|---:|---:|---:|
| 1 | 387.65 | 267.80 | **1.4×** |
| 8 | 155.93 | 39.14 | **4.0×** |
| 32 | 27.35 | 30.81 | 0.9× |
| 400 | 7.88 | 25.41 | 0.3× |

Mojo wins below ~32 concurrent dreams, BLAS wins above.

*M8 — Evolved Controller.* sep-CMA-ES over a 598-param linear policy, evolved in the **real** Mojo sim, reaches **0.658 held-out** fitness against 0.264 for standing still — **940,800 rollouts / 94 M turns in 5–11 s**, ≥70× a Python equivalent.

*Open Question — What does the dream cost?* The same controller evolved two ways at identical budget, both scored in the **real** sim:

| policy | fitness in reality |
|---|---:|
| static (never moves) | 0.237 |
| **evolved in the DREAM** | **0.241** |
| **evolved in the REAL sim** | **0.393** |

**The dream transfers minimal gain** — 2.4% of the achievable gain. Its own fitness climbed 0.30 → 0.57 across 50 generations while reality never moved: model exploitation, not learning. And it ran **98× slower** than the simulator it replaces.

*M7 — Visualization.* `viewer/compare.html`. Reality and dream, same episode, same actions, side by side; watch the right pane drift.

Traces for seed 424242 ship in `viewer/`, so this works on a fresh checkout with no build:

```bash
python3 -m http.server 8731
open http://localhost:8731/viewer/compare.html    # fetch() needs http, not file://
```

To render a different seed, regenerate first — this needs the build and dataset steps below:

```bash
cd python && uv run python make_trace.py --seed <n>
```

## 💡 The Thesis

Chasing matmul throughput against PyTorch/MPS on GPUs is a losing bet. But a world-model dream loop lives in a different regime — tiny state vector, tiny MLP, millions of strictly sequential steps. The gridworld simulator is further still from BLAS's comfort zone: branchy integer BFS and line-of-sight raycasting.

**This project measures the small-sequential-branchy regime.**

## 🗂️ Layout

```
MOJO_WORLD_MODELS_SHARE/
├── pyproject.toml          # uv deps: mojo==1.0.0b2 (pinned), torch, numpy
├── README.md               # project documentation & benchmark summary
├── PLAN.md                 # design decisions & milestone definitions (code comments cite this)
├── RESULTS_M*.md           # per-milestone measurement write-ups
├── python/
│   ├── gridworld.py        # vendored simulator reference / parity oracle
│   ├── episode.py          # reference driver: map + action export, frame digests
│   ├── parity.py           # M1 gate: byte-diff + coverage accounting
│   ├── bench.py            # M2 benchmark harness
│   ├── gen_gate.py         # M3 gate: Mojo-generated data vs Python reference
│   ├── dataset.py          # M3 coverage report + memmap loader for dynamics model
│   ├── model.py            # M4 dynamics model (24 softmax heads) + weight export
│   ├── train.py            # M4 training + held-out accuracy
│   ├── dream_gate.py       # M5 gates: numerical, divergence, speed crossover
│   └── make_trace.py       # M7 real + dreamed trace.json for the viewer
├── mojo/
│   ├── world.mojo          # Mojo simulator implementation
│   ├── gridworld.mojo      # single-episode runner (parity gate)
│   ├── bench_sim.mojo      # M2 timing harness
│   ├── gen.mojo            # M3 dataset generator (LCG, 4 arms, parallel)
│   ├── dream.mojo          # learned simulator / SIMD MLP rollouts
│   └── evolve.mojo         # M8 sep-CMA-ES controller search
├── viewer/                 # three.js replay viewer + compare.html (real vs dream)
└── data/                   # ep000 spec/digests + trained dynamics weights (.pt for
                            # PyTorch, .weights.bin for Mojo). The 100k-episode
                            # training set is NOT shipped — regenerate it with ./build/gen.
```

## 🚀 Quick Start & Building

**Requirements:**
- **macOS on Apple Silicon** (or Linux with appropriate Mojo environment).
- **[uv](https://docs.astral.sh/uv/)** — Python packaging tool (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

Everything Python-side (`torch`, `numpy`) is declared in `pyproject.toml`, so `uv run`
provisions it on first call — no manual environment setup.

**Rebuild everything from scratch:**

```bash
# 1. Build binaries  (mojo does NOT create the output directory)
mkdir -p build
uv run mojo build mojo/gridworld.mojo -o build/gridworld
uv run mojo build mojo/bench_sim.mojo -o build/bench_sim
uv run mojo build mojo/gen.mojo       -o build/gen
uv run mojo build mojo/dream.mojo     -o build/dream
uv run mojo build mojo/evolve.mojo    -o build/evolve

# 2. Generate dataset (100k episodes, 4-arm mix, ~1.1 s)
./build/gen 12 100 1 data/train 40000 20000 15000 25000

# 3. Train dynamics model (~160 s on MPS) -> data/dynamics.weights.bin
cd python && uv run python train.py --steps 9000 --batch 4096 --hidden 1536 --layers 3
```

**Verification Gates — ensure parity and fidelity:**

Run steps 1 and 2 first. `parity.py` and `gen_gate.py` need only the binaries; the other
two also read `data/train`, which is regenerated rather than shipped (539 MB, ~1.1 s).
Step 3 is not required — trained weights ship in `data/`.

```bash
cd python
uv run python parity.py   --seeds 200 --turns 40     # M1: Mojo sim == Python sim, byte-exact
uv run python gen_gate.py --episodes 80 --turns 100  # M3: Mojo-generated data == Python reference
uv run python dataset.py  ../data/train              # M3: coverage (all actions + fort states)
uv run python dream_gate.py --episodes 200 --steps 100   # M5: Mojo dream == PyTorch, + divergence
```

## ▶️ Other Commands

```bash
# One reference episode (writes data/ep000.spec and data/ep000.digest)
cd python && uv run python episode.py --seed 0 --turns 24 --out ../data/ep000
uv run mojo mojo/gridworld.mojo data/ep000.spec data/ep000.mojo.digest

# M2 benchmark
cd python && uv run python bench.py --episodes 400 --turns 100 --reps 5

# M4 data-scaling curve (one point)
cd python && uv run python train.py --steps 4000 --episodes 15000 --quiet
```

## 🔬 How Parity Works

Three choices make an exact cross-language diff tractable:

1. **RNG never crosses the boundary.** `random.Random` appears only in the Python sim's `_spawn`; everything downstream is deterministic. Python generates the map *and* the full action sequence, exports both, and Mojo replays them.
2. **Actions are state-independent**, so the whole sequence can be sampled up front.
3. **Compare a digest, not JSON.** Both sides emit one canonical line per frame. The gate fails on simulation divergence, not on float formatting or key ordering.

`python/gridworld.py` **defines correct.** Where it has a quirk, the Mojo port reproduces the quirk.

## ⚠️ Notes

- **One transition = one tick at `move_budget: 1`.** At `move_budget: 8` a turn is plan-then-walk, and mid-walk frames carry the remaining path as hidden state. The world model models the **primitive-step variant** of HIDE_SEEK.
- **Self-contained.** `python/gridworld.py` is vendored so this project requires no external path dependencies.
