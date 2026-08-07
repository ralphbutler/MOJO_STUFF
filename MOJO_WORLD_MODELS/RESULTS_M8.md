# 🧬 RESULTS — M8, evolving a controller in the real Mojo sim

**Machine:** Apple M4 Max, 128 GB · Mojo 1.0.0b2
**Optimiser:** separable CMA-ES (Ros & Hansen 2008), 598 parameters
**Controller:** linear, 45 features → 13 action logits, argmax. Ha's controller is linear too.
**Opponent:** greedy full-knowledge seeker — deterministic, so fitness carries no opponent noise.
**Fitness:** `hider_reward` — share of post-prep frames with no sightline to the hider.

```bash
./build/evolve 150 48 128 1 data/evolve_hist.txt
```

Per PLAN.md decision 9 this evolves against the **real** simulator, not the dream.
See open question 7 for what that costs conceptually.

---

## 🏁 Result

| policy | held-out fitness |
|---|---:|
| static (never moves) | 0.264 |
| random parameters | 0.248 |
| **evolved** | **0.658** |

**2.5× better than standing still**, on 128 maps disjoint from the 128 used for training.
Training best reached 0.80 against held-out 0.658 — a real but modest generalisation gap.

Throughput: **940,800 rollouts = 94.08 M simulated turns**, at 932% CPU across 12
performance cores.

Two runs of identical work took **5.3 s** and **11.1 s** — 2× run-to-run variance, from
background load and thermals rather than anything in the code. Using the slower run to stay
conservative: **8.5 M turns/s**, including the controller forward pass and the seeker each
step.

At Python's measured 3.948 µs/turn (RESULTS_M2, *simulation only*), 94.08 M turns is
**~6.2 minutes** — and that ignores running a 598-parameter policy per step in Python, which
would dominate. So ≥70× wall-clock, conservatively, and the real figure is higher.

This is where M2's speedup cashes out. A 150-generation search is a coffee-break decision
rather than an overnight job, and the overfitting at 8 maps (below) was diagnosed and fixed
by re-running with 16× more maps — a fix that costs seconds instead of an afternoon.

---

## 📐 Measured, replacing the estimate above

`python/evolve_ref.py` implements the same search in Python (NumPy for the sep-CMA-ES
algebra and the controller matvec, the vendored Python simulator for rollouts). Both sides
score two fixed controllers identically — `./build/evolve probe 1 16` and
`python evolve_ref.py --verify --maps 16` both return `zero=0.62625 det=0.56875` — so the
comparison is of equal work.

| single-threaded, 204,800 turns | time | per turn |
|---|---:|---:|
| Python | 1.316 s | 6.43 µs |
| Mojo, scalar controller | 0.209 s | 1.02 µs |
| **Mojo, SIMD controller** | **0.078 s** | **0.38 µs** |

**16.8×**, not the ≥70× estimated above. The estimate extrapolated from the *simulator's*
per-turn cost and ignored that the controller is a dense matvec — which Python gives to
NumPy. Our first Mojo version did it scalar and measured only 6.3×; vectorising recovered
2.7× with identical scores. Third time this project has caught itself racing unoptimised
Mojo against a tuned library.

Reproduce:
```bash
./build/evolve bench 1 32 64
cd python && python3 evolve_ref.py --bench --maps 32 --reps 64 --seed 1
```

## 🔍 What it actually learned — and did not

Action histogram over 128 held-out episodes, and end-of-episode state:

```
wait=0   N=3090  S=2454  E=1898  W=120
grab=0   drop=0  lock=0  unlock=0
lockN=0  lockS=439  lockE=1945  lockW=2854
boxes_locked = 11 of 384        hider_sealed = 0 of 128
```

**`grab` is never chosen. Not once.** The controller never picks up a box, therefore never
places one, and **never builds a fort in any of 128 episodes**. The 0.658 comes entirely
from *evasion* — kiting a pursuer that has full knowledge but no speed advantage, and using
occlusion that was already on the map.

The `lock*` actions firing 5,238 times are not construction. With nothing carried and
nothing adjacent, `lock` is a no-op, so the policy is using them as substitute *wait*
actions — while `wait` itself is never selected, an arbitrary tie-break in the linear layer.
The 11 locked boxes are incidental: a `lock` that happened to land next to a box.

### This is the same wall M3 hit

M3 measured that random *and* build-biased play produced **zero** seals in 80,000 episodes.
M8 now shows that 150 generations of CMA-ES also produce zero. Two very different search
procedures, same outcome.

The reason is structural, not a tuning failure. A fort requires a long, precisely ordered
sequence — grab, carry to a specific cell, place-and-lock, repeat, then move inside — where
every intermediate step scores **zero or negative** reward (carrying a box means not
running away). There is no fitness gradient to climb, and a reactive linear policy over 45
features has no state in which to hold a multi-step plan. Undirected search cannot find it
and a memoryless controller could not execute it if handed to it.

That is a genuinely informative negative result, and it is the same gap the original
HIDE_SEEK_LLM project hit from the opposite direction: an LLM *could* design a fort but only
executed it ~1 in 5 episodes. Planning is the hard part in both.

---

## ⚠️ Honest accounting of what M8 does and does not show

**Shows:** Mojo turns a controller search that Python would make painful into something
interactive — 94 M turns in seconds, with enough headroom that the fix for overfitting was
"use 16× more maps and re-run."

**Does not show:** anything about world models. Decision 9 deliberately evolved against the
real simulator, so this is not Ha's "sever the environment and train inside the model"
claim. The world-model result in this project rests entirely on **M5's fidelity
measurement** (median 49 steps before divergence), not on transfer.

The experiment that *would* close that gap is still the one in open question 7: run the same
ES harness against `dream.mojo` as a second backend, then evaluate both controllers in the
real sim. The delta would price what a 49-step-faithful dream costs. The harness was written
with that swap in mind — `rollout()` is the only function that would change.

---

## 🐛 Overfitting, caught by the held-out split

The first run used 8 training maps. Training fitness climbed 0.505 → 0.62 while held-out sat
at **0.2675 — exactly the static baseline**. The controller had memorised 8 layouts and
learned nothing transferable.

Worth noting because the training curve alone looked like progress. Only the held-out
column, evaluated every generation on maps the optimiser never saw, revealed it was flat.
Raising to 128 maps fixed it.
