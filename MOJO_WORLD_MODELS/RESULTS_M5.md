# 💭 RESULTS — M5, the dream

**Machine:** Apple M4 Max, 128 GB · Mojo 1.0.0b2 · PyTorch 2.6.0 · CPython 3.12.9
**Model:** the M4 MLP — 208 → 1536 → 1536 → 1536 → 180, 5.3 M params
**Setup:** closed loop. The dream gets a true start state and the true action
sequence, then runs free — every predicted state becomes the next input.

```bash
python python/dream_gate.py --episodes 200 --steps 100
```

---

## ✅ Gate 1 — Mojo matches PyTorch exactly

**100.0000% full-state agreement** across 20,000 dream steps; 100% per-field.

Better than expected: float32 summation order differs between Mojo's SIMD dot product and
PyTorch's BLAS, so near-tied class logits could have flipped an argmax. None did. The
argmax gaps in this model are evidently wide enough to absorb the drift.

---

## 🎯 Gate 2 — how far the dream actually gets

| | steps |
|---|---:|
| mean survived | **40.88** |
| median | **49** |
| p10 / p90 | 5 / 77 |
| reached the full 100 | **6.0%** |

Survival curve:

| steps | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still exact | 100% | 97% | 95.5% | 94% | **82%** | 71% | 44% | 6% |

### The naive compounding estimate was badly pessimistic

M4 predicted this from per-step accuracy alone: 0.93^10 ≈ 48% survival at 10 steps, median
around 10. **Measured: 82% at 10 steps, median 49.** Nearly 5× better.

The estimate assumed errors are independent. They are not. The state space has large easy
regions — a hider sealed in a fort with nothing moving, a seeker frozen through prep — where
the model is essentially never wrong, and episodes that enter them survive a long time.
Difficulty is concentrated, not spread.

Worth recording as a reasoning error, not just a number: per-step accuracy is a *lower bound
generator* for trajectory length, and a loose one. The honest move is to measure the rollout
rather than extrapolate from single-step metrics.

Closed-loop per-step exact match is **51.2%** against 93.0% open-loop, which is the
compounding showing up the other way: once a dream leaves the data distribution it stays off
it, and never recovers.

---

## ⚡ Gate 3 — speed, and the crossover that defines this project

CPU both sides (protocol rule 1). "Batch" = concurrent dreams; Mojo parallelises across
them, PyTorch fuses them into one GEMM.

| batch | PyTorch µs/step | Mojo µs/step | ratio |
|---:|---:|---:|---:|
| 1 | 387.65 | 267.80 | **1.4×** |
| 2 | 733.88 | 194.16 | **3.8×** |
| 8 | 155.93 | 39.14 | **4.0×** |
| 32 | 27.35 | 30.81 | 0.9× |
| 128 | 10.56 | 26.01 | 0.4× |
| 400 | 7.88 | 25.41 | 0.3× |

**Crossover at batch ≈ 32.** Mojo wins up to ~16 concurrent dreams, by up to 4×. PyTorch
wins beyond that, by up to 3×.

This is the thesis, measured. Mojo wins where per-call dispatch dominates and there is not
enough work to amortise a BLAS call; BLAS wins as soon as there is. Mojo's cost flattens at
~25 µs/step once all 12 performance cores are busy, while PyTorch keeps improving because
larger GEMMs are more efficient — so the gap widens with batch, and it will not close.

It also reproduces the shape of `MOJO_CURRICULUM/04b`'s CPU/GPU crossover on entirely
different hardware and a different axis. Same lesson: the question is never "is X fast," it
is "is there enough work here for the tuned path to pay for itself."

### Fairness note

The first Mojo kernel was a naive scalar dot product and ran at 398 µs/step — 35× slower
than PyTorch. Reporting that against tuned BLAS would have been the strawman warned about in
`PLAN.md`, only pointed the other way. SIMD-widening the dot product (16 lanes) took it to
25.9 µs/step, a **15× improvement**, with agreement still exactly 100%. The table above uses
the SIMD version. Further tiling and register blocking would narrow the large-batch gap but
cannot change its direction.

---

## 🐛 The bug that made the first run read 0.00%

The first closed-loop run reported **zero** steps survived on every episode — not degraded,
zero — while Mojo and PyTorch agreed with each other perfectly.

Cause: the sim's `emit_frame()` runs **before** `self.turn += 1`, so frame *k* carries turn
*k−1*, and frames 0 and 1 **both** carry turn 0. The rollout derived the next turn as
`s[0] + 1`, which is right at every step except the first. One column wrong at step 0, and
since turn feeds back in as an input, every subsequent step was off-distribution forever.

Both implementations shared the bug, so gate 1 passed at 100% while gate 2 read 0% — a
useful reminder that **cross-implementation agreement proves consistency, not correctness**.
Only the comparison against ground truth caught it. Fixed by passing the step index in
explicitly rather than deriving it; both `model.py::decode` and `dream.mojo::step` now carry
the gotcha in a comment.

M4's 93.0% is unaffected — `turn` was always an input, never a predicted target.

---

## 🔭 What this means for M6–M8

- **M6** is largely answered by the table above; what remains is choosing which batch regime
  to headline. Both, with the crossover, is the honest answer.
- **M7** can be closed early: the divergence measurement *is* M7's metric. What is left is
  rendering a dream and its ground truth side by side in the existing three.js viewer, which
  is presentation rather than measurement.
- **M8 needs rethinking.** CMA-ES evaluates hundreds of candidate controllers per
  generation — squarely the large-batch regime where PyTorch wins by 3×. The plan assumed
  CMA-ES was Mojo's best-fit workload; at the dream level it is the opposite. Mojo's 300×
  still stands for CMA-ES against the *real* sim (M2), so the interesting version of M8 may
  be "evolve in the real Mojo sim" rather than "evolve in the dream."

Also unchanged: at 6% survival over 100 steps, a controller trained in this dream would be
optimising against a fiction for most of an episode. Raising per-step accuracy is a
prerequisite for M8 in either venue.
