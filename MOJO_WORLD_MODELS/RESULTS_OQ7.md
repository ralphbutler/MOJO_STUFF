# 🔬 RESULTS — Open Question 7: what does training in the dream cost?

The experiment M8 was missing. Evolve the same controller two ways — against the **real**
simulator and against the **learned dream** — then score both champions in the **real** sim.
The gap prices the world model.

**Controls.** Both arms share one featuriser, one seeker, one optimiser (sep-CMA-ES, 598
params), identical true spawn states, identical seeds, identical budget (50 generations ×
48 population × 32 maps). The *only* difference is where the next state comes from.
Held-out fitness is always scored in the real sim, so a dream-exploiting controller has
nowhere to hide.

```bash
./build/evolve 50 48 32 1 real   data/hist_real50.txt
./build/evolve 50 48 32 1 dream  data/hist_dream.txt
```

---

## 🏁 The answer: the dream transfers nothing

| policy | fitness in the REAL sim |
|---|---:|
| random parameters | 0.177 |
| **static (never moves)** | **0.237** |
| **evolved in the DREAM** | **0.241** |
| **evolved in the REAL sim** | **0.393** |

A controller evolved inside the dream is **statistically indistinguishable from standing
still**. It captured **2.4%** of the gain the real-sim arm achieved (+0.004 against +0.156
over the static baseline).

Meanwhile, inside the dream, that same controller looked like it was learning beautifully:

| generation | dream fitness | real held-out |
|---:|---:|---:|
| 0 | 0.301 | 0.237 |
| 12 | 0.434 | 0.204 |
| 24 | 0.493 | 0.296 |
| 36 | 0.568 | 0.264 |
| 48 | **0.570** | **0.243** |

**Dream fitness nearly doubled. Reality never moved.** This is textbook model exploitation:
the optimiser is not learning to hide, it is learning to find the inputs on which the
dynamics model is wrong.

---

## 🎭 A good evaluator is not a good training signal

The most useful thing here is that the dream is *not* broadly inaccurate. Scoring a **fixed**
controller, it agrees with reality closely:

| controller | scored in dream | scored in reality | error |
|---|---:|---:|---:|
| real-trained champion (150 gen / 128 maps) | 0.598 | 0.630 | 5% |
| real-trained champion (50 gen / 32 maps) | 0.373 | 0.393 | 5% |

Within 5% both times. So a 93%-per-step, median-49-step model is a *perfectly serviceable
evaluator*.

It collapses the moment you **optimise against it**. Evaluating a fixed policy samples the
distribution the model was trained on. Optimising deliberately searches for where the model
disagrees with reality, because that is exactly where free reward appears. The 0.57 is real
— the dream really does believe that controller stays hidden. It is just wrong.

Corroborating detail: the dream-trained champion scores 0.570 on its **training** maps inside
the dream but only **0.253** on held-out maps inside the dream. It did not even generalise
*within the dream* — the exploits it found were map-specific quirks of the model, not
strategy.

The real-sim arm had the *same* 32 maps and generalised fine (0.393 held-out). So this is not
a map-count artefact. It is specific to training against a learned model.

---

## 🐌 And it was 98× slower

| arm | wall clock | turns |
|---|---:|---:|
| real sim | **2.4 s** | 7.84 M |
| dream | **232.5 s** | 7.84 M |

The learned model is **98× slower than the simulator it replaces** — a 5.3 M-parameter MLP
forward pass per step against a handful of integer comparisons, made worse because M2 already
made the sim nearly free.

So for this environment the dream loses on **both** axes: slower to run, and worthless as a
training signal. That is a real result, not a failure to try.

---

## 🧭 What this does and does not say about world models

**It does not refute Ha.** The premise of a world model is that you *cannot* query the real
environment — it is slow, expensive, dangerous, or nonexistent. This project's environment is
a 12×12 gridworld that Mojo simulates 17 M turns per second, exactly and for free. Replacing
it with a learned approximation was always going to be a bad trade *here*; the gridworld is a
stand-in used to measure the machinery, not a domain where world models pay.

**What it does show, concretely:**

1. **Per-step accuracy is the wrong headline.** 93.0% sounded strong at M4 and median-49-step
   fidelity sounded decent at M5. Neither predicted zero transfer. The number that mattered
   was one nobody had measured: how the model behaves *under adversarial optimisation*.
2. **Fidelity metrics measured on the training distribution do not bound control performance.**
   Every M5 metric was collected by replaying *recorded* action sequences. A controller
   chooses its own, and immediately walks off that distribution.
3. **The failure is silent.** Nothing in the dream arm looked broken. Fitness rose smoothly
   for 50 generations. Only the real-sim held-out column, evaluated every generation, showed
   it flat. Without that column this would have been written up as a success.

Point 3 is the same lesson as three earlier ones in this project: M2's benchmark bug (caught
by an ordering violation), M5's turn bug (caught only by ground truth, while
cross-implementation agreement read 100%), and M8's 8-map overfit (caught only by the
held-out column). **Always keep one measurement the optimisation cannot touch.**

---

## 🔭 If someone wants to make the dream work

In rough order of expected value:

- **Train on-policy.** Alternate evolution with data collection from the current controller,
  so the model is retrained where the optimiser is actually looking. This is the standard fix
  and it directly targets the failure above.
- **Ensemble disagreement as a penalty.** Train several dynamics models; where they disagree,
  the optimiser is probably in exploit territory. Penalise it in the fitness.
- **Shorten the horizon.** Dreams stay exact for a median of 49 steps; evolving on 20-step
  rollouts with real-sim restarts would stay inside the faithful region.
- **Raise per-step accuracy.** Necessary but, on this evidence, nowhere near sufficient — the
  dream was already accurate enough to be a 5%-error evaluator and still transferred nothing.
