# 🧠 RESULTS — M4, the dynamics model

**Machine:** Apple M4 Max, 128 GB · PyTorch 2.6.0 on **MPS** · CPython 3.12.9
**Task:** `(state, joint action) → next state`, held out by episode
**Model:** MLP 208 → 1536 → 1536 → 1536 → 180, ReLU, 5.3 M params
**Trained** 9,000 steps × batch 4,096 in **160 s** on MPS.

```bash
python python/train.py --steps 9000 --batch 4096 --hidden 1536 --layers 3
```

---

## 🎯 Design call: classification, not regression

Every field in the state is categorical — grid coordinates on a 12×12 board, facing in
{-1,0,1}, carry as an object index, locked as a flag. So the model is **24 independent
softmax heads**, one per field, trained with summed cross-entropy.

Regressing coordinates would have let the model emit 7.3 and need a rounding rule to hide
it, and would have put "off by one cell" and "off by six cells" on the same smooth loss
surface. **Exact match** — all 24 heads right — means the predicted next state *is* the
true next state, cell for cell.

`turn` and `phase` are inputs but **not** predicted: `turn' = turn+1` and
`phase' = turn' ≥ prep_turns` are deterministic, and scoring them would inflate accuracy
with free wins.

Split is by **episode**, stratified across all four generation arms. A transition-level
split would leak — consecutive frames are near-identical, so random splitting scores
memorisation as generalisation. Arms are contiguous in the file, so a plain tail split
would have held out only one arm.

---

## 📈 Headline

| Metric | Value |
|---|---:|
| **Exact match (24/24 fields)** | **93.00%** |
| Worst field — `a0.seen` (LOS) | 97.99% |
| Next worst — `a0.fx` / `a0.fy` (facing) | 99.04% |
| Positions, locks, carry heads | 99.3 – 100% |

Hard subsets:

| subset | exact match | transitions |
|---|---:|---:|
| carrying an object | **77.31%** | 8,532 |
| ≥2 boxes locked (fort-shaped) | 93.03% | 109,810 |
| hider currently seen | 88.42% | 72,779 |

---

## ⚠️ The number that actually governs M5–M7

Per-step accuracy compounds. At 93.0%:

| dream length | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| exact trajectory survives | 93% | 70% | 48% | 23% | 2.6% | 0.1% |

**A 100-turn dream is essentially guaranteed to diverge.** M7's "steps until divergence"
metric will land around **10**. For a dream worth training a controller in, per-step needs
to be ~99.9% (90% survival at 100 steps), which is two orders of magnitude of error
reduction away.

This is the single most important finding of M4, and it is a *good* thing to learn before
building M5 rather than after: it reframes M7 from "check the dream is faithful" to "find
out how far a 93% dream actually gets," and it makes the M8 stretch goal look considerably
harder than the plan assumed.

---

## ❓ PLAN.md open questions, now answered

### Q1 — MLP or GRU/MDN-RNN? → **MLP. Recurrence would not help.**

Ha used an MDN-RNN because a mixture-density head handles stochastic environments and
recurrence carries hidden state. Neither applies here: decision 7 (`move_budget: 1`) made
states Markov by construction, and the sim is deterministic given the state. There is no
hidden state for a GRU to carry.

The evidence says the remaining error is **capacity**, not architecture:

| model | params | exact match |
|---|---:|---:|
| 512 × 2 | 0.46 M | 91.85% |
| 1536 × 3 | 5.3 M | **93.00 – 93.40%** |

Width and depth are the lever, not recurrence.

### Q2 — How many transitions is enough? → **~1.5 M. We generated 6.7× too many.**

Fixed 4,000 steps, varying training episodes:

| episodes | transitions | exact match |
|---:|---:|---:|
| 300 | 30 k | 54.55% |
| 1,000 | 100 k | 86.06% |
| 4,000 | 400 k | 90.05% |
| **15,000** | **1.5 M** | **91.02%** |
| 50,000 | 5 M | 91.07% |
| 95,000 | 9.5 M | 91.04% |

Dead flat past 1.5 M. Accuracy is **not data-limited** — which is what justified spending
the next effort on capacity and on targeted coverage rather than on more episodes.

Ironic footnote: M2 made data generation 300× faster, and M4 then showed we needed 6.7×
less data than we generated. The speedup's real value is turnaround time on experiments
like the scaling curve above (six full training runs, ~5 minutes total), not raw volume.

---

## 🔁 Closing the M3 loop: arm D

M3 flagged carrying at 1.0% of frames as the thinnest coverage and **deliberately left it
untuned**, on the grounds that M4's accuracy curve should decide. It did: carry was the
worst subset by a wide margin at **64.68%** against 93.40% overall.

So a fourth generation arm was added — box parked beside the hider at spawn so `grab` can
actually succeed (the vendored `_adjacent_obj` quirk needs Manhattan distance ≤ 1, and one
rarely is).

**The first attempt failed.** Arm D reused arm C's build-biased weights, and carry coverage
went 1.0% → 1.0%. Diagnosis: locks are 52% of C's distribution, so the hider dropped the box
within ~2 turns of grabbing it. Grabbing was never the bottleneck; *holding* was. Reweighting
arm D toward moves and away from locks:

| | carry frames | carry exact match |
|---|---:|---:|
| before arm D | 100,512 (1.0%) | 64.68% |
| arm D w/ C's weights | 102,023 (1.0%) | — |
| **arm D carry-biased** | **217,330 (2.2%)** | **77.31%** |

**+12.6 pp on the targeted weakness.**

Note the headline moved 93.40% → 93.00% across this change, and that is **not** a
regression: the validation set itself got harder, with carry transitions doubling from
4,281 to 8,532 while easy random-arm frames fell. The two headline numbers are measured on
different distributions and should not be compared directly.

---

## 📦 Export for M5

`data/dynamics.weights.bin` — flat `float32`, per `Linear`: weight `[out, in]` row-major
then bias `[out]`. No parser needed on the Mojo side, same principle as the keyword-free
`.spec` files in M1. Verified: 5,319,348 params → 21,277,392 bytes, exactly matching the
shapes in the sidecar.

`data/dynamics.weights.json` — layer shapes, field table (name/column/classes/shift),
grid and prep_turns, so `mojo/dream.mojo` can reconstruct the encoding.

Numerical agreement between Mojo and PyTorch inference is **M5's** first gate, not
established here.
