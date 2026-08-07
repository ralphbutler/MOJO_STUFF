# 📊 RESULTS — M2, Target A (the gridworld sim)

**Machine:** Apple M4 Max, 128 GB unified memory, macOS 26.5.1
**Toolchain:** Mojo 1.0.0b2, CPython 3.12.9
**Workload:** 400 episodes × 100 turns, grid 12 (40,000 turns / 40,400 frames)
**Device:** CPU on both sides. No MPS — Mojo is CPU here, so the baseline is too.
**Method:** construction and parsing outside the timed region; warmup rep discarded;
best of 5 reported. Reproduce with `python python/bench.py --episodes 400 --turns 100 --reps 5`.

**Correctness:** all 40,400 frames of trace text are byte-identical between the two
languages, checked inside the benchmark run. The parallel row's accumulator matches the
serial row's exactly.

---

## 🏁 Headline

| Configuration | Time | µs/turn | vs Python |
|---|---:|---:|---:|
| Python — simulation only | 157.91 ms | 3.948 | 1× |
| **Mojo — 1 thread** | **5.11 ms** | **0.128** | **30.9×** |
| **Mojo — 12 threads** | **0.53 ms** | **0.013** | **299.6×** |

Parallel efficiency: **9.7× of 12× ideal (81%)** across performance cores. Episodes are
fully independent — no shared state, no reduction — so this is close to the best case for
`parallelize`.

**What this buys M3:** 75 M turns/s. Generating the millions of `(s, a, s′)` transitions
the dynamics model needs stops being a scheduling concern and becomes a rounding error.

### With trace serialization

| Configuration | Time | µs/turn | vs Python |
|---|---:|---:|---:|
| Python — sim + digest text | 217.91 ms | 5.448 | 1× |
| Mojo — sim + digest text | 17.45 ms | 0.436 | **12.5×** |
| *Python as-written (vendored HIDE_SEEK, frame dicts)* | *307.91 ms* | *7.698* | *17.6× vs Mojo* |

---

## 🔍 What the numbers say

### There is no hotspot — it's diffuse interpreter tax

`cProfile` over 300 episodes × 100 turns put `emit_frame` at 14%, `resolve_turn` 12%,
`bfs_path` 11%, `occupied_cells` 8%, `_plan` 8%, and all of LOS (`seen_map` +
`line_clear` + `occludes`) at only ~4%.

This **refuted two guesses** written into `PLAN.md` before profiling:

- *"BFS is cheap at `move_budget: 1`."* It isn't — 11%, third highest. The adjacent-goal
  shortcut is real, but `bfs_path` rebuilds the entire `occupied_cells()` set on every call
  (43,320 calls in that run) and the set construction dominates the search.
- *"LOS is a co-leading hotspot."* It isn't — ~4%. `seen_map` no-ops during prep, which is
  half of every episode, and `line_clear` only runs inside vision range.

So there is no algorithmic win on offer, just a constant factor: set/dict/list allocation
and attribute lookup spread thinly across the whole sim. That is the textbook case for a
compiled language, and 30.9× single-threaded is what it's worth here.

Corollary: an "optimized Python" row wasn't built, because the profile prices it. Memoizing
`occupied_cells` per tick and skipping trace dicts removes ~22% (≈1.3×) and the remaining
~70% has no concentrated target.

### Mojo's string building is disproportionately expensive

| | core | +digest | overhead |
|---|---:|---:|---:|
| Python | 157.91 ms | 217.91 ms | **1.38×** |
| Mojo | 5.11 ms | 17.45 ms | **3.42×** |

Serialization costs Mojo 2.5× more *relative to its own compute* than it costs Python.
`self.out += String(x)` per field reallocates and allocates a temporary per conversion —
Python's `"{},{}".format(...)` amortizes far better. That is why the digest speedup (12.5×)
is less than half the core speedup (30.9×).

Not on the critical path: M3 data generation uses core mode with binary output, not digest
text. Logged as an optimization target if trace output ever becomes hot.

---

## ⚠️ A harness bug worth recording

The first run reported **33.2×** on core, with Python's core mode (137 ms) somehow *slower*
than its digest mode (109 ms) — despite core doing strictly less work.

Cause: correctness instrumentation inside the timed loop. Both sims folded every emitted
field into an FNV-1a checksum so the harness could prove they computed the same thing. In
Mojo that inlines to one multiply. In Python it was a function call per field — 27 per
frame — so the "cheaper" mode measured mostly the instrumentation, and Mojo's win was
inflated by ~33%.

Fix: the checksum came out of the hot loop entirely. Core mode now sums the LOS result into
a single integer (one add per agent per frame — enough to stop either compiler eliminating
`seen_by`, cheap enough not to distort). Correctness moved to diffing the digest text that
digest mode already produces, plus the M1 parity gate.

**Lesson: verification belongs outside the timed region.** The tell was an ordering
violation — a mode that does less work timing slower. Worth checking for explicitly rather
than trusting a favourable number.

---

## 🔗 Relation to `MOJO_CURRICULUM/RESULTS01.md`

That file measured the **big-dense-matmul** regime and found Mojo *losing*: PyTorch MPS at
13,381 GFLOP/s against ~4,300 for the best tuned Mojo GPU kernel, and a naive Mojo GPU
kernel losing outright to Apple's Accelerate BLAS on CPU.

This file measures the opposite regime — small, branchy, sequential, allocation-bound — and
finds Mojo winning by 30.9× single-threaded and 299.6× across cores. Both results are about
the same language on the same machine. Together they say the question isn't "is Mojo fast"
but "does the workload have a tuned library already sitting on it." Dense matmul does. A
hide-and-seek gridworld does not.
