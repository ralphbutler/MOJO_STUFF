# 📦 RESULTS — M3, the transition dataset

**Machine:** Apple M4 Max, 128 GB, macOS 26.5.1 · Mojo 1.0.0b2 · CPython 3.12.9
**Dataset:** `data/train` — 100,000 episodes × 100 turns, grid 12
**Generated in 1.16 s.** 10.1 M state vectors, 525 MB states + 40 MB actions.

Reproduce:
```bash
./build/gen 12 100 1 data/train 60000 20000 20000   # 60% random / 20% fort / 20% build-biased
python python/dataset.py data/train                 # coverage gate
python python/gen_gate.py --episodes 60 --turns 100 # correctness gate
```

---

## 🔐 How self-generated data stays trustworthy

M1 and M2 kept the RNG in Python and shipped specs across the boundary. That does not
scale to 10⁵ episodes, so Mojo now draws maps *and* actions itself — using the **same
32-bit LCG** as `MOJO_CURRICULUM/04_train_mlp.mojo`: `s = (1103515245·s + 12345) mod 2³¹`.

`python/gen_gate.py` reimplements every draw in the same call order, runs the **vendored**
Python sim, and compares state vectors element-by-element. **60/60 episodes identical
across all three arms.** The M1 digest gate still passes independently.

One bug caught by inspection rather than test: per-episode seeds are `base + idx·7919`,
which passes 2³¹ at ~271k episodes, after which `1103515245·s` overflows Int64 — silently,
and only at scale. Both implementations now reduce the seed mod 2³¹ at construction.

---

## 📊 Coverage

| | | |
|---|---:|---:|
| Hider cells visited | 97 / 97 free | **100%** |
| Distinct full state vectors | 9,780,355 / 10,100,000 | 96.8% unique |
| Transitions that change state | 9,937,008 | 99.4% |
| Hider seen by seeker | 1,897,991 | 18.8% |
| Hider carrying an object | 100,512 | **1.0%** |

Action coverage is uniform-by-design and all 13 fire: moves 12.1% each, `grab` 10.0%,
place-and-lock 7.5% each direction, `lock` 5.0%, `wait`/`drop`/`unlock` 2.1% each.

### Fort-shaped states

| | frames | share |
|---|---:|---:|
| ≥1 box locked | 6,063,238 | 60.0% |
| ≥2 boxes locked | 3,027,061 | 30.0% |
| 3 boxes locked | 1,284,548 | 12.7% |
| **Hider fully sealed** | **1,358,640** | **13.5%** |

---

## 🎯 The finding that justified arm B

Sealed states, broken down by which arm produced them:

| arm | sealed frames | share of arm |
|---|---:|---:|
| random policy (60k eps) | **0** | 0.0% |
| pre-locked fort (20k eps) | 1,358,640 | 67.3% |
| build-biased policy (20k eps) | **0** | 0.0% |

**Neither policy ever sealed a fort — not once in 80,000 episodes.** Building a seal needs
grab → carry to a specific cell → place-and-lock → repeat → move inside, and random action
selection essentially never produces that sequence, even with `grab`/`lock` weighted up 2×.

So arm C **failed at its stated purpose** (generating fort-building *trajectories*), and
without arm B the dataset would contain zero fort states. Every one of the 1.36 M sealed
frames comes from spawning the fort directly.

### And a gap that only showed up once measured

Arm B spawns the seal already complete, so the dataset had sealed states but **no transition
*into* one** — the lock that completes a fort was entirely unobserved. Fixed by handing the
hider the next box at spawn half the time when the seal is one box short, so a single
`lock_[NSEW]` completes it. Now **1,075 seal completions across 1,017 episodes**, up from
zero.

Whether that matters is arguable: the dynamics model learns *mechanics*, and sealing is an
emergent property of box positions rather than a distinct mechanic — place-and-lock itself
is covered 7.5% per direction. The completion transitions are cheap insurance for **B2**,
where the LLM would need the dream to render fort completion.

---

## ⚠️ Known thin spot — deliberately not tuned yet

**Carrying is 1.0% of frames** (100,512), the thinnest slice of the dataset. `grab` is
issued 10% of the time but usually fails: the vendored `_adjacent_obj` quirk requires an
object within Manhattan distance 1, and one rarely is.

100k examples is probably enough, but this is the first place to look if M4's held-out
accuracy is weak on carry transitions. **Not pre-emptively fixed on purpose** — two earlier
guesses about where cost and coverage would land (BFS being cheap, LOS being a hotspot) were
both wrong, and both were settled by measurement in minutes. M4's accuracy curve decides
this one too.

---

## 🐌 The bottleneck moved

Generation ran at **219% CPU**, not the ~1200% M2 achieved. From M2's parallel figure
(0.013 µs/turn), simulating 10 M turns is ≈130 ms of the 1,155 ms total. The other ~89% is
spawn, int16 packing, and file I/O — all single-threaded.

This is not a problem to fix (1.16 s for the whole dataset), but it is worth stating plainly:
**the 300× sim speedup has made simulation a minority of data-generation cost.** If the
dataset ever needs to be 100× larger, parallelize serialization, not the sim.

---

## 🗃️ Format

Self-describing via `data/train.meta.json`. `python/dataset.py::load(prefix)` returns
memmapped arrays, so opening 525 MB is free.

- `states.bin` — `int16`, shape `[episodes, turns+1, 26]`
- `actions.bin` — `int16`, shape `[episodes, turns, 2]`
- Layout: `turn, phase`, then per agent `(x, y, face_x, face_y, carry, seen_by)`,
  then per object `(x, y, locked)`.

**Walls are not encoded.** `_spawn` adds only a border ring plus a fixed 3-cell centre stub,
with no RNG, so the layout is byte-identical for every seed at a given grid size. The model
therefore learns dynamics for this one topology; generalising across wall layouts is out of
scope for v1 and would need a wall plane in the state.
