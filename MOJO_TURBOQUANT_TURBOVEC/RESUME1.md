# 🚀 TURBOQUANT — Resume Notes (Session 1)

**Goal:** Reimplement Google's TurboQuant (data-oblivious vector quantization) — Ryan Codrai's Rust `turbovec` — in **100% Mojo**, and run apples-to-apples comparisons. Purpose: a colleague talk showcasing Mojo as the up-and-coming way to do this work.

**Status at pause: Phase A COMPLETE — clean win on all three axes.** ✅

---

## 🏆 Headline result (reproduced, stable across runs)

DBpedia OpenAI-1536 · 100k base / 1k queries · **4-bit = 768 B/vec** (equal memory to FAISS)

| Axis | TurboQuant (Mojo, FastScan) | FAISS IndexPQ | Winner |
|---|---|---|---|
| recall@1 | **0.960** | 0.955 | Mojo (+0.5 pt) |
| recall@2 | 0.996 | 0.997 | tie |
| recall@4+ | 1.000 | 1.000 | tie |
| search speed | **0.72 ms/query** | 3.66 ms/query | **Mojo 5.1× faster** |
| memory | 768 B + 4 (scale) | 768 B | tie |
| index build | ~10.5 s, **no training** | 16.2 s | Mojo |

Matches Ryan's claim ("4-bit beats FAISS by 0.2–1.9 pts"). Independent numpy reference (`compare/reference_tq.py`) gives 0.967 — confirms algorithm correctness. This is **without TQ+** yet.

---

## 🧰 Environment / tooling

- **One env:** `~/VENVS/BASE` — has `mojo 1.0.0b2`, numpy, datasets, faiss-cpu, scipy.
- Mojo binary: `~/VENVS/BASE/bin/mojo`. Python: `~/VENVS/BASE/bin/python`.
- Machine: M4 Max, 16 cores, NEON SIMD width W=4 (f32).
- Mojo runs emit a harmless `Failed to initialize Crashpad` line — filter with `grep -v Crashpad`.

## ▶️ Reproduce the result (from repo root `/Users/rbutler/Desktop/TURBOQUANT`)

```bash
# 1. data already fetched to data/ (base.f32, queries.f32, true_top1.i32, *.npy, meta.json)
#    to refetch: ~/VENVS/BASE/bin/python compare/fetch_data.py
# 2. FAISS baseline
~/VENVS/BASE/bin/python compare/faiss_baseline.py
# 3. Mojo TurboQuant (FastScan) — build + run
cd turboquant_mojo && ~/VENVS/BASE/bin/mojo build run_dbpedia_fast.mojo -o /tmp/run_fast
/tmp/run_fast
cd .. && ~/VENVS/BASE/bin/python compare/eval_mojo.py   # scores mojo output
~/VENVS/BASE/bin/python compare/report.py               # prints head-to-head table
```

## 🗂️ File map

- `turboquant_mojo/` — the 100% Mojo implementation:
  - `codebook.mojo` — Gaussian-limit Lloyd–Max quantizer (validated vs textbook values). Codebook = std-normal levels × 1/√d.
  - `rotation.mojo` — SplitMix64 PRNG + Householder QR → random orthogonal Q (validated QᵀQ=I). Has standalone `main()`.
  - `turboquant.mojo` — **reference** List-based encode+search (single-thread, validated on synthetic + real subset).
  - `run_dbpedia.mojo` — scalar SIMD-dot search (correct, 11.5 ms/q). Keep as baseline.
  - `run_dbpedia_fast.mojo` — **FastScan** search (0.72 ms/q). ← the winner.
  - `test_synth.mojo`, `test_real.mojo` — validation drivers.
- `compare/` — Python (comparison ONLY, not the implementation):
  - `fetch_data.py` (streams dataset → raw .bin), `faiss_baseline.py`, `eval_mojo.py`, `report.py`, `reference_tq.py` (numpy oracle).
- `data/` — base.f32 (614MB), queries.f32, true_top1.i32, base.npy, queries.npy, meta.json, results/*.json.
- `ryan_turbovec/` — cloned reference Rust repo (read for spec).

## 🧠 Algorithm (as implemented)

encode: normalize → rotate by Q → per-coord Lloyd–Max quantize (code = #boundaries exceeded) → bit-pack (4-bit: 2 codes/byte, hi=dim2g, lo=dim2g+1) → store `scale = ‖v‖ / <u_rot, x_hat>` (RaBitQ-style; makes self-score exact).
search: rotate query → per-coord LUT `lut[d][c]=qrot[d]·centroid[c]` → score = `<qrot, x_hat> · scale` → top-k.
FastScan: repack codes to 16-lane blocked layout; quantize LUT to uint8 (per-coord min subtract, one shared scale, per-query bias — closed-form since each sub-table is qrot[d]·centroids); score 16 vectors/instruction via `SIMD._dynamic_shuffle` (NEON tbl). Metric: **recall@1-in-top-k** (true NN within top-k), same as FAISS side.

## ⚠️ THREE Mojo-1.0 gotchas that caused the whole debugging saga (talk gold)

1. **`fn` is gone** — Mojo 1.0 unified on `def` (strict, compiled). Also `alias`→`comptime`, stdlib under `std.` (`std.math`, `std.algorithm`, `std.memory`, `std.sys`, `std.time`). No implicit List copy (`.copy()` / `^` move required). Tuple-of-List return fails → use a struct or `mut` out-params.
2. **alloc/free INSIDE parallel workers races** → nondeterministic corruption. Fix: pre-allocate one disjoint scratch slice per work-item; workers only index into it (never alloc).
3. **THE KILLER — `rebind` to `MutUntrackedOrigin` severs lifetime tracking.** The compiler then frees the backing `List` early and reuses its memory (silent NaN/garbage). Fix: keep every such List alive to end of `main()` (reference it at the end, e.g. `print("...", lst[0])`). This one bug produced random recall for a long time.

## ✅ Verified

FAISS baseline, numpy oracle, cross-run determinism (0.960 / 0.714 ms both runs). Codebook vs textbook Lloyd–Max. Rotation orthogonality across dims 8–1536.

---

## 📋 Next steps (agreed order: discuss each briefly BEFORE doing)

**C — vs Ryan's actual Rust `turbovec` (RECOMMENDED NEXT).** Build his crate (`cargo`/`maturin`) into BASE, run our same harness (same data/metric) against his index. The true Mojo-vs-Rust slide (recall parity + speed). Risk: build/dependency friction — watch token cost. His pkg: `turbovec-python/` (maturin). Rust 1.90 present.

**D — Package results** into a short writeup + recall/speed plot for the talk. Low effort. Could be an Artifact.

**B — Add TQ+ calibration.** Per-coord (shift,scale) from empirical 5/95% quantiles → canonical Gaussian quantiles (sort per coord across 100k, once at build); query applies inverse + per-query bias. Reproduces Ryan's exact recall-edge feature. Lower payoff here (we already beat FAISS at 4-bit/1536); do for fidelity/completeness. See `ryan_turbovec/turbovec/src/encode.rs` for the spec.

**Stretch ideas mentioned:** Mojo-GPU scan (neither turbovec nor faiss-cpu uses the Mac GPU); GloVe d=200 (needs exact-Beta codebook, not Gaussian approx — low-dim regime); d=3072.

## 🔓 Open decisions for next session
- Confirm C approach (build his Rust) vs skip to D packaging.
- Whether to also compare at 2-bit (Ryan's other operating point) — would need 2-bit path (code supports it via BITS, but FastScan LUT/packing assumed 4-bit nibbles; 2-bit = 4 codes/byte needs layout tweak).
