# 🚀 TURBOQUANT — Resume Notes (Session 2)

**Read `RESUME1.md` first** (Phase A: the 100% Mojo TurboQuant reimplementation, env, algorithm, Mojo-1.0 gotchas). This file covers Session 2 only.

**Session 2 did: C (vs Ryan's Rust `turbovec`) + optimizations (b)+(c) + D (talk artifact). All COMPLETE.** ✅

---

## 🏆 Final scoreboard (apples-to-apples, 4-bit = 768 B/vec, all parallel on M4 Max 16-core)

DBpedia OpenAI-1536 · 100k base / 1k queries · recall@1-in-top-k

| | TurboQuant (Mojo) | turbovec (Rust) | FAISS IndexPQ |
|---|---|---|---|
| recall@1 | 0.959 | 0.964 | 0.955 |
| recall@2 | 0.994 | 0.996 | 0.997 |
| recall@4+ | 0.999→1.0 | 1.0 | 1.0 |
| **index build** | **0.7 s** | 0.9 s | 16.1 s |
| **search** | 0.38 ms/q | 0.15 ms/q | 3.66 ms/q |
| bytes/vec | 768 | 768 | 768 |
| training | none | none | required |

**Headline:** Mojo ties on recall, **beats Rust on build**, within 2.5× on search, ~10× faster than FAISS on search. All deltas vs Rust are ≤0.005 recall (noise on 1k queries).

---

## 🧩 C — building & running Ryan's Rust turbovec

- `maturin` installed into BASE: `~/VENVS/BASE/bin/pip install "maturin>=1.12,<2.0"`.
- Build (needs `VIRTUAL_ENV` set or it errors):
  ```bash
  cd ryan_turbovec/turbovec-python
  export VIRTUAL_ENV=~/VENVS/BASE
  ~/VENVS/BASE/bin/maturin develop --release   # ~27 s compile, no friction
  ```
- Python API: `turbovec.TurboQuantIndex(dim=1536, bit_width=4)` → `.add(base)` → `.prepare()` → `.search(queries, k)` returns `(scores, indices)` (indices int64). It's data-oblivious like ours (no train); build = add + prepare. Search parallelizes across queries via rayon (`search.rs`).
- Harness: `compare/turbovec_baseline.py` (mirrors `faiss_baseline.py` exactly) → writes `data/results/turbovec.json`.
- `compare/report.py` now prints the **three-way** table + speed/build summary.

## ⚡ Optimizations applied to `turboquant_mojo/run_dbpedia_fast.mojo`

Both are standard FastScan/GEMM techniques Rust/FAISS already use; ported to readable Mojo. Recall held (0.960→0.959, from the 7-bit LUT cap; noise).

**(b) FastScan scan kernel — search 0.71 → 0.37 ms/q (1.9×).** Root cause found by reading `ryan_turbovec/turbovec/src/search.rs:96-101,1188+`: Rust caps LUT ≤127, combines both nibble sub-tables in **uint8** (`vaddq_u8`), then **one** widening add into a **uint16** accumulator, flushing to uint32 periodically. Ours had cast each shuffle uint8→uint32 and accumulated in uint32 every byte (2-4× the SIMD register traffic). Fix: LUT scale `/255`→`/127` (clamp 127); hot loop `var s = hLUT._dynamic_shuffle(hi) + lLUT._dynamic_shuffle(lo); acc16 += s.cast[uint16]()`; branch-free chunked flush every `FLUSH_EVERY=256` groups (256×254<65535). NOTE: a per-iteration `if since_flush==...` branch REGRESSED to 1.0 ms/q — must be a branch-free chunk loop.

**(c) Rotation as a register-blocked GEMM — build 7.0 → 0.7 s (7.6× encode).** Root cause: encode rotated per-vector as `dot(Q+d*DIM, vb, DIM)` — re-streamed the whole 9.4 MB Q matrix for each of 100k vectors (memory-bound, ~34 GFLOP/s). Rust batches via faer/ndarray GEMM. Fix: 3-pass encode (norms → GEMM → quantize+pack). GEMM = MR×NR (4×4) register-blocked microkernel; `xrot = alloc(N*DIM)` (614 MB, fine). **THE UNLOCK: use `comptime for` for the MR/NR unroll** so accumulator indices are compile-time → live in registers. A runtime-index `InlineArray[SIMD]` (or `@parameter`+runtime idx) spills acc to the stack every FMA and gives NO speedup (was stuck at 5.2-5.6 s). Requires N%MR==0 & DIM%NR==0 (100000%4, 1536%4 ✓) so no edge tiles.
- Mojo-1.0 note: `@parameter for` is DEPRECATED → use `comptime for`. `InlineArray` is builtin (no import) with `InlineArray[T,size](fill=x)`.
- Query-side rotation left as-is (per-query dot; only ~10% of search, NQ=1000). Remaining 2.5× search gap = block width (Rust 32 vec/block vs our LANES=16) + proper top-k heap (ours is O(K) linear-rescan). Bounded future work.

## 📊 D — talk artifact

- **`talk/turboquant_results.html`** — self-contained (no external libs), dark/light toggle, built per the `dataviz` skill (palette validated: Mojo=blue slot1, Rust=green slot2, FAISS=magenta slot3; light-mode magenta contrast WARN mitigated by direct value labels + table view). Sections: stat tiles → speed bars → recall@k SVG line chart → full tables → optimization arc → (b)/(c) code snippets → advocacy bullets → method footnote.
- **`talk/render_dark.png`, `talk/render_light.png`** — full-page screenshots (via headless Chrome) for dropping into slides.
  ```bash
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu \
    --hide-scrollbars --window-size=920,2400 --screenshot=out.png "file://<abs path to html>"
  ```

## ▶️ Reproduce everything (repo root)

```bash
~/VENVS/BASE/bin/python compare/faiss_baseline.py
export VIRTUAL_ENV=~/VENVS/BASE   # (only needed if rebuilding turbovec)
~/VENVS/BASE/bin/python compare/turbovec_baseline.py
cd turboquant_mojo && ~/VENVS/BASE/bin/mojo build run_dbpedia_fast.mojo -o /tmp/run_fast && /tmp/run_fast
cd .. && ~/VENVS/BASE/bin/python compare/eval_mojo.py   # scores mojo output (0.959)
~/VENVS/BASE/bin/python compare/report.py               # three-way table
```
NOTE: `eval_mojo.py` overwrites `data/results/mojo.json` with recalls only (no timing). Mojo timings (build 0.7 / search 0.38) are printed by the binary; re-inject into mojo.json if you want them in `report.py`'s speed summary.

## 💾 Backups

- `BAKDEL1/` — code state after C, BEFORE (b) [baseline: search 0.71, build 7.0].
- `BAKDEL2/` — code state after (b), BEFORE (c) [search 0.37, build 7.0].

## 📋 Possible next steps

- **(2, deferred)** 5-run median timings for defensibility (single-run warm now). User doubts anyone will ask; low effort if wanted.
- **B / TQ+ calibration** — still skipped (see RESUME1). Lower payoff.
- **Search gap** — implement 32-wide block + real top-k heap to chase Rust's 0.15 ms/q. Diminishing returns.
- **2-bit operating point** — needs FastScan layout tweak (4 codes/byte). Skipped.
- Stretch (RESUME1): Mojo-GPU scan, GloVe d=200 (needs exact-Beta codebook), d=3072.

## 🔓 Open decisions for next session
- Is the talk artifact final, or want edits (add speedup annotations, a "no-BLAS" dependency-tree visual, live demo)?
- Present as HTML or export the PNGs into existing slide deck?
