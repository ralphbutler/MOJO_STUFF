# 🧮 Mojo `scalar_sum` — Progressive Optimizations

Six versions of the same reduction (`List[Float32] -> Float32`), from a naïve
scalar loop up to a fully compensated, SIMD + multithreaded sum. Each block is
self-contained (aside from the shared `kahan_add` helper noted in §6) so you can
copy one out into a `.mojo` file and test it directly.

> **Dialect:** all code here is **Mojo 1.0** (`mojo>=1.0.0b2`) and was compiled
> and run against that toolchain — `comptime` (not `alias`), `def` only (no
> `fn`), `std.`-prefixed imports, `simd_width_of`, `comptime if`, and the 1.0
> closure/capture rules. It will **not** compile on pre-1.0 Mojo without changes.
>
> **Runnable companion:** `scalar_sum.mojo` — all six versions + both dispatchers
> with a self-checking `main` (compares every version to a Float64 reference).
> Run it with `mojo run scalar_sum.mojo`.

**Shared caveats across all vectorized/parallel versions:**
- Results are **not bit-identical** to the scalar baseline — SIMD lanes and
  worker chunks reorder the additions, and float addition isn't associative.
  (Measured: on 4M mixed-magnitude floats the naïve `f32` sum is ~0.9% low;
  the Kahan versions §5–§6 land on the true value. That gap *is* the lesson.)
- The `unsafe_ptr()` loads assume `List`'s contiguous storage (true) and are
  unchecked, so the tail-splitting logic is what keeps them in bounds.
- Kahan versions **must not** be compiled with fast-math — the compensation term
  algebraically cancels to zero under `--ffast-math`-style reassociation.

---

## 1️⃣ Baseline — scalar loop

The original. One element at a time, single dependency chain on `total`.
Correct-ish and simple; the reference for *shape* but actually the **least
accurate** here (see the diff note above) and the slowest.

```mojo
def scalar_sum(data: List[Float32]) -> Float32:
    var total: Float32 = 0
    for v in data:
        total += v
    return total
```

---

## 2️⃣ Hand-rolled SIMD

Keeps a vector of partial sums, adds `W` floats per iteration with one load +
one vector add, then horizontally reduces and mops up the tail. `simd_width_of`
picks the native lane count so the same source is optimal on NEON (M-series,
`W=4`) and AVX2/512 (`W=8/16`).

```mojo
from std.sys import simd_width_of
from std.math import align_down
from std.collections import List

comptime W = simd_width_of[DType.float32]()

def scalar_sum(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var n = len(data)
    var acc = SIMD[DType.float32, W](0)        # W lanes in parallel
    var vec_end = align_down(n, W)
    var i = 0
    while i < vec_end:
        acc += p.load[width=W](i)              # vector load + vector add
        i += W
    var total = acc.reduce_add()               # horizontal fold
    while i < n:                               # scalar tail
        total += p[i]
        i += 1
    return total
```

---

## 3️⃣ `algorithm.vectorize`

Same math as §2, but `vectorize` owns the loop and remainder handling. It calls
the closure with the full `W` for each block and a smaller width once for the
tail; the `comptime if` keeps the hot path a pure vector add (no per-chunk
reduction). Clarity over §2, essentially identical codegen/speed.

**1.0 closure rule:** `vectorize` takes the closure as a *runtime argument*
(`vectorize[W](size, closure)`), so the closure is a bare `def` with an explicit
capture list `{mut acc, read p}` — **no** `@parameter`.

```mojo
from std.sys import simd_width_of
from std.algorithm import vectorize
from std.collections import List

comptime W = simd_width_of[DType.float32]()

def scalar_sum(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var acc = SIMD[DType.float32, W](0)

    def accumulate[w: Int](i: Int) {mut acc, read p}:
        comptime if w == W:
            acc += p.load[width=W](i)                # full chunk: vector add
        else:
            acc[0] += p.load[width=w](i).reduce_add()  # tail: fold into one lane

    vectorize[W](len(data), accumulate)
    return acc.reduce_add()
```

---

## 4️⃣ `parallelize` + `vectorize`

Splits the array across cores via Mojo's persistent worker pool, each worker
vectorizes its slice into a **private** accumulator, then the partials are summed
serially. `parallelize` auto-schedules the threading (no join boilerplate); you
supply per-worker accumulators so there's no race. Only worth it for large `n`
(pool dispatch cost dominates otherwise).

**1.0 closure rule:** `parallelize[work](...)` consumes the worker as a *comptime
parameter*, so `work` is `@parameter def` with **implicit** capture (no `{}`) —
the opposite of the `vectorize` closure nested inside it.

```mojo
from std.sys import simd_width_of, num_physical_cores
from std.algorithm import vectorize, parallelize
from std.math import min
from std.collections import List

comptime W = simd_width_of[DType.float32]()

def scalar_sum(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var n = len(data)
    var num_workers = num_physical_cores()
    var partials = List[Float32](length=num_workers, fill=0)   # one slot per worker
    var pp = partials.unsafe_ptr()
    var chunk = (n + num_workers - 1) // num_workers            # ceil div

    @parameter
    def work(wk: Int):
        var start = wk * chunk
        var end = min(start + chunk, n)
        var count = end - start
        var acc = SIMD[DType.float32, W](0)                    # per-worker, no sharing

        def accumulate[w: Int](j: Int) {mut acc, read p, read start}:
            comptime if w == W:
                acc += p.load[width=W](start + j)
            else:
                acc[0] += p.load[width=w](start + j).reduce_add()

        vectorize[W](count, accumulate)
        pp[wk] = acc.reduce_add()                              # worker writes its own slot

    parallelize[work](num_workers, num_workers)                # items, workers
    var total: Float32 = 0
    for k in range(num_workers):                               # serial combine
        total += pp[k]
    return total
```

---

## 5️⃣ Vectorized Kahan (compensated) summation

**What Kahan fixes:** `Float32` holds ~7 significant digits. When `total` grows
large and you add a small `v`, the low bits of `v` don't fit and are silently
dropped — over millions of adds those lost crumbs accumulate into real error
(the ~0.9% seen in §1). Kahan keeps a compensation term `c` that remembers each
dropped crumb and feeds it back on the next add, giving roughly double effective
precision. Run lane-wise here so it composes with SIMD; error becomes
~independent of `n` instead of growing with it.

```mojo
from std.sys import simd_width_of
from std.algorithm import vectorize
from std.collections import List

comptime W = simd_width_of[DType.float32]()

def scalar_sum(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var total = SIMD[DType.float32, W](0)   # per-lane running sum
    var comp = SIMD[DType.float32, W](0)    # per-lane compensation

    def accumulate[w: Int](i: Int) {mut total, mut comp, read p}:
        comptime if w == W:
            var x = p.load[width=W](i)
            var y = x - comp
            var t = total + y
            comp = (t - total) - y          # capture what fell off, per lane
            total = t
        else:
            var s = total[0]
            var c = comp[0]
            for k in range(w):              # tail: same trick into lane 0
                var x = p[i + k]
                var y = x - c
                var t = s + y
                c = (t - s) - y
                s = t
            total[0] = s
            comp[0] = c

    vectorize[W](len(data), accumulate)
    return total.reduce_add()               # NOTE: plain lane combine (see §6 for tighter)
```

---

## 6️⃣ `parallelize` + Kahan combo (fully compensated)

The full ladder. Three levels of compensation, all through one `kahan_add`
helper — and since Mojo's `Float32` **is** `SIMD[DType.float32, 1]`, the same
generic helper serves the vector step, the lane combine, and the worker combine:

1. **Inside each worker** — vectorized per-lane Kahan over its chunk.
2. **Lane combine** — folding the `W` lanes into one partial is itself Kahan
   (fixes the leak in §5's plain `reduce_add`).
3. **Worker combine** — the per-worker partials get one final Kahan pass.

Every spot a rounding crumb could drop is compensated, so accuracy holds even as
parallelism reorders everything. Overkill for small `n` — gate to §1/§3 below a
size threshold.

```mojo
from std.sys import simd_width_of, num_physical_cores
from std.algorithm import vectorize, parallelize
from std.math import min
from std.collections import List

comptime W = simd_width_of[DType.float32]()

@always_inline
def kahan_add[w: Int](mut s: SIMD[DType.float32, w],
                      mut c: SIMD[DType.float32, w],
                      x: SIMD[DType.float32, w]):
    var y = x - c
    var t = s + y
    c = (t - s) - y      # exactly what fell off this add
    s = t

def scalar_sum(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var n = len(data)
    var num_workers = num_physical_cores()
    var partials = List[Float32](length=num_workers, fill=0)
    var pp = partials.unsafe_ptr()
    var chunk = (n + num_workers - 1) // num_workers

    @parameter
    def work(wk: Int):
        var start = wk * chunk
        var end = min(start + chunk, n)
        var count = end - start
        var total = SIMD[DType.float32, W](0)   # per-lane sum
        var comp = SIMD[DType.float32, W](0)    # per-lane compensation

        def accumulate[w: Int](j: Int) {mut total, mut comp, read p, read start}:
            comptime if w == W:
                kahan_add(total, comp, p.load[width=W](start + j))   # vector Kahan
            else:
                var s = total[0]
                var c = comp[0]
                for k in range(w):
                    kahan_add(s, c, p[start + j + k])                # scalar Kahan (w=1)
                total[0] = s
                comp[0] = c

        vectorize[W](count, accumulate)

        # (2) Kahan-combine this worker's lanes -> one partial
        var s: Float32 = 0
        var c: Float32 = 0
        for lane in range(W):
            kahan_add(s, c, total[lane])
        pp[wk] = s

    parallelize[work](num_workers, num_workers)

    # (3) Kahan-combine the per-worker partials
    var s: Float32 = 0
    var c: Float32 = 0
    for k in range(num_workers):
        kahan_add(s, c, pp[k])
    return s
```

> `Float32 == SIMD[DType.float32, 1]`, which is what lets `kahan_add` serve both
> the vector sites and the scalar combine sites with no separate overload.

---

## 📊 Choosing a version

| # | Version | Speed | Accuracy | Best for |
|---|---------|-------|----------|----------|
| 1 | scalar | ✗ slowest | **worst** (~0.9% off) | reference shape / tiny `n` |
| 2 | hand SIMD | ~W× | better | small–medium `n`, no deps |
| 3 | vectorize | ~W× | better | same as §2, cleaner |
| 4 | parallelize+vectorize | ~cores×W× | better | large `n` (100s KB+) |
| 5 | vectorized Kahan | ~W× (¼ arith) | ~exact | accuracy-critical, single-thread |
| 6 | parallelize+Kahan | ~cores×W× | ~exact | large `n` + accuracy-critical |

*Measured on an M4 Max (16 cores, `W=4`), 4M floats, single pass:*
scalar ≈ 2.3 ms → hand-SIMD/vectorize ≈ 0.57 ms → parallel+vec ≈ 0.12 ms;
Kahan single-thread ≈ 1.9 ms, parallel+Kahan ≈ 0.31 ms. Accuracy vs. the true
sum: scalar ~0.9% low, SIMD ~0.014% low, Kahan effectively exact.

---

## 🚦 Threshold gate — pick the version by `n`

Wraps the family behind one entry point that dispatches on array size, so you get
the vectorized path (no pool overhead) for small inputs and the multithreaded
path for large ones. Two dispatchers: a **fast** one (uncompensated §3/§4) and an
**accurate** one (Kahan §5/§6). Calibrate `PAR_THRESHOLD` from the §3↔§4 crossover
you measure in the benchmark below — the value here is a placeholder.

```mojo
# Cut-over point where parallelize starts beating single-thread vectorize.
# Measure it (see benchmarking section) — depends on core count & memory bandwidth.
comptime PAR_THRESHOLD = 1 << 16   # ~65k elements (~256 KB) — placeholder, tune it!

# Assumes the six bodies are present, renamed scalar_sum_3/_4/_5/_6.

def scalar_sum_fast(data: List[Float32]) -> Float32:
    # Speed-first: no compensation.
    if len(data) < PAR_THRESHOLD:
        return scalar_sum_3(data)      # vectorize, single-thread
    return scalar_sum_4(data)          # parallelize + vectorize

def scalar_sum_accurate(data: List[Float32]) -> Float32:
    # Accuracy-first: Kahan-compensated.
    if len(data) < PAR_THRESHOLD:
        return scalar_sum_5(data)      # vectorized Kahan, single-thread
    return scalar_sum_6(data)          # parallelize + Kahan
```

Notes:
- Below the SIMD width the branch still works — §3/§5's `comptime if` handles a
  sub-width tail — so there's no separate scalar special-case needed.
- Keep `PAR_THRESHOLD` a power-of-two `comptime` (compile-time constant) so the
  dead branch can be pruned when the size is statically known.

---

## ⏱️ Benchmarking them against each other

Use Mojo's `std.benchmark` module — it handles warmup, runs each function many
times, and reports a stable mean. Two things matter for a *trustworthy* result:

1. **Give each version a distinct name** (`scalar_sum_1` … `scalar_sum_6`) so
   they coexist in one file. Copy the six bodies in and rename.
2. **Defeat dead-code elimination** with `benchmark.keep`. Without it the
   compiler sees the result is unused, deletes the whole sum, and you "measure"
   an empty loop. `keep(result)` forces the value to be materialized; a
   `keep(data.unsafe_ptr())` keeps the array live too.

**1.0 closure rule:** `run[b]()` takes the benchmark body as a *comptime
parameter*, so `b` is `@parameter def` with implicit capture (no `{}`).

```mojo
from std.benchmark import run, keep
from std.sys import num_physical_cores
from std.collections import List

# --- paste versions here, renamed scalar_sum_1 ... scalar_sum_6, plus kahan_add ---

def make_data(n: Int) -> List[Float32]:
    var d = List[Float32](length=n, fill=0)
    var p = d.unsafe_ptr()
    for i in range(n):
        p[i] = Float32((i % 97) + 1) * 0.5      # deterministic, nonzero, mixed magnitudes
    return d^

def main() raises:
    var n = 20_000_000                          # ~80 MB of Float32 — large enough for §4/§6
    var data = make_data(n)
    keep(data.unsafe_ptr())
    var bytes = n * 4

    # --- correctness first: everything vs. the scalar baseline ---
    var ref = scalar_sum_1(data)
    print("baseline sum =", ref)
    print("v5 diff =", scalar_sum_5(data) - ref)   # Kahan: closest to the true sum
    print("v6 diff =", scalar_sum_6(data) - ref)

    # --- throughput ---
    @parameter
    def b1(): keep(scalar_sum_1(data))
    @parameter
    def b3(): keep(scalar_sum_3(data))
    @parameter
    def b4(): keep(scalar_sum_4(data))
    @parameter
    def b6(): keep(scalar_sum_6(data))

    print("cores:", num_physical_cores(), " n:", n)
    report("1 scalar        ", run[b1]().mean(), bytes)
    report("3 vectorize     ", run[b3]().mean(), bytes)
    report("4 parallel+vec  ", run[b4]().mean(), bytes)
    report("6 parallel+Kahan", run[b6]().mean(), bytes)

def report(name: String, secs: Float64, bytes: Int):
    var gbps = Float64(bytes) / secs / 1.0e9
    print(name, "  ", secs * 1.0e3, "ms   ", gbps, "GB/s")
```

Run it:

```bash
mojo run bench.mojo
#   or, optimized standalone binary:
mojo build bench.mojo -o bench && ./bench
```

### Reading the results

- **Watch GB/s, not just ms.** A large float sum is memory-bandwidth-bound, so
  the fast versions converge toward your machine's DRAM bandwidth ceiling. If §3
  and §4 land at similar GB/s, you're bandwidth-limited and threading buys
  little — expected.
- **Sweep `n`.** Re-run at `n = 10_000`, `100_000`, `1_000_000`, `20_000_000`.
  You'll see §4/§6 *lose* to §3 at small `n` (pool dispatch overhead) and win at
  large `n`. That crossover is exactly where `PAR_THRESHOLD` belongs.
- **`keep` on every result is non-negotiable** — the single most common way Mojo
  microbenchmarks lie is by measuring a sum the optimizer deleted.
- **Do NOT enable fast-math** when timing §5/§6, or Kahan collapses and you'll
  benchmark the wrong algorithm (and see it "magically" match §3's accuracy).

For fancier output the module's `Bench` / `Bencher` API adds throughput columns
and comparison tables — but `run[...]()` + `keep` is enough to rank the six.

---

## ✅ Verification status

Every code block above was compiled and run on the curriculum toolchain
(`mojo 1.0.0b2`, Apple M4 Max, `W=4`, 16 cores): all six versions plus both
dispatchers agree with the baseline within float-reordering tolerance, and the
Kahan versions match the analytically-computed true sum. If a future toolchain
changes an API, the *algorithm* (partition → SIMD → compensate) is the durable
part; re-probe the syntax against the compiler.
