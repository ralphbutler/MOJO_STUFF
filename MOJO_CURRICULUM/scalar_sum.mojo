# scalar_sum.mojo — reduction on the CPU: scalar -> SIMD -> parallel -> compensated
#
# Companion to MOJO_SCALAR_SUM.md. Sums a List[Float32] six ways, from the naive
# scalar loop to a SIMD + multithreaded + Kahan-compensated version, and checks
# every version against a Float64 reference so a fast-but-wrong answer can't hide.
#
# The lesson is two-dimensional — both SPEED and ACCURACY move:
#   1 scalar     — one add at a time; slowest AND least accurate (f32 loses crumbs)
#   2 hand SIMD  — W lanes/instruction; ~W x faster, error partly cancels
#   3 vectorize  — same, but algorithm.vectorize owns the loop + tail
#   4 par+vec    — rows split across cores; ~cores x on top of SIMD
#   5 vec Kahan  — compensated summation; error ~independent of n (near-exact)
#   6 par+Kahan  — Kahan at three levels (lanes, workers, combine): fast AND exact
#
# On 4M mixed-magnitude floats the naive f32 sum is ~0.9% low; Kahan nails it.
# W = simd_width_of[f32] adapts to the machine (Mac NEON 4, AVX-512 16).

from std.sys import simd_width_of, num_physical_cores
from std.algorithm import vectorize, parallelize
from std.math import align_down, min
from std.collections import List
from std.time import perf_counter_ns

comptime dtype = DType.float32
comptime W = simd_width_of[dtype]()


# --- 1: scalar baseline. One element at a time; the reference *shape*. ---
def scalar_sum_1(data: List[Float32]) -> Float32:
    var total: Float32 = 0
    for v in data:
        total += v
    return total


# --- 2: hand-rolled SIMD. Vector of partial sums, W floats per step, then fold. ---
def scalar_sum_2(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var n = len(data)
    var acc = SIMD[dtype, W](0)
    var vec_end = align_down(n, W)
    var i = 0
    while i < vec_end:
        acc += p.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:                              # scalar tail (< W leftover elements)
        total += p[i]
        i += 1
    return total


# --- 3: algorithm.vectorize owns the loop + remainder. Runtime-arg closure ---
# takes an explicit capture list {mut acc, read p}; comptime-if keeps the hot
# path a pure vector add (no per-chunk reduction).
def scalar_sum_3(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var acc = SIMD[dtype, W](0)

    def accumulate[w: Int](i: Int) {mut acc, read p}:
        comptime if w == W:
            acc += p.load[width=W](i)
        else:
            acc[0] += p.load[width=w](i).reduce_add()

    vectorize[W](len(data), accumulate)
    return acc.reduce_add()


# --- 4: parallelize + vectorize. Each worker sums its chunk into a PRIVATE ---
# accumulator (no shared state, no locks); partials combined serially.
def scalar_sum_4(data: List[Float32]) -> Float32:
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
        var acc = SIMD[dtype, W](0)

        def accumulate[w: Int](j: Int) {mut acc, read p, read start}:
            comptime if w == W:
                acc += p.load[width=W](start + j)
            else:
                acc[0] += p.load[width=w](start + j).reduce_add()

        vectorize[W](count, accumulate)
        pp[wk] = acc.reduce_add()

    parallelize[work](num_workers, num_workers)
    var total: Float32 = 0
    for k in range(num_workers):
        total += pp[k]
    return total


# --- shared Kahan step: carry the lost low-order bits in `c`, feed them back. ---
# Float32 == SIMD[f32, 1], so this one helper serves both vector and scalar sites.
@always_inline
def kahan_add[w: Int](mut s: SIMD[dtype, w], mut c: SIMD[dtype, w], x: SIMD[dtype, w]):
    var y = x - c
    var t = s + y
    c = (t - s) - y                          # exactly what fell off this add
    s = t


# --- 5: vectorized Kahan. Per-lane running sum + compensation. ---
def scalar_sum_5(data: List[Float32]) -> Float32:
    var p = data.unsafe_ptr()
    var total = SIMD[dtype, W](0)
    var comp = SIMD[dtype, W](0)

    def accumulate[w: Int](i: Int) {mut total, mut comp, read p}:
        comptime if w == W:
            kahan_add(total, comp, p.load[width=W](i))
        else:
            var s = total[0]
            var c = comp[0]
            for k in range(w):
                kahan_add(s, c, p[i + k])
            total[0] = s
            comp[0] = c

    vectorize[W](len(data), accumulate)
    return total.reduce_add()


# --- 6: parallelize + Kahan at three levels (per-lane, lane-combine, worker-combine). ---
def scalar_sum_6(data: List[Float32]) -> Float32:
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
        var total = SIMD[dtype, W](0)
        var comp = SIMD[dtype, W](0)

        def accumulate[w: Int](j: Int) {mut total, mut comp, read p, read start}:
            comptime if w == W:
                kahan_add(total, comp, p.load[width=W](start + j))
            else:
                var s = total[0]
                var c = comp[0]
                for k in range(w):
                    kahan_add(s, c, p[start + j + k])
                total[0] = s
                comp[0] = c

        vectorize[W](count, accumulate)
        var s: Float32 = 0                    # (2) Kahan-combine this worker's lanes
        var c: Float32 = 0
        for lane in range(W):
            kahan_add(s, c, total[lane])
        pp[wk] = s

    parallelize[work](num_workers, num_workers)
    var s: Float32 = 0                        # (3) Kahan-combine the per-worker partials
    var c: Float32 = 0
    for k in range(num_workers):
        kahan_add(s, c, pp[k])
    return s


# --- threshold-gate dispatchers: vectorize for small n, parallelize for large. ---
comptime PAR_THRESHOLD = 1 << 16             # ~65k elems — tune from the crossover

def scalar_sum_fast(data: List[Float32]) -> Float32:
    if len(data) < PAR_THRESHOLD:
        return scalar_sum_3(data)
    return scalar_sum_4(data)

def scalar_sum_accurate(data: List[Float32]) -> Float32:
    if len(data) < PAR_THRESHOLD:
        return scalar_sum_5(data)
    return scalar_sum_6(data)


# Deterministic, mixed-magnitude fill so f32 crumb-loss is visible and any
# indexing mistake produces a wrong number.
def make_data(n: Int) -> List[Float32]:
    var d = List[Float32](length=n, fill=0)
    var p = d.unsafe_ptr()
    for i in range(n):
        p[i] = Float32((i % 97) + 1) * 0.5
    return d^


# Independent high-precision reference: the same sum accumulated in Float64.
def reference_f64(data: List[Float32]) -> Float64:
    var acc: Float64 = 0
    for v in data:
        acc += Float64(v)
    return acc


def main() raises:
    var n = 4_000_003                        # not a multiple of W -> exercises tails
    var data = make_data(n)
    var truth = reference_f64(data)

    print("scalar_sum — reduction ladder")
    print("  cores :", num_physical_cores(), "  W :", W, "  n :", n)
    print("  f64 reference (truth):", truth)
    print("  ---- version ---- | ---- sum ---- | rel.err vs truth ----")

    var results = [
        scalar_sum_1(data), scalar_sum_2(data), scalar_sum_3(data),
        scalar_sum_4(data), scalar_sum_5(data), scalar_sum_6(data),
    ]
    var names = ["1 scalar    ", "2 hand SIMD ", "3 vectorize ",
                 "4 par+vec   ", "5 vec Kahan ", "6 par+Kahan "]

    # Fast (uncompensated) versions may drift ~1%; Kahan (5,6) must be near-exact.
    var fast_tol = 2.0e-2
    var kahan_tol = 1.0e-5
    var passed = True
    for k in range(6):
        var rel = abs(Float64(results[k]) - truth) / truth
        var tol = kahan_tol if (k == 4 or k == 5) else fast_tol
        var ok = rel <= tol
        if not ok:
            passed = False
        print("  ", names[k], ":", results[k], " rel.err", rel, "" if ok else "  <-- FAIL")

    # Dispatchers must match their underlying version.
    if scalar_sum_fast(data) != scalar_sum_4(data):
        passed = False
    if scalar_sum_accurate(data) != scalar_sum_6(data):
        passed = False

    print("  RESULT :", "PASS" if passed else "FAIL")

    # --- light single-pass timing (relative; use std.benchmark for rigor) ---
    print("\n  timing (ms, single pass):")
    var t0 = perf_counter_ns(); var a = scalar_sum_1(data); var t1 = perf_counter_ns()
    print("    v1 scalar    :", Float64(t1 - t0) / 1.0e6)
    t0 = perf_counter_ns(); var b = scalar_sum_3(data); t1 = perf_counter_ns()
    print("    v3 vectorize :", Float64(t1 - t0) / 1.0e6)
    t0 = perf_counter_ns(); var c = scalar_sum_4(data); t1 = perf_counter_ns()
    print("    v4 par+vec   :", Float64(t1 - t0) / 1.0e6)
    t0 = perf_counter_ns(); var d2 = scalar_sum_6(data); t1 = perf_counter_ns()
    print("    v6 par+Kahan :", Float64(t1 - t0) / 1.0e6)
    print("    (checksum", a + b + c + d2, ")")     # keep results live
