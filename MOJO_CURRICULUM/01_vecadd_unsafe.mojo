# 01_vecadd.mojo — SIMD on the CPU
#
# SIMD = Single Instruction, Multiple Data: one CPU instruction operates on a
# whole register of values at once, so we add W floats per step instead of one.
# Vector add (c = a + b) is the canonical first example — pure element-wise work,
# no data reuse, no cross-element dependencies — so it isolates the single idea:
# process W lanes per loop iteration.

from std.sys import simd_width_of
from std.memory import alloc
from std.time import perf_counter_ns

comptime dtype = DType.float32
comptime N = 1_000_000
comptime W = simd_width_of[dtype]()   # SIMD lanes per register for this dtype/CPU
comptime ITERS = 100                  # repeat the compute so the timer isn't just noise


def main() raises:
    # Three heap buffers. alloc[T](n) -> UnsafePointer[T]; freed at the end.
    var a = alloc[Scalar[dtype]](N)
    var b = alloc[Scalar[dtype]](N)
    var c = alloc[Scalar[dtype]](N)

    # Asymmetric fill so any indexing mistake shows up as a wrong number.
    for i in range(N):
        a[i] = Float32(i) * 0.5
        b[i] = Float32(i) * -1.5 + 2.0

    # --- the SIMD core (timed, repeated ITERS times) ---
    # Walk the array W elements at a time. load[width=W] pulls W contiguous floats
    # into one SIMD register; `+` adds all W lanes in a single instruction; store
    # writes them back. The scalar tail handles the leftover < W elements.
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var i = 0
        while i + W <= N:
            var va = a.load[width=W](i)
            var vb = b.load[width=W](i)
            c.store(i, va + vb)
            i += W
        while i < N:
            c[i] = a[i] + b[i]
            i += 1
    var t1 = perf_counter_ns()
    var cpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    # --- verify against a plain scalar computation ---
    var mismatches = 0
    for j in range(N):
        if c[j] != a[j] + b[j]:
            mismatches += 1

    print("01_vecadd — CPU SIMD vector add")
    print("  SIMD width :", W, "floats/instruction")
    print("  N          :", N)
    print("  c[0], c[1] :", c[0], c[1])
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  CPU time   :", cpu_ms, "ms/pass  <-- baseline; compare with 02_vecadd_gpu")

    a.free()
    b.free()
    c.free()
