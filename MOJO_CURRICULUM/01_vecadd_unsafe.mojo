# 01_vecadd.mojo — SIMD on the CPU
#
# SIMD = Single Instruction, Multiple Data: one CPU instruction operates on a
# whole register of values at once, so we add W floats per step instead of one.
# Vector add (c = a + b) is the canonical first example — pure element-wise work,
# no data reuse, no cross-element dependencies — so it isolates the single idea:
# process W lanes per loop iteration.

from std.sys import simd_width_of
from std.memory import alloc, dealloc, Layout
from std.time import perf_counter_ns

comptime dtype = DType.float32
comptime N = 1_000_000
comptime W = simd_width_of[dtype]()   # SIMD lanes per register for this dtype/CPU
comptime ITERS = 100                  # repeat the compute so the timer isn't just noise


def main() raises:
    # Three heap buffers. In Mojo 1.0 the raw-memory API is two pieces:
    #   alloc(Layout[T](count=n)) -> Allocation[T]   the OWNER of the memory
    #   .unsafe_ptr()             -> Pointer[T, ...] the borrowed view you compute through
    # Keeping the Allocation alive is what keeps the pointer valid: the pointer's
    # origin is tied to it, so the compiler will not destroy the buffer underneath us.
    var a_buf = alloc(Layout[Scalar[dtype]](count=N))
    var b_buf = alloc(Layout[Scalar[dtype]](count=N))
    var c_buf = alloc(Layout[Scalar[dtype]](count=N))
    var a = a_buf.unsafe_ptr()
    var b = b_buf.unsafe_ptr()
    var c = c_buf.unsafe_ptr()

    # Asymmetric fill so any indexing mistake shows up as a wrong number.
    # Raw pointer indexing is spelled `unsafe_offset=` — no bounds check happens,
    # and the keyword is there to make that visible at every use site.
    for i in range(N):
        a[unsafe_offset=i] = Float32(i) * 0.5
        b[unsafe_offset=i] = Float32(i) * -1.5 + 2.0

    # --- the SIMD core (timed, repeated ITERS times) ---
    # Walk the array W elements at a time. unsafe_load[width=W] pulls W contiguous
    # floats into one SIMD register; `+` adds all W lanes in a single instruction;
    # unsafe_store writes them back. The scalar tail handles the leftover < W elements.
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var i = 0
        while i + W <= N:
            var va = a.unsafe_load[width=W](i)
            var vb = b.unsafe_load[width=W](i)
            c.unsafe_store(i, va + vb)
            i += W
        while i < N:
            c[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]
            i += 1
    var t1 = perf_counter_ns()
    var cpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    # --- verify against a plain scalar computation ---
    var mismatches = 0
    for j in range(N):
        if c[unsafe_offset=j] != a[unsafe_offset=j] + b[unsafe_offset=j]:
            mismatches += 1

    print("01_vecadd — CPU SIMD vector add")
    print("  SIMD width :", W, "floats/instruction")
    print("  N          :", N)
    print("  c[0], c[1] :", c[unsafe_offset=0], c[unsafe_offset=1])
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  CPU time   :", cpu_ms, "ms/pass  <-- baseline; compare with 02_vecadd_gpu")

    # Free by consuming the owners. There is no pointer .free() any more — the
    # thing that owns the memory is the thing that releases it.
    dealloc(a_buf^)
    dealloc(b_buf^)
    dealloc(c_buf^)
