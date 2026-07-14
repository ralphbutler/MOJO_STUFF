# 01_vecadd_safe.mojo — SIMD on the CPU, the safe/idiomatic way
#
# Same computation as 01_vecadd_unsafe.mojo, but memory is owned by List:
#   - bounds-checked element access (a[i])
#   - automatically freed when the List goes out of scope (no .free())
# We drop to a raw pointer ONLY inside the timed SIMD kernel via .unsafe_ptr(),
# which is the one place the unsafety actually buys performance.

from std.sys import simd_width_of
from std.time import perf_counter_ns

comptime dtype = DType.float32
comptime N = 1_000_000
comptime W = simd_width_of[dtype]()   # SIMD lanes per register for this dtype/CPU
comptime ITERS = 100                  # repeat the compute so the timer isn't just noise


def main() raises:
    # Three owned buffers. No alloc, no free — List manages the heap for us.
    # capacity=N avoids reallocation as we append; length grows to N below.
    var a = List[Scalar[dtype]](capacity=N)
    var b = List[Scalar[dtype]](capacity=N)
    var c = List[Scalar[dtype]](capacity=N)

    # Asymmetric fill so any indexing mistake shows up as a wrong number.
    # append() grows each List to length N; c is pre-filled with 0 to store into.
    for i in range(N):
        a.append(Float32(i) * 0.5)
        b.append(Float32(i) * -1.5 + 2.0)
        c.append(0)

    # --- the SIMD core (timed, repeated ITERS times) ---
    # unsafe_ptr() hands us an UnsafePointer into the List's storage so we can use
    # SIMD load/store. This is the deliberate, localized escape hatch: the pointers
    # live only inside this block, and the Lists still own (and will free) the memory.
    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var pc = c.unsafe_ptr()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var i = 0
        while i + W <= N:
            var va = pa.load[width=W](i)
            var vb = pb.load[width=W](i)
            pc.store(i, va + vb)
            i += W
        while i < N:              # scalar tail for the leftover < W elements
            c[i] = a[i] + b[i]    # bounds-checked List access
            i += 1
    var t1 = perf_counter_ns()
    var cpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    # --- verify against a plain scalar computation (all bounds-checked) ---
    var mismatches = 0
    for j in range(N):
        if c[j] != a[j] + b[j]:
            mismatches += 1

    print("01_vecadd_safe — CPU SIMD vector add (List-owned)")
    print("  SIMD width :", W, "floats/instruction")
    print("  N          :", N)
    print("  c[0], c[1] :", c[0], c[1])
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  CPU time   :", cpu_ms, "ms/pass")

    # No a.free()/b.free()/c.free() — the Lists free themselves here at end of scope.
