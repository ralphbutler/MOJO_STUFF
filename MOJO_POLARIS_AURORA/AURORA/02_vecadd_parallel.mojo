# 02_vecadd_parallel.mojo — L3: SIMD + parallelize, memory-bandwidth scaling
#
# Extends 01_vecadd_safe.mojo: the same List-owned SIMD vector-add, but the pass is
# split across worker threads with `parallelize`, swept over a range of worker counts.
# Vector-add is memory-bandwidth bound, so this measures how far shared-memory scaling
# goes on one Aurora node before it saturates HBM — the core L3 question.
#
# Bytes moved per pass = 2 loads + 1 store over N float32 = 12*N bytes.
# GB/s = 12*N / seconds / 1e9.

from std.sys.info import simd_width_of, num_physical_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize

comptime dtype = DType.float32
comptime N = 64_000_000               # 256 MB/array, 768 MB total — far exceeds cache -> hits HBM
comptime W = simd_width_of[dtype]()   # SIMD lanes/register (16 on Aurora AVX-512)
comptime ITERS = 30                   # average several passes per worker count


def main() raises:
    var cores = num_physical_cores()
    print("02_vecadd_parallel — L3 SIMD + parallelize bandwidth scaling")
    print("  SIMD width      :", W, "floats/instruction")
    print("  N               :", N, "(", 12 * N // (1024 * 1024), "MB moved/pass )")
    print("  physical cores  :", cores)
    print("  ITERS/measure   :", ITERS)
    print()

    # Three owned buffers (List frees them at end of scope).
    var a = List[Scalar[dtype]](capacity=N)
    var b = List[Scalar[dtype]](capacity=N)
    var c = List[Scalar[dtype]](capacity=N)
    for i in range(N):
        a.append(Float32(i) * 0.5)
        b.append(Float32(i) * -1.5 + 2.0)
        c.append(0)

    # Localized escape hatch: raw pointers for SIMD load/store; Lists still own the memory.
    var pa = a.unsafe_ptr()
    var pb = b.unsafe_ptr()
    var pc = c.unsafe_ptr()

    # Worker-count sweep (clamped to physical cores).
    var counts = [1, 2, 4, 8, 16, 32, 64, 102]

    print("  workers |   ms/pass |     GB/s | speedup")
    print("  --------+-----------+----------+--------")

    var base_s = 0.0
    for idx in range(len(counts)):
        var p = counts[idx]
        if p > cores:
            continue
        var chunk = (N + p - 1) // p     # contiguous elements per worker

        # One work item per worker; each processes a disjoint contiguous chunk with SIMD.
        @parameter
        def worker(w: Int):
            var start = w * chunk
            var end = start + chunk
            if end > N:
                end = N
            var i = start
            while i + W <= end:
                pc.store(i, pa.load[width=W](i) + pb.load[width=W](i))
                i += W
            while i < end:               # scalar tail of this chunk
                pc.store(i, pa.load(i) + pb.load(i))
                i += 1

        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            parallelize[worker](p, p)
        var t1 = perf_counter_ns()

        var s = Float64(t1 - t0) / Float64(ITERS) / 1.0e9
        if idx == 0:
            base_s = s
        var ms = s * 1.0e3
        var gbps = 12.0 * Float64(N) / s / 1.0e9
        var speedup = base_s / s
        print("  ", p, "|", ms, "|", gbps, "|", speedup)

    # --- correctness: the parallel result must match a plain scalar computation ---
    var mismatches = 0
    for j in range(N):
        if c[j] != a[j] + b[j]:
            mismatches += 1
    print()
    print("  c[0], c[1]      :", c[0], c[1])
    print("  mismatches      :", mismatches, "/", N)
    print("  RESULT          :", "PASS" if mismatches == 0 else "FAIL")
