# 02b_vecadd_numa.mojo — L5: does NUMA-aware first-touch unlock the HBM bandwidth?
#
# L3 (02_vecadd_parallel.mojo) peaked at only ~176 GB/s — ~10x under Aurora's HBM potential.
# Hypothesis: the serial fill in L3 first-touches every page onto ONE socket, so multi-socket
# workers hit remote HBM. This version does the SAME vector-add + worker sweep, but INITIALIZES
# each chunk in parallel with the SAME decomposition used for compute, so first-touch places
# each worker's pages on its own socket's HBM. Everything else (N, ITERS, kernel) matches L3
# so the GB/s columns are directly comparable.
#
# Per worker count we re-allocate + re-first-touch so init and compute use matching chunks.

from std.sys.info import simd_width_of, num_physical_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize

comptime dtype = DType.float32
comptime N = 64_000_000               # same as L3: 256 MB/array, 768 MB total
comptime W = simd_width_of[dtype]()
comptime ITERS = 30


def main() raises:
    var cores = num_physical_cores()
    print("02b_vecadd_numa — L5 NUMA-aware first-touch bandwidth (compare to L3)")
    print("  SIMD width      :", W, "floats/instruction")
    print("  N               :", N, "(", 12 * N // (1024 * 1024), "MB moved/pass )")
    print("  physical cores  :", cores)
    print("  ITERS/measure   :", ITERS)
    print("  init            : PARALLEL first-touch (per-worker chunk)")
    print()

    var counts = [1, 2, 4, 8, 16, 32, 64, 102]

    print("  workers |   ms/pass |     GB/s | speedup")
    print("  --------+-----------+----------+--------")

    var base_s = 0.0
    var total_mm = 0
    var c0 = Float32(0)
    var c1 = Float32(0)
    for idx in range(len(counts)):
        var p = counts[idx]
        if p > cores:
            continue
        var chunk = (N + p - 1) // p

        # Owned raw buffers for this measurement (freed at end of iteration).
        var pa = alloc[Scalar[dtype]](N)
        var pb = alloc[Scalar[dtype]](N)
        var pc = alloc[Scalar[dtype]](N)

        # --- PARALLEL first-touch init: worker w writes only its own chunk ---
        @parameter
        def init(w: Int):
            var start = w * chunk
            var end = start + chunk
            if end > N:
                end = N
            for i in range(start, end):
                pa.store(i, Float32(i) * 0.5)
                pb.store(i, Float32(i) * -1.5 + 2.0)
                pc.store(i, Float32(0))
        parallelize[init](p, p)

        # --- timed compute: same chunk decomposition as init ---
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
            while i < end:
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

        # correctness for this iteration (via pointers), then free
        for j in range(N):
            if pc.load(j) != pa.load(j) + pb.load(j):
                total_mm += 1
        c0 = pc.load(0)
        c1 = pc.load(1)
        pa.free()
        pb.free()
        pc.free()

    print()
    print("  c[0], c[1]      :", c0, c1)
    print("  total mismatches:", total_mm)
    print("  RESULT          :", "PASS" if total_mm == 0 else "FAIL")
