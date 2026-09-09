# 03e_matmul_cpu.mojo — the same matmul, on the CPU: naive -> SIMD -> parallel
#
# 03a–03d optimized C = A @ B on the GPU. This is the CPU counterpart, for the case
# where there is no GPU backend at all — e.g. Aurora's Intel Max GPUs, which Mojo
# can't target today (has_accelerator() == False there). The levers change from
# "shared memory + registers" to the CPU's two: wide SIMD registers and many cores.
#
# Three stages of the SAME C = A @ B, each timed in GFLOP/s and checked against the
# naive result so a vectorizing/threading bug can't hide behind a fast wrong answer:
#   1. naive   — textbook i,j,k triple loop, one scalar at a time (the 03a of the CPU)
#   2. simd    — i,k,j order; the inner j-loop moves W floats per instruction
#   3. parallel— stage 2, but rows are split across cores with parallelize()
#
# W = simd_width_of[f32] adapts to the machine: 4 on Mac NEON, 16 on AVX-512 (Aurora).
# The GFLOP/s jump from naive -> simd is the SIMD width; simd -> parallel is the core
# count. On a fat HBM node (many cores + GPU-class bandwidth) the parallel stage flies.

from std.sys import simd_width_of, num_physical_cores
from std.time import perf_counter_ns
from std.memory import alloc, dealloc, Layout
from max.algorithm import parallelize

comptime dtype = DType.float32
comptime N = 1024                     # square; a multiple of W keeps the SIMD tail empty
comptime W = simd_width_of[dtype]()   # floats per SIMD register (Mac NEON 4, Aurora AVX-512 16)

# One pointer type for all the buffers. In Mojo 1.0 a raw pointer carries an
# ORIGIN — the compiler's record of which value owns the memory it points into.
# `alloc` hands back a tracked origin; these helpers take pointers from five
# different Allocations, so we erase the origin once with `unsafe_origin_cast`
# and keep the Allocations alive in main() until the explicit dealloc.
comptime Ptr = Pointer[Scalar[dtype], MutAnyOrigin]


# Deterministic, asymmetric fills — any row/col mix-up produces wrong numbers.
def a_val(i: Int, j: Int) -> Scalar[dtype]:
    return Float32((i * 7 + j * 3) % 13) * 0.1 - 0.6


def b_val(i: Int, j: Int) -> Scalar[dtype]:
    return Float32((i * 5 + j * 11) % 17) * 0.1 - 0.8


def zero(c: Ptr):
    for idx in range(N * N):
        c[unsafe_offset=idx] = 0


# --- stage 1: naive i,j,k. One dot product per output, accumulated in a scalar. ---
def matmul_naive(a: Ptr, b: Ptr, c: Ptr):
    for i in range(N):
        for j in range(N):
            var acc: Scalar[dtype] = 0
            for k in range(N):
                acc += a[unsafe_offset=i * N + k] * b[unsafe_offset=k * N + j]
            c[unsafe_offset=i * N + j] = acc


# --- stage 2: i,k,j. Broadcast one A element, stream a B row and a C row as SIMD. ---
# Reordering k above j lets the innermost loop walk contiguous memory (B[k, :] and
# C[i, :]) so each iteration is one wide fused multiply-add. C must start at zero.
def matmul_simd(a: Ptr, b: Ptr, c: Ptr):
    zero(c)
    for i in range(N):
        var row_c = i * N
        for k in range(N):
            var aik = SIMD[dtype, W](a[unsafe_offset=i * N + k])   # splat scalar across all W lanes
            var row_b = k * N
            var j = 0
            while j + W <= N:
                var cv = c.unsafe_load[width=W](row_c + j)
                var bv = b.unsafe_load[width=W](row_b + j)
                c.unsafe_store(row_c + j, cv + aik * bv)     # fused multiply-add, W lanes at once
                j += W
            while j < N:                              # scalar tail (none when N % W == 0)
                c[unsafe_offset=row_c + j] += a[unsafe_offset=i * N + k] * b[unsafe_offset=row_b + j]
                j += 1


# --- stage 3: same math, but each ROW of C is an independent work item across cores. ---
# Row i writes only c[unsafe_offset=i*N : i*N+N] and reads a[unsafe_offset=i, :] + all of B, so the rows never
# collide — no locks needed. parallelize() hands the N rows to `workers` threads.
def matmul_parallel(a: Ptr, b: Ptr, c: Ptr, workers: Int):
    zero(c)

    @parameter
    def row_worker(i: Int):
        var row_c = i * N
        for k in range(N):
            var aik = SIMD[dtype, W](a[unsafe_offset=i * N + k])
            var row_b = k * N
            var j = 0
            while j + W <= N:
                var cv = c.unsafe_load[width=W](row_c + j)
                var bv = b.unsafe_load[width=W](row_b + j)
                c.unsafe_store(row_c + j, cv + aik * bv)
                j += W
            while j < N:
                c[unsafe_offset=row_c + j] += a[unsafe_offset=i * N + k] * b[unsafe_offset=row_b + j]
                j += 1

    parallelize[row_worker](N, workers)


# Time one call, return (ms, GFLOP/s). Matmul does 2*N^3 flops (a multiply + an add each).
def timed(name: String, a: Ptr, b: Ptr, c: Ptr, kind: Int, workers: Int) -> None:
    var t0 = perf_counter_ns()
    if kind == 0:
        matmul_naive(a, b, c)
    elif kind == 1:
        matmul_simd(a, b, c)
    else:
        matmul_parallel(a, b, c, workers)
    var t1 = perf_counter_ns()
    var ms = Float64(t1 - t0) / 1.0e6
    var gflops = (2.0 * Float64(N) * Float64(N) * Float64(N)) / (Float64(t1 - t0)) # flops/ns
    print("  ", name, "-> ", ms, "ms  |  ", gflops, "GFLOP/s")


# Compare a result buffer against the naive reference; return mismatch count.
def diff(reference: Ptr, got: Ptr) -> Int:
    var mismatches = 0
    for idx in range(N * N):
        var ae = abs(Float64(got[unsafe_offset=idx]) - Float64(reference[unsafe_offset=idx]))
        var re = ae / (abs(Float64(reference[unsafe_offset=idx])) + 1.0e-12)
        if re > 1.0e-4 and ae > 1.0e-4:
            mismatches += 1
    return mismatches


def main() raises:
    var workers = num_physical_cores()
    print("03e_matmul_cpu — CPU matmul ladder (naive -> SIMD -> parallel)")
    print("  N           :", N)
    print("  SIMD width W :", W, "floats/instruction")
    print("  workers      :", workers, "physical cores")
    print("  GFLOP/pass   :", (2.0 * Float64(N) * Float64(N) * Float64(N)) / 1.0e9)

    # alloc(Layout[T](count=n)) returns an Allocation[T] that OWNS the memory;
    # .unsafe_ptr() borrows a pointer into it. The Allocation must outlive the
    # pointer, which it does — nothing frees until the dealloc calls below.
    var a_buf = alloc(Layout[Scalar[dtype]](count=N * N))
    var b_buf = alloc(Layout[Scalar[dtype]](count=N * N))
    var cn_buf = alloc(Layout[Scalar[dtype]](count=N * N))
    var cs_buf = alloc(Layout[Scalar[dtype]](count=N * N))
    var cp_buf = alloc(Layout[Scalar[dtype]](count=N * N))
    var a = a_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var b = b_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var c_naive = cn_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var c_simd = cs_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var c_par = cp_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

    for i in range(N):
        for j in range(N):
            a[unsafe_offset=i * N + j] = a_val(i, j)
            b[unsafe_offset=i * N + j] = b_val(i, j)

    print("--- timings ---")
    timed("naive   ", a, b, c_naive, 0, workers)
    timed("simd    ", a, b, c_simd, 1, workers)
    timed("parallel", a, b, c_par, 2, workers)

    var bad_simd = diff(c_naive, c_simd)
    var bad_par = diff(c_naive, c_par)
    print("--- correctness (vs naive reference) ---")
    print("  simd mismatches    :", bad_simd, "/", N * N)
    print("  parallel mismatches:", bad_par, "/", N * N)
    print("  RESULT             :", "PASS" if (bad_simd == 0 and bad_par == 0) else "FAIL")

    dealloc(a_buf^)
    dealloc(b_buf^)
    dealloc(cn_buf^)
    dealloc(cs_buf^)
    dealloc(cp_buf^)
