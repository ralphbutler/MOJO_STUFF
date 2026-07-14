# Naive matmul on the Apple GPU via Mojo.
# One GPU thread computes one output element C[row, col].
# This is the simplest correct GPU matmul — NOT tiled/optimized — so it shows
# "real GPU kernel" without the shared-memory machinery. Tiling is a next step.
#
# N is a compile-time constant below — edit it to sweep sizes (1024, 2048, 4096).

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 2048                 # matrix is N x N; edit to compare sizes
comptime BLOCK = 16               # 16x16 = 256 threads per block
comptime NB = ceildiv(N, BLOCK)   # blocks per grid dimension
comptime ITERS = 50
comptime layout = row_major[N, N]()


def matmul_kernel(
    A: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    B: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    C: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var row = global_idx.y
    var col = global_idx.x
    if row < N and col < N:
        var acc: Scalar[dtype] = 0.0
        for k in range(N):
            var a = rebind[Scalar[dtype]](A[row, k])
            var b = rebind[Scalar[dtype]](B[k, col])
            acc += a * b
        C[row, col] = rebind[C.ElementType](acc)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N * N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    # Warmup — first launch includes kernel compilation/allocation.
    ctx.enqueue_function[matmul_kernel](
        a, b, c, grid_dim=(NB, NB), block_dim=(BLOCK, BLOCK)
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[matmul_kernel](
            a, b, c, grid_dim=(NB, NB), block_dim=(BLOCK, BLOCK)
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var avg_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6
    var flops = 2.0 * Float64(N) * Float64(N) * Float64(N)
    var gflops = flops / (avg_ms / 1.0e3) / 1.0e9

    # Sanity check: A filled with 1, B filled with 2 -> every C = sum_k 1*2 = 2*N
    with c_buf.map_to_host() as host:
        var hc = TileTensor(host, layout)
        print("C[0,0] =", hc[0, 0], " (expected", 2.0 * Float64(N), ")")

    print("--- Mojo (Apple GPU, naive kernel) ---")
    print("  N       :", N)
    print("  avg time:", avg_ms, "ms")
    print("  perf    :", gflops, "GFLOP/s")
