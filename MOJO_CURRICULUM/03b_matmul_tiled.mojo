# Tiled matmul on the Apple GPU via Mojo — the optimization MPS uses.
#
# The naive kernel (03a_matmul_naive.mojo) re-reads every A row and B column from slow
# global memory, so it is memory-bandwidth-bound. This version cooperatively loads
# a TILE x TILE block of A and B into fast on-chip SHARED memory once, then each
# thread reuses those values TILE times before loading the next block. That turns
# the kernel from bandwidth-bound toward compute-bound — the whole point of tiling.
#
# Each block has TILE x TILE threads; each thread computes one C[row, col].
# N is a compile-time constant below — edit to sweep sizes (must keep N % TILE == 0
# for the fast path, though bounds checks make any N correct).

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major, stack_allocation

comptime dtype = DType.float32
comptime N = 2048                     # matrix is N x N; edit to compare sizes
comptime TILE = 16                    # TILE x TILE = 256 threads per block
comptime NB = ceildiv(N, TILE)        # blocks per grid dimension
comptime ITERS = 50
comptime layout = row_major[N, N]()
comptime tile_layout = row_major[TILE, TILE]()


def matmul_tiled(
    A: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    B: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    C: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var tx = thread_idx.x
    var ty = thread_idx.y
    var row = block_idx.y * TILE + ty
    var col = block_idx.x * TILE + tx

    # Per-block shared-memory scratch for one tile of A and one tile of B.
    var sa = stack_allocation[dtype, address_space = AddressSpace.SHARED](
        tile_layout
    )
    var sb = stack_allocation[dtype, address_space = AddressSpace.SHARED](
        tile_layout
    )
    comptime assert sa.flat_rank == 2 and sb.flat_rank == 2

    var acc: Scalar[dtype] = 0.0

    # March across the K dimension one TILE-wide block at a time.
    for k0 in range(0, N, TILE):
        # Cooperative load: each thread brings in one element of each tile.
        if row < N and (k0 + tx) < N:
            sa[ty, tx] = rebind[sa.ElementType](A[row, k0 + tx])
        else:
            sa[ty, tx] = 0.0
        if (k0 + ty) < N and col < N:
            sb[ty, tx] = rebind[sb.ElementType](B[k0 + ty, col])
        else:
            sb[ty, tx] = 0.0
        barrier()  # all loads must finish before anyone reads the tile

        # Multiply the two tiles from fast shared memory (inner loop unrolled).
        comptime for k in range(TILE):
            var av = rebind[Scalar[dtype]](sa[ty, k])
            var bv = rebind[Scalar[dtype]](sb[k, tx])
            acc += av * bv
        barrier()  # done reading before the next block overwrites the tile

    if row < N and col < N:
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
    ctx.enqueue_function[matmul_tiled](
        a, b, c, grid_dim=(NB, NB), block_dim=(TILE, TILE)
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[matmul_tiled](
            a, b, c, grid_dim=(NB, NB), block_dim=(TILE, TILE)
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

    print("--- Mojo (Apple GPU, TILED kernel) ---")
    print("  N       :", N)
    print("  tile    :", TILE)
    print("  avg time:", avg_ms, "ms")
    print("  perf    :", gflops, "GFLOP/s")
