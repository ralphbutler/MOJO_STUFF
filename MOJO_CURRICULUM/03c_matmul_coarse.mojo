# Thread-coarsened / register-blocked matmul on the Apple GPU via Mojo.
#
# The lever is ARITHMETIC INTENSITY — math done per memory access and per barrier.
#   naive      : 1 output/thread, no barriers, cache does the reuse
#   simple tile: 1 output/thread, 2 barriers per 16-MAC K-tile  -> too little work/sync
#   THIS       : each thread computes an TM x TN (8x8) micro-tile of C, holding 64
#                partial sums in registers. Per K-slab a thread does TM*TN*BK = 512
#                MACs between barriers -> ~32x more work per barrier than simple tiling.
#
# Layout (classic 2D block-tiling GEMM):
#   - Each BLOCK computes a BM x BN (128x128) output tile of C.
#   - Threads: (BM/TM) x (BN/TN) = 16 x 16 = 256, launched as a 1D block of 256.
#   - The K dimension is walked in BK-deep (8) slabs staged in shared memory.
#
# Requires N % BM == 0 and N % BK == 0 (no ragged-edge handling — keeps it clean/fast).

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.collections import InlineArray
from layout import TileTensor, row_major, stack_allocation

comptime dtype = DType.float32
comptime N = 2048                              # matrix is N x N (must be multiple of BM and BK)
comptime BM = 128                              # block output tile rows
comptime BN = 128                              # block output tile cols
comptime BK = 8                                # K-slab depth staged in shared memory
comptime TM = 8                                # per-thread micro-tile rows
comptime TN = 8                                # per-thread micro-tile cols
comptime NUM_THREADS = (BM * BN) // (TM * TN)  # 256 threads / block
comptime TCOLS = BN // TN                       # 16 threads span one block row
comptime STRIDE_A = NUM_THREADS // BK           # rows of A loaded per pass (32)
comptime STRIDE_B = NUM_THREADS // BN           # rows of B loaded per pass (2)
comptime ITERS = 50

comptime layout = row_major[N, N]()
comptime as_layout = row_major[BM, BK]()
comptime bs_layout = row_major[BK, BN]()


def matmul_coarse(
    A: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    B: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    C: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var tid = thread_idx.x
    var c_row = block_idx.y * BM        # top-left of this block's C tile
    var c_col = block_idx.x * BN

    # Which TM x TN micro-tile within the block this thread owns.
    var thread_row = tid // TCOLS
    var thread_col = tid % TCOLS

    # Cooperative-load index mapping (A: BM x BK, B: BK x BN).
    var inner_row_a = tid // BK
    var inner_col_a = tid % BK
    var inner_row_b = tid // BN
    var inner_col_b = tid % BN

    # Shared-memory staging for one K-slab of A and B.
    var sa = stack_allocation[dtype, address_space = AddressSpace.SHARED](as_layout)
    var sb = stack_allocation[dtype, address_space = AddressSpace.SHARED](bs_layout)
    comptime assert sa.flat_rank == 2 and sb.flat_rank == 2

    # Per-thread register tiles (live in registers, indexed by comptime loops).
    var acc = InlineArray[Scalar[dtype], TM * TN](fill=0)
    var reg_m = InlineArray[Scalar[dtype], TM](fill=0)
    var reg_n = InlineArray[Scalar[dtype], TN](fill=0)

    for k0 in range(0, N, BK):
        # --- stage one BK-deep slab into shared memory ---
        comptime for off in range(0, BM, STRIDE_A):
            sa[inner_row_a + off, inner_col_a] = rebind[sa.ElementType](
                A[c_row + inner_row_a + off, k0 + inner_col_a]
            )
        comptime for off in range(0, BK, STRIDE_B):
            sb[inner_row_b + off, inner_col_b] = rebind[sb.ElementType](
                B[k0 + inner_row_b + off, c_col + inner_col_b]
            )
        barrier()

        # --- multiply the slab, accumulating into registers ---
        comptime for dot in range(BK):
            comptime for i in range(TM):
                reg_m[i] = rebind[Scalar[dtype]](sa[thread_row * TM + i, dot])
            comptime for j in range(TN):
                reg_n[j] = rebind[Scalar[dtype]](sb[dot, thread_col * TN + j])
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += reg_m[i] * reg_n[j]
        barrier()

    # --- write the TM x TN micro-tile back to C ---
    comptime for i in range(TM):
        comptime for j in range(TN):
            C[c_row + thread_row * TM + i, c_col + thread_col * TN + j] = rebind[
                C.ElementType
            ](acc[i * TN + j])


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    comptime assert N % BM == 0 and N % BN == 0 and N % BK == 0, "N must divide the block tiles"
    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N * N)
    a_buf.enqueue_fill(1.0)
    b_buf.enqueue_fill(2.0)

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    var grid = (N // BN, N // BM)

    # Warmup — first launch includes kernel compilation/allocation.
    ctx.enqueue_function[matmul_coarse](
        a, b, c, grid_dim=grid, block_dim=NUM_THREADS
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[matmul_coarse](
            a, b, c, grid_dim=grid, block_dim=NUM_THREADS
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

    print("--- Mojo (Apple GPU, COARSE / register-blocked) ---")
    print("  N       :", N)
    print("  block   :", BM, "x", BN, " micro-tile:", TM, "x", TN)
    print("  avg time:", avg_ms, "ms")
    print("  perf    :", gflops, "GFLOP/s")
