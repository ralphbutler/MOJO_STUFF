# PVC geometry sweep for the curriculum's coarse matmul (G6).
#
# This is MOJO_CURRICULUM/03c_matmul_coarse.mojo with its five comptime geometry
# constants lifted to `-D` defines. The kernel body is otherwise unchanged, so a
# run with the default defines is the curriculum kernel and any difference in the
# numbers is the geometry, not a rewrite.
#
# WHY: on a PVC tile the curriculum geometry (BM=BN=128, BK=8, TM=TN=8) makes IGC
# report `compiled SIMD32 allocated 128 regs and spilled around 247`. Each thread
# holds TM*TN=64 accumulators plus TM+TN staging lanes; at SIMD32 that exceeds the
# EU register budget and spills. The measured consequence is that the "optimized"
# kernels run BACKWARDS on PVC -- naive 2,449 > tiled 2,177 > coarse 1,903 GFLOP/s
# -- against 8,966 for the same coarse kernel on an A100.
#
# Set a geometry with set_matmul_geometry.sh (it rewrites the block below in place):
#   bash set_matmul_geometry.sh BM=64 BN=64 BK=8 TM=4 TN=4
#   mojo run --target-accelerator intel-pvc matmul_tunable.mojo
#
# Cross this with the runtime's IGC flags (MOJO_LZ_BUILD_FLAGS), which control the
# other half of the register story -- 128 vs 256 GRF per thread.
#
# Constraints are comptime asserts: an invalid combination fails to compile rather
# than producing wrong numbers.

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major, stack_allocation

comptime dtype = DType.float32
# --- SWEEP GEOMETRY (rewritten in place by set_matmul_geometry.sh; keep as literals) ---
# These are literals, NOT get_defined_int[...] defines, on purpose. A `-D` define
# stays an unfolded expression inside the mangled kernel name, which doubled the
# entry-point name from 4,891 to 9,747 characters. The generated SPIR-V was
# instruction-for-instruction identical either way, but a name that long is an
# untested risk against IGC and Level Zero (the longest we have run is 3,889).
# With literals the baseline geometry emits a kernel byte-identical to the
# curriculum's 03c, which is also the control the sweep needs.
comptime N = 2048                              # matrix is N x N
comptime BM = 128                              # block output tile rows
comptime BN = 128                              # block output tile cols
comptime BK = 8                                # K-slab depth staged in shared memory
comptime TM = 8                                # per-thread micro-tile rows
comptime TN = 8                                # per-thread micro-tile cols
comptime ITERS = 50
# --- END SWEEP GEOMETRY ---

comptime NUM_THREADS = (BM * BN) // (TM * TN)
comptime TCOLS = BN // TN
comptime STRIDE_A = NUM_THREADS // BK
comptime STRIDE_B = NUM_THREADS // BN

comptime layout = row_major[N, N]()
comptime as_layout = row_major[BM, BK]()
comptime bs_layout = row_major[BK, BN]()


def matmul_pvc(
    A: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    B: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    C: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2

    var tid = thread_idx.x
    var c_row = block_idx.y * BM
    var c_col = block_idx.x * BN

    var thread_row = tid // TCOLS
    var thread_col = tid % TCOLS

    var inner_row_a = tid // BK
    var inner_col_a = tid % BK
    var inner_row_b = tid // BN
    var inner_col_b = tid % BN

    var sa = stack_allocation[dtype, address_space = AddressSpace.SHARED](as_layout)
    var sb = stack_allocation[dtype, address_space = AddressSpace.SHARED](bs_layout)
    comptime assert sa.flat_rank == 2 and sb.flat_rank == 2

    # Register tiles. A SIMD vector (not an Array) so the value stays register
    # resident -- see the note in 03c_matmul_coarse.mojo.
    var acc = SIMD[dtype, TM * TN](0)
    var reg_m = SIMD[dtype, TM](0)
    var reg_n = SIMD[dtype, TN](0)

    for k0 in range(0, N, BK):
        comptime for off in range(0, BM, STRIDE_A):
            sa[inner_row_a + off, inner_col_a] = rebind[sa.ElementType](
                A[c_row + inner_row_a + off, k0 + inner_col_a]
            )
        comptime for off in range(0, BK, STRIDE_B):
            sb[inner_row_b + off, inner_col_b] = rebind[sb.ElementType](
                B[k0 + inner_row_b + off, c_col + inner_col_b]
            )
        barrier()

        comptime for dot in range(BK):
            comptime for i in range(TM):
                reg_m[i] = rebind[Scalar[dtype]](sa[thread_row * TM + i, dot])
            comptime for j in range(TN):
                reg_n[j] = rebind[Scalar[dtype]](sb[dot, thread_col * TN + j])
            comptime for i in range(TM):
                comptime for j in range(TN):
                    acc[i * TN + j] += reg_m[i] * reg_n[j]
        barrier()

    comptime for i in range(TM):
        comptime for j in range(TN):
            C[c_row + thread_row * TM + i, c_col + thread_col * TN + j] = rebind[
                C.ElementType
            ](acc[i * TN + j])


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    # Geometry must tile the matrix and the cooperative loads must cover the slabs
    # exactly -- otherwise the kernel silently reads or writes the wrong cells.
    comptime assert N % BM == 0 and N % BN == 0 and N % BK == 0, "N must divide the block tiles"
    comptime assert BM % TM == 0 and BN % TN == 0, "block tile must divide by the micro-tile"
    comptime assert STRIDE_A > 0 and STRIDE_B > 0, "too few threads for the staging loads"
    comptime assert BM % STRIDE_A == 0, "A staging pass must tile BM"
    comptime assert BK % STRIDE_B == 0, "B staging pass must tile BK"
    comptime assert NUM_THREADS <= 1024, "PVC workgroup limit is 1024 threads"

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

    ctx.enqueue_function[matmul_pvc](a, b, c, grid_dim=grid, block_dim=NUM_THREADS)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[matmul_pvc](a, b, c, grid_dim=grid, block_dim=NUM_THREADS)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var avg_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6
    var flops = 2.0 * Float64(N) * Float64(N) * Float64(N)
    var gflops = flops / (avg_ms / 1.0e3) / 1.0e9

    # FULL check, not sampled: A=1 and B=2 so every C cell must be 2*N. A geometry
    # bug shows up as untouched cells, which a sampled check can miss.
    var expected = Float32(2.0 * Float64(N))
    var wrong = 0
    var zeros = 0
    with c_buf.map_to_host() as host:
        var hc = TileTensor(host, layout)
        for i in range(N):
            for j in range(N):
                var v = rebind[Scalar[dtype]](hc[i, j])
                if v != expected:
                    wrong += 1
                    if v == 0.0:
                        zeros += 1

    print("RESULT geometry BM=", BM, " BN=", BN, " BK=", BK, " TM=", TM, " TN=", TN, sep="")
    print("  threads/block :", NUM_THREADS, " acc regs/thread:", TM * TN)
    print("  N             :", N)
    print("  check         :", wrong, "/", N * N, "wrong (", zeros, "exact zeros )")
    print("  avg time      :", avg_ms, "ms")
    print("  perf          :", gflops, "GFLOP/s")
