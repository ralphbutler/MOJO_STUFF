# Correctness check for the register-blocked kernel (03c_matmul_coarse.mojo).
#
# The benchmark's all-ones x all-twos fill makes every C element = 2*N, so it cannot
# catch a tile-indexing bug. Here we use NON-constant, non-symmetric inputs (so row/col
# or block/tile mix-ups produce wrong numbers) and compare the GPU result against a CPU
# reference triple-loop, element by element.
#
# N is small (256) so the CPU reference is fast, but 256 = 2*BM still spans a 2x2 grid
# of blocks and 32 K-slabs, exercising cross-block and cross-slab indexing.

from std.sys import has_accelerator
from std.gpu import thread_idx, block_idx
from std.gpu.sync import barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.collections import InlineArray
from layout import TileTensor, row_major, stack_allocation

comptime dtype = DType.float32
comptime N = 256                               # multiple of BM and BK
comptime BM = 128
comptime BN = 128
comptime BK = 8
comptime TM = 8
comptime TN = 8
comptime NUM_THREADS = (BM * BN) // (TM * TN)
comptime TCOLS = BN // TN
comptime STRIDE_A = NUM_THREADS // BK
comptime STRIDE_B = NUM_THREADS // BN

comptime layout = row_major[N, N]()
comptime as_layout = row_major[BM, BK]()
comptime bs_layout = row_major[BK, BN]()


# Deterministic, non-constant, asymmetric input generators — used both to fill the
# device inputs and to compute the CPU reference (identical values on both sides).
def a_val(i: Int, j: Int) -> Scalar[dtype]:
    return Float32((i * 7 + j * 3) % 13) * 0.1 - 0.6


def b_val(i: Int, j: Int) -> Scalar[dtype]:
    return Float32((i * 5 + j * 11) % 17) * 0.1 - 0.8


# --- kernel: identical to 03c_matmul_coarse.mojo ---
def matmul_coarse(
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

    var acc = InlineArray[Scalar[dtype], TM * TN](fill=0)
    var reg_m = InlineArray[Scalar[dtype], TM](fill=0)
    var reg_n = InlineArray[Scalar[dtype], TN](fill=0)

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
    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N * N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N * N)

    # Fill inputs with the asymmetric generators (writes sync back to device on exit).
    with a_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 2
        for i in range(N):
            for j in range(N):
                t[i, j] = rebind[t.ElementType](a_val(i, j))
    with b_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 2
        for i in range(N):
            for j in range(N):
                t[i, j] = rebind[t.ElementType](b_val(i, j))

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    ctx.enqueue_function[matmul_coarse](
        a, b, c, grid_dim=(N // BN, N // BM), block_dim=NUM_THREADS
    )
    ctx.synchronize()

    # Compare GPU output against a CPU reference triple-loop.
    var max_abs_err: Float64 = 0.0
    var max_rel_err: Float64 = 0.0
    var mismatches = 0
    with c_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 2
        for i in range(N):
            for j in range(N):
                var expected: Scalar[dtype] = 0.0
                for k in range(N):
                    expected += a_val(i, k) * b_val(k, j)
                var got = rebind[Scalar[dtype]](t[i, j])
                var ae = abs(Float64(got) - Float64(expected))
                var re = ae / (abs(Float64(expected)) + 1.0e-12)
                if ae > max_abs_err:
                    max_abs_err = ae
                if re > max_rel_err:
                    max_rel_err = re
                if re > 1.0e-3 and ae > 1.0e-3:
                    mismatches += 1

    print("--- Correctness check: coarse kernel vs CPU reference ---")
    print("  N          :", N)
    print("  max abs err:", max_abs_err)
    print("  max rel err:", max_rel_err)
    print("  mismatches :", mismatches, "/", N * N)
    if mismatches == 0:
        print("  RESULT     : PASS")
    else:
        print("  RESULT     : FAIL")
