# stencil.mojo — G6: the benchmark that looks like the code scientists actually write.
#
# WHY THIS EXISTS. Everything we had measured was vector add, matrix multiply, or a tiny
# MLP — trivial or GEMM-shaped. Quoting "36.6% of oneMKL" invites the fatal reply: nobody
# writes their own SGEMM, they call the library, so being 3x off it says nothing about
# the code I write. A 5-point Jacobi stencil is the opposite case: memory-bound, no vendor
# library to call, and the shape of a large share of real HPC kernels (diffusion, PDE
# solvers, image filters, anything with a halo).
#
# ITS OWN YARDSTICK. Rather than compare against a vendor number, the program measures a
# pure streaming copy on the same tile in the same run. Copy is the achievable-bandwidth
# ceiling for a kernel that touches each element once; the stencil is reported as a
# percentage of it. Both use the same 8-bytes-per-point accounting, so the ratio is
# meaningful and needs no outside claim.
#
# EXACT CORRECTNESS, NOT A TOLERANCE. The field is initialised to f(i,j) = 1 + i/2 + j/4,
# which is linear and therefore a fixed point of the 5-point average:
#     0.25*(f(i-1,j) + f(i+1,j) + f(i,j-1) + f(i,j+1)) = f(i,j)   exactly.
# All values are multiples of 0.25 below 2^23, so float32 represents every intermediate
# exactly and the answer after any number of iterations is the initial field, bit for bit.
# A constant field would also be a fixed point but would hide indexing bugs; a linear one
# does not -- read the wrong neighbour and the value changes.
#
# PORTABILITY IS THE POINT. This file is plain Mojo with no Intel in it. It runs on an
# Apple GPU with stock Mojo, on Polaris with stock Mojo, and on Aurora through our fork.
# Run it on all three and the table is the argument.
#
#   Aurora:  qsub stencil.pbs
#   Mac:     mojo run stencil.mojo
#   Polaris: mojo run stencil.mojo     (stock Mojo, unmodified)

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major

comptime dtype = DType.float32
# --- SWEEP GEOMETRY (rewritten in place by set_matmul_geometry.sh; keep as literals) ---
comptime N = 2048                              # grid is N x N
comptime BLOCK = 16                            # BLOCK x BLOCK threads per block
comptime ITERS = 100                           # stencil sweeps, timed
# --- END SWEEP GEOMETRY ---

comptime NB = ceildiv(N, BLOCK)
comptime layout = row_major[N, N]()
# Each point moves 4 bytes in and 4 bytes out at best; the same accounting for both
# kernels makes the stencil directly comparable to the copy ceiling.
comptime BYTES_PER_POINT = 8


def f_init(i: Int, j: Int) -> Scalar[dtype]:
    # Linear, so it is an exact fixed point of the stencil. Multiples of 0.25.
    return Float32(1.0) + Float32(i) * 0.5 + Float32(j) * 0.25


def stencil_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    comptime assert a.flat_rank == 2 and b.flat_rank == 2
    var i = Int(global_idx.y)
    var j = Int(global_idx.x)
    # Interior only; the boundary ring keeps its initial value in BOTH buffers.
    if i >= 1 and i < N - 1 and j >= 1 and j < N - 1:
        b[i, j] = rebind[b.ElementType](
            (a[i - 1, j] + a[i + 1, j] + a[i, j - 1] + a[i, j + 1]) * 0.25
        )


def copy_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    # The bandwidth ceiling: touch every element once, do nothing with it.
    comptime assert a.flat_rank == 2 and b.flat_rank == 2
    var i = Int(global_idx.y)
    var j = Int(global_idx.x)
    if i < N and j < N:
        b[i, j] = rebind[b.ElementType](a[i, j])


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var buf_a = ctx.enqueue_create_buffer[dtype](N * N)
    var buf_b = ctx.enqueue_create_buffer[dtype](N * N)

    # Both buffers get the field: the stencil never writes the boundary, so whichever
    # buffer it writes into must already hold the correct ring.
    for buf in [buf_a, buf_b]:
        with buf.map_to_host() as h:
            var t = TileTensor(h, layout)
            for i in range(N):
                for j in range(N):
                    t[i, j] = rebind[t.ElementType](f_init(i, j))

    var a = TileTensor(buf_a, layout)
    var b = TileTensor(buf_b, layout)
    var grid = (NB, NB)
    var block = (BLOCK, BLOCK)

    # --- bandwidth ceiling: streaming copy ---
    ctx.enqueue_function[copy_kernel](a, b, grid_dim=grid, block_dim=block)
    ctx.synchronize()
    var c0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[copy_kernel](a, b, grid_dim=grid, block_dim=block)
    ctx.synchronize()
    var c1 = perf_counter_ns()
    var copy_ms = Float64(c1 - c0) / Float64(ITERS) / 1.0e6
    var points = Float64(N) * Float64(N)
    var copy_gbs = points * Float64(BYTES_PER_POINT) / (copy_ms / 1.0e3) / 1.0e9

    # --- the stencil: ping-pong so each sweep reads what the last one wrote ---
    ctx.enqueue_function[stencil_kernel](a, b, grid_dim=grid, block_dim=block)
    ctx.synchronize()
    var s0 = perf_counter_ns()
    for it in range(ITERS):
        if it % 2 == 0:
            ctx.enqueue_function[stencil_kernel](a, b, grid_dim=grid, block_dim=block)
        else:
            ctx.enqueue_function[stencil_kernel](b, a, grid_dim=grid, block_dim=block)
    ctx.synchronize()
    var s1 = perf_counter_ns()
    var sten_ms = Float64(s1 - s0) / Float64(ITERS) / 1.0e6
    var sten_gbs = points * Float64(BYTES_PER_POINT) / (sten_ms / 1.0e3) / 1.0e9
    # 3 adds + 1 multiply per interior point.
    var sten_gflops = points * 4.0 / (sten_ms / 1.0e3) / 1.0e9

    # --- exact check: a linear field is a fixed point, so nothing may have changed ---
    var wrong = 0
    var zeros = 0
    for buf in [buf_a, buf_b]:
        with buf.map_to_host() as h:
            var t = TileTensor(h, layout)
            for i in range(N):
                for j in range(N):
                    var v = rebind[Scalar[dtype]](t[i, j])
                    if v != f_init(i, j):
                        wrong += 1
                        if v == 0.0:
                            zeros += 1

    print("--- G6 stencil (5-point Jacobi, ", N, "x", N, ", ", ITERS, " sweeps) ---", sep="")
    print("  block        : ", BLOCK, "x", BLOCK, sep="")
    print("  check        : ", wrong, " / ", Int(2 * points), " wrong (", zeros, " exact zeros )", sep="")
    print("  copy  (ceiling): ", copy_ms, " ms  ", copy_gbs, " GB/s", sep="")
    print("  stencil        : ", sten_ms, " ms  ", sten_gbs, " GB/s  ", sten_gflops, " GFLOP/s", sep="")
    print("  stencil / copy : ", 100.0 * sten_gbs / copy_gbs, " % of achievable bandwidth", sep="")
