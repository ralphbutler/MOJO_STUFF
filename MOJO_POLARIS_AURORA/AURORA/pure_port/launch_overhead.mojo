# launch_overhead.mojo — G6: measure the HOST cost of a DeviceContext launch on PVC.
#
# This is the half of the performance question that matters to people who never write
# a kernel. Through G5 we measured, on the same tile:
#
#     02 vecadd   0.0403 ms/pass through DeviceContext   vs  0.0063 through our own
#                                                            hand-written Level Zero harness
#     03c matmul  11.54 ms/pass                          vs  9.26
#
# The gap is not kernel time -- it is the same kernel either way. `enqueue_function`
# builds a fresh DeviceFunction per call, so AsyncRT_DeviceContext_loadFunction runs
# once per LAUNCH, and its first act is to build a cache key holding the entire SPIR-V
# module (86 KB for the coarse matmul) and hash it. A program doing many small launches
# pays that every time. 04b does 15 launches per epoch; real scientific codes do more.
#
# METHOD: one trivial kernel, one fixed buffer, and the GRID varied. Host cost per launch
# is constant in the grid; device time is not. The grid=1 row is therefore the host-cost
# floor, and the difference between rows is real GPU work. Run with MOJO_LZ_PROFILE=1 to
# get the runtime's own breakdown (key build / lookup / argument encoding) underneath it.
#
#   qsub launch_overhead.pbs

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 1 << 20                 # buffer stays fixed; only the grid changes
comptime BLOCK = 256
comptime LAUNCHES = 2000             # per grid size
comptime layout = row_major[N]()


def bump_kernel(
    buf: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    size: Int32,
):
    # Deliberately trivial: one add. Anything this kernel costs on the device is
    # noise next to what we are trying to measure on the host.
    comptime assert buf.flat_rank == 1
    var tid = global_idx.x
    if tid < Int(size):
        buf[tid] = buf[tid] + 1.0


def time_grid(ctx: DeviceContext, buf: TileTensor[dtype, type_of(layout), MutAnyOrigin],
              blocks: Int, label: String) raises:
    # Warmup: the first launch carries the IGC compile.
    ctx.enqueue_function[bump_kernel](
        buf, Int32(N), grid_dim=blocks, block_dim=BLOCK
    )
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(LAUNCHES):
        ctx.enqueue_function[bump_kernel](
            buf, Int32(N), grid_dim=blocks, block_dim=BLOCK
        )
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var per_launch_us = Float64(t1 - t0) / Float64(LAUNCHES) / 1.0e3
    var threads = blocks * BLOCK
    print("  ", label, ": ", blocks, " blocks (", threads, " threads) -> ",
          per_launch_us, " us/launch", sep="")


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()
    var buf_dev = ctx.enqueue_create_buffer[dtype](N)
    buf_dev.enqueue_fill(0.0)
    var buf = TileTensor(buf_dev, layout)

    print("--- G6 launch overhead: one trivial kernel, ", LAUNCHES, " launches per row ---", sep="")
    print("  buffer: ", N, " float32; block_dim ", BLOCK, sep="")
    print()

    # grid=1 is the host-cost floor: one workgroup of 256 threads doing one add.
    time_grid(ctx, buf, 1, "floor    ")
    time_grid(ctx, buf, 16, "small    ")
    time_grid(ctx, buf, 256, "medium   ")
    time_grid(ctx, buf, ceildiv(N, BLOCK), "full     ")

    print()
    print("  Reference points on this tile, same 1M-element vecadd:")
    print("    DeviceContext (G5 run 2)      : 40.3 us/pass")
    print("    our Level Zero harness (G5 s3):  6.3 us/pass")
    print("  The floor row is what a program pays per launch before any GPU work.")
    print("  Set MOJO_LZ_PROFILE=1 for the runtime's own per-launch breakdown.")

    # Keep the result observable so nothing above can be optimized away.
    with buf_dev.map_to_host() as host:
        var h = TileTensor(host, layout)
        print("  buf[0] after all launches: ", h[0], sep="")
