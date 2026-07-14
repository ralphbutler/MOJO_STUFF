# 02_vecadd_gpu.mojo — the SAME vector add as 01, but on the GPU
#
# CPU SIMD (01): one core walks the array, adding W=4 lanes per step.
# GPU (this):    launch ~N threads at once; each thread adds exactly ONE element.
# The parallelism moves from "4 wide per instruction" to "thousands of threads".
#
# The kernel is a plain function (no CUDA decorators). `global_idx.x` gives each
# thread its own global element index. We round the launch up to whole blocks, so
# the last block overhangs N — hence the `if tid < size` guard.

from std.math import ceildiv
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime N = 1_000_000
comptime BLOCK = 256                 # threads per block
comptime ITERS = 100                 # repeat launches for a stable timing average
comptime layout = row_major[N]()     # 1D layout describing the buffers


# Same asymmetric generators as 01 — used to fill inputs AND to check the result.
def a_val(i: Int) -> Scalar[dtype]:
    return Float32(i) * 0.5

def b_val(i: Int) -> Scalar[dtype]:
    return Float32(i) * -1.5 + 2.0


# --- the GPU kernel: runs once per thread ---
def vecadd_kernel(
    a: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
    size: Int,
):
    comptime assert a.flat_rank == 1 and b.flat_rank == 1 and c.flat_rank == 1
    var tid = global_idx.x           # this thread's element (block_idx*block_dim + thread_idx)
    if tid < size:                   # last block overhangs N — guard it
        c[tid] = a[tid] + b[tid]     # one element; same layout, so no rebind needed


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)

    # Fill inputs on the host; map_to_host syncs the writes back to the device on exit.
    with a_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 1
        for i in range(N):
            t[i] = rebind[t.ElementType](a_val(i))
    with b_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 1
        for i in range(N):
            t[i] = rebind[t.ElementType](b_val(i))

    var a = TileTensor(a_buf, layout)
    var b = TileTensor(b_buf, layout)
    var c = TileTensor(c_buf, layout)

    # Launch: grid_dim blocks x block_dim threads. Monomorphic kernel → pass by name.
    var grid = ceildiv(N, BLOCK)

    # Warmup: the first launch pays one-time compile/allocation costs — exclude it.
    ctx.enqueue_function[vecadd_kernel](a, b, c, N, grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()

    # Timed: kernel execution only (NOT the host<->device copies, which would make
    # the GPU look even worse for a job this trivial).
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        ctx.enqueue_function[vecadd_kernel](a, b, c, N, grid_dim=grid, block_dim=BLOCK)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var gpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    # Verify GPU output against the CPU formula (same float32 add → expect exact match).
    var mismatches = 0
    with c_buf.map_to_host() as h:
        var t = TileTensor(h, layout)
        comptime assert t.flat_rank == 1
        for j in range(N):
            if rebind[Scalar[dtype]](t[j]) != a_val(j) + b_val(j):
                mismatches += 1

    print("02_vecadd_gpu — vector add on the GPU (Metal)")
    print("  N          :", N)
    print("  block size :", BLOCK, "threads")
    print("  grid size  :", grid, "blocks")
    print("  total thr  :", grid * BLOCK, "(overhang guarded)")
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  GPU time   :", gpu_ms, "ms/pass (kernel only)")
    print("")
    print("  NOTE: the GPU is NOT expected to beat the CPU here, and that is the")
    print("        point of the experiment. Vector add does 1 add per 2 loads —")
    print("        ~zero arithmetic intensity — so it is memory-bound on both")
    print("        backends, and the GPU also pays data-transfer costs (excluded")
    print("        above). matmul reuses each value O(N) times; that is where the")
    print("        GPU actually wins. Watch that flip in 03.")
