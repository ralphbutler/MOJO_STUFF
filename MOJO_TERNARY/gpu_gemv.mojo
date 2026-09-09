# gpu_gemv.mojo — step 1 of the GPU port: the ternary GEMV, on the Apple GPU.
#
# Verified against the CPU `sdot` kernel on a real tensor from the GGUF before any
# of it is wired into the graph. The CPU path stays the oracle: a GPU kernel that
# is merely fast is worthless, and the only way to know it is right is to diff it
# against something already trusted.
#
# THREAD MAPPING. One block per output row, 128 threads, and each thread reads
# **16 consecutive weights with a vectorized load**. That last part is the whole
# game: the first version had each thread read a single int8, so a 32-thread warp
# fetched only 32 consecutive bytes where the GPU wants 128+, and it ran at 90 GB/s
# against 265 GB/s for the CPU. With 16-byte loads a warp fetches 512 contiguous
# bytes.
#
# 16 divides the 128-weight group size, so a thread's 16 elements always sit inside
# one group. That means the group's scale applies cleanly to a single int32 dot
# product — the accumulation is exact, as on the CPU, and only the final sum across
# threads is fp32. Hence the tolerance check below is tight.
#
# ============================================================================
# RESULT: THE GPU LOSES THIS ONE, AND THAT IS THE USEFUL FINDING.
#
# Measured on the 310MB token_embd tensor, this M4 Max:
#     CPU sdot (parallel)      265 GB/s
#     this GPU kernel          194 GB/s   (0.73x)
#     GPU pure streaming read  245 GB/s   <- the GPU's own ceiling
#
# An int8 ternary GEMV is pure streaming: every weight is read once, ~1 MAC per
# byte. Both processors are therefore bandwidth-bound, and on this chip **the GPU
# has no bandwidth advantage over the CPU** — 245 GB/s versus a CPU that already
# reaches 265 on this operation. There is no headroom to win.
#
# It is worse on small tensors: on the 12.6MB ffn_gate the GPU manages ~28-46 GB/s
# against the CPU's 63, because a kernel launch plus sync (~0.3-0.4 ms) costs more
# than the CPU's ~300us `parallelize` dispatch. Per-GEMV launches are the same trap
# in a new costume.
#
# So porting the *existing int8 format* to the GPU cannot pay. What the GPU is
# actually for is making PACKED weights affordable: unpacking 2-bit in-kernel is
# compute-heavy, which is why llama.cpp's packed CPU path reaches only 56 GB/s while
# its packed Metal path reaches 281. Fewer bytes is the only thing that raises the
# ceiling; the GPU is what supplies the compute to unpack them.
#
# Honest ladder from llama.cpp's own numbers plus ours:
#     ours now, CPU int8        12.9 tok/s (8B)
#     CPU + packed             ~25.7 tok/s  (llama.cpp measured)
#     GPU + packed            ~129   tok/s  (llama.cpp measured)
#
# Keep this file: the kernel is correct and verified, and it is the starting point
# for the packed version, where the GPU's compute finally matters.
# ============================================================================
#
# Usage: mojo run gpu_gemv.mojo [tensor-name] [gguf-path]

from gguf import GGUF, TERNARY_BLOCK_BYTES, TERNARY_GROUP, load_header
from model import F32, I8, QTensor, gemv_sdot, quantize_groups, take_q
# GPU import paths for Mojo 1.0.0, each established by compiling: thread indices
# live in `std.gpu`, but `barrier` is in `max.gpu`, `stack_allocation` in
# `std.memory`, and `AddressSpace` is in the prelude. 1.0.0b2 had `std.gpu.sync`
# and `std.gpu.memory`, which is why MOJO_CURRICULUM imports them from there.
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu import barrier
from std.memory import stack_allocation
from std.sys import argv, num_physical_cores
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

# 32 threads per row won the sweep on the 310MB tensor (GB/s): 32 -> 194, 64 -> 193,
# 128 -> 179. Fewer threads per row means more vectorized loads each and a shallower
# reduction. Even so the GPU LOSES to the CPU here — see the header note.
comptime THREADS = 32
comptime VEC = 16                       # int8 per vectorized load; must divide 128
comptime REPS = 20


def gemv_kernel(
    w: Pointer[Scalar[DType.int8], MutAnyOrigin],
    xq: Pointer[Scalar[DType.int8], MutAnyOrigin],
    wscale: Pointer[Scalar[DType.float32], MutAnyOrigin],
    ascale: Pointer[Scalar[DType.float32], MutAnyOrigin],
    y: Pointer[Scalar[DType.float32], MutAnyOrigin],
    cols: Int32,
    ngroups: Int32,
):
    var r = block_idx.x
    var t = thread_idx.x
    var ng = Int(ngroups)
    var nc = Int(cols)
    var row = r * nc

    var acc: Float32 = 0.0
    var c = t * VEC
    while c < nc:
        var g = c // TERNARY_GROUP
        var wv = w.unsafe_load[width=VEC](row + c).cast[DType.int32]()
        var xv = xq.unsafe_load[width=VEC](c).cast[DType.int32]()
        var isum = (wv * xv).reduce_add()        # exact, within one group
        acc += (
            Float32(isum)
            * wscale[unsafe_offset = r * ng + g]
            * ascale[unsafe_offset=g]
        )
        c += THREADS * VEC

    # Tree-reduce the 128 partial sums down to y[r].
    var sh = stack_allocation[
        THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    sh[unsafe_offset=t] = acc
    barrier()
    var active = THREADS
    while active > 1:
        active >>= 1
        if t < active:
            sh[unsafe_offset=t] += sh[unsafe_offset = t + active]
        barrier()
    if t == 0:
        y[unsafe_offset=r] = sh[unsafe_offset=0]


def main() raises:
    var args = argv()
    comptime SNAP = "/Users/rbutler/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-gguf/snapshots/983b5dec2ff16aab79990711ba0f828a499a7e6a/"
    var tname = String("blk.0.ffn_gate.weight")
    var path = String(SNAP) + "Ternary-Bonsai-1.7B-Q2_0.gguf"
    if len(args) > 1:
        tname = String(args[1])
    if len(args) > 2:
        path = String(args[2])

    var workers = num_physical_cores()
    var g = load_header(path.copy(), False)
    var ti = g.index_of(tname)
    if ti < 0:
        raise Error("tensor not found: " + tname)
    var cols = g.ne0[ti]
    var rows = g.ne1[ti]
    if cols % TERNARY_GROUP != 0:
        raise Error("cols must be a multiple of 128")
    if cols % (THREADS * VEC) != 0:
        raise Error(
            "cols must be a multiple of THREADS*VEC ("
            + String(THREADS * VEC) + "); got " + String(cols)
        )

    print("GPU ternary GEMV — step 1 of the GPU port")
    print("  tensor :", tname)
    print("  shape  :", rows, "rows x", cols, "cols =", rows * cols, "weights")

    var t = take_q(g, path, tname, rows, cols, workers)
    var ngroups = cols // TERNARY_GROUP

    # Activations, quantized per group of 128 exactly as the CPU path does.
    var x = List[F32](length=cols, fill=0)
    for i in range(cols):
        x[i] = F32((i * 37) % 211) * 0.0094 - 1.0
    var xq = List[I8](length=cols, fill=0)
    var ascale = List[F32](length=ngroups, fill=0)
    quantize_groups(x, cols, xq, ascale)

    # --- CPU reference -----------------------------------------------------
    var y_cpu = List[F32](length=rows, fill=0)
    gemv_sdot(t, x, xq, ascale, y_cpu, workers)
    var best_cpu = 0.0
    for rep in range(REPS):
        var c0 = perf_counter_ns()
        gemv_sdot(t, x, xq, ascale, y_cpu, workers)
        var ms = Float64(perf_counter_ns() - c0) / 1e6
        if rep == 0 or ms < best_cpu:
            best_cpu = ms

    # --- GPU ---------------------------------------------------------------
    var ctx = DeviceContext()
    print("  device :", ctx.name())
    var d_w = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    var d_xq = ctx.enqueue_create_buffer[DType.int8](cols)
    var d_ws = ctx.enqueue_create_buffer[DType.float32](rows * ngroups)
    var d_as = ctx.enqueue_create_buffer[DType.float32](ngroups)
    var d_y = ctx.enqueue_create_buffer[DType.float32](rows)

    var up0 = perf_counter_ns()
    with d_w.map_to_host() as m:
        var p = m.unsafe_ptr()
        for i in range(rows * cols):
            p[unsafe_offset=i] = t.w[i]
    with d_ws.map_to_host() as m:
        var p = m.unsafe_ptr()
        for i in range(rows * ngroups):
            p[unsafe_offset=i] = t.scales[i]
    with d_xq.map_to_host() as m:
        var p = m.unsafe_ptr()
        for i in range(cols):
            p[unsafe_offset=i] = xq[i]
    with d_as.map_to_host() as m:
        var p = m.unsafe_ptr()
        for i in range(ngroups):
            p[unsafe_offset=i] = ascale[i]
    print("  upload :", Float64(perf_counter_ns() - up0) / 1e6, "ms (one-time)")

    var best_gpu = 0.0
    for rep in range(REPS):
        var g0 = perf_counter_ns()
        ctx.enqueue_function[gemv_kernel](
            d_w.unsafe_ptr(), d_xq.unsafe_ptr(), d_ws.unsafe_ptr(),
            d_as.unsafe_ptr(), d_y.unsafe_ptr(), Int32(cols), Int32(ngroups),
            grid_dim=rows, block_dim=THREADS,
        )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - g0) / 1e6
        if rep == 0 or ms < best_gpu:
            best_gpu = ms

    var y_gpu = List[F32](length=rows, fill=0)
    with d_y.map_to_host() as m:
        for i in range(rows):
            y_gpu[i] = m[i]

    # --- correctness: GPU against the trusted CPU kernel -------------------
    var peak: F32 = 0.0
    for i in range(rows):
        var a = y_cpu[i] if y_cpu[i] >= 0.0 else -y_cpu[i]
        if a > peak:
            peak = a
    var floor = peak * 0.01
    var worst: F32 = 0.0
    var worst_abs: F32 = 0.0
    for i in range(rows):
        var d = y_gpu[i] - y_cpu[i]
        if d < 0.0:
            d = -d
        if d > worst_abs:
            worst_abs = d
        var m = y_cpu[i] if y_cpu[i] >= 0.0 else -y_cpu[i]
        if m >= floor:
            var rel = d / m
            if rel > worst:
                worst = rel
    print("")
    print("=== correctness (GPU vs CPU sdot) ===")
    print("  max rel:", worst, "  max abs:", worst_abs, "  (floor", floor, ")")
    print("  sample cpu:", y_cpu[0], y_cpu[1], y_cpu[rows - 1])
    print("  sample gpu:", y_gpu[0], y_gpu[1], y_gpu[rows - 1])

    print("")
    print("=== speed (min of", REPS, ") ===")
    var wb = Float64(rows * cols)
    print("  CPU sdot:", best_cpu, "ms  ", wb / best_cpu / 1e6, "GB/s")
    print("  GPU     :", best_gpu, "ms  ", wb / best_gpu / 1e6, "GB/s")
    print("  speedup :", best_cpu / best_gpu, "x")
    print("")
    if worst > 1e-3:
        raise Error(
            "GPU disagrees with the CPU kernel by " + String(worst)
            + " relative -- that is a bug, not fp rounding"
        )
    print("  STEP 1 KERNEL OK: matches the CPU oracle within fp tolerance.")
