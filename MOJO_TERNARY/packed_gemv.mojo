# packed_gemv.mojo — the smart version: weights stay PACKED in RAM.
#
# The engine so far expands 2-bit weights to int8 at load (IDEA_CURR.md Correction
# #2, taken from Litespark) and therefore streams 1 byte per weight. That advice
# holds only when the kernel is compute-bound. Ours is bandwidth-bound, measured, so
# the expansion costs 3.75x the traffic on the one resource we are short of. This
# reads 2.25 bits per weight instead of 8 and unpacks in-register.
#
# THE UNPACKING TRICK. A 16-byte load of codes covers 64 weights, 4 per byte, low
# bits first. Shifting by 0/2/4/6 and masking gives four int8x16 vectors — but each
# holds weights at STRIDE 4, not consecutively:
#
#     (codes >> 0) & 3  ->  weights 0, 4, 8, ... 60
#     (codes >> 2) & 3  ->  weights 1, 5, 9, ... 61
#     (codes >> 4) & 3  ->  weights 2, 6, ...   62
#     (codes >> 6) & 3  ->  weights 3, 7, ...   63
#
# Rather than shuffle the weights back into order on every row — which would cost
# more than it saves — the ACTIVATIONS are permuted once into that same order, which
# is O(cols) against the GEMV's O(rows*cols) and therefore free. `sdot` then consumes
# matching pairs directly.
#
# EXACTNESS. Reordering changes only the order of int32 additions, which is
# associative and exact, so this must agree with the int8 kernel BIT FOR BIT. The
# check below asserts equality, not a tolerance — anything else is a bug.
#
# Usage: mojo run packed_gemv.mojo [tensor-name] [gguf-path]

from gguf import GGUF, TERNARY_BLOCK_BYTES, TERNARY_GROUP, load_header, read_blob
from model import F32, I8, QTensor, gemv_sdot, quantize_groups, take_q
from std.gpu import block_idx, thread_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from std.memory import bitcast, stack_allocation
from std.sys import argv, llvm_intrinsic, num_physical_cores
from std.time import perf_counter_ns
from max.algorithm import parallelize

comptime I32x4 = SIMD[DType.int32, 4]
comptime I8x16 = SIMD[DType.int8, 16]
comptime U8x16 = SIMD[DType.uint8, 16]
comptime REPS = 20
comptime MIN_PARALLEL_ELEMS = 18 * 1024 * 1024
# GPU-NATIVE: many threads, each doing a little SCALAR work. See the kernel comment.
comptime GPU_THREADS = 32


# Packed weights: 2-bit codes exactly as they sit in the file, plus the group scales
# lifted out into their own fp32 array. 2 bits + 4 bytes per 128 weights = 2.25
# bits/weight, against 8 for the int8 layout.
struct PackedTensor(Movable):
    var codes: List[UInt8]        # rows * cols / 4
    var scales: List[F32]         # rows * cols / 128
    var rows: Int
    var cols: Int

    def __init__(out self, blob: List[UInt8], rows: Int, cols: Int) raises:
        if cols % TERNARY_GROUP != 0:
            raise Error("cols must be a multiple of 128")
        self.rows = rows
        self.cols = cols
        var nblocks = rows * cols // TERNARY_GROUP
        self.codes = List[UInt8](length=rows * cols // 4, fill=0)
        self.scales = List[F32](length=nblocks, fill=0)
        # Split the interleaved 34-byte blocks into contiguous codes + scales, so
        # the kernel's loads stay 16-byte aligned and uninterrupted.
        for b in range(nblocks):
            var src = b * TERNARY_BLOCK_BYTES
            var lo = UInt16(blob[src])
            var hi = UInt16(blob[src + 1])
            self.scales[b] = bitcast[DType.float16](lo | (hi << 8)).cast[
                DType.float32
            ]()
            for j in range(32):
                self.codes[b * 32 + j] = blob[src + 2 + j]

    def groups_per_row(self) -> Int:
        return self.cols // TERNARY_GROUP


# Permute activations into the order the shift-and-mask unpacking produces.
# Done once per GEMV: O(cols) against O(rows*cols).
def permute_activations(xq: List[I8], cols: Int, mut xp: List[I8]) raises:
    var k = 0
    for blk in range(cols // TERNARY_GROUP):
        var base = blk * TERNARY_GROUP
        for chunk in range(2):                 # two 16-byte code loads per block
            for sft in range(4):               # shift 0, 2, 4, 6
                for i in range(16):
                    xp[k] = xq[base + chunk * 64 + 4 * i + sft]
                    k += 1


def gemv_packed(
    t: PackedTensor,
    xperm: List[I8],
    ascale: List[F32],
    mut y: List[F32],
    workers: Int,
):
    var cp = t.codes.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var xp = xperm.unsafe_ptr()
    var ap = ascale.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var ng = t.groups_per_row()
    var cols = t.cols
    var rows = t.rows
    var code_bytes_per_row = cols // 4

    def do_rows(r0: Int, r1: Int) capturing:
        for r in range(r0, r1):
            var crow = r * code_bytes_per_row
            var acc: F32 = 0.0
            for g in range(ng):
                var cbase = crow + g * 32       # 32 code bytes per 128 weights
                var pbase = g * TERNARY_GROUP   # permuted activations, same length
                var lanes = I32x4(0)
                for chunk in range(2):
                    var cv = cp.unsafe_load[width=16](cbase + chunk * 16)
                    for sft in range(4):
                        # 4 codes per byte; (code - 1) maps {0,1,2} -> {-1,0,+1}
                        var w = (
                            (cv >> UInt8(2 * sft)) & 0x3
                        ).cast[DType.int8]() - 1
                        var xv = xp.unsafe_load[width=16](
                            pbase + chunk * 64 + sft * 16
                        )
                        lanes = llvm_intrinsic[
                            "llvm.aarch64.neon.sdot", I32x4
                        ](lanes, w, xv)
                acc += (
                    F32(lanes.reduce_add())
                    * sp[unsafe_offset = r * ng + g]
                    * ap[unsafe_offset=g]
                )
            yp[unsafe_offset=r] = acc

    if rows * cols < MIN_PARALLEL_ELEMS:
        do_rows(0, rows)
        return
    var rb = (rows + workers - 1) // workers

    def row_block(blk: Int) capturing:
        do_rows(blk * rb, min(blk * rb + rb, rows))

    parallelize[row_block]((rows + rb - 1) // rb, workers)


# ---------------------------------------------------------------------------
# The packed GEMV on the GPU. THREE SHAPES TRIED; this is the best of them, and it
# is still only par with the CPU int8 kernel. Recorded so none is retried blind:
#
#   1. wide per-thread (this one): 16-byte code loads, SIMD[int32,16] accumulator,
#      activations read from global memory in the stride-4 permuted order  74 GB/s
#   2. fully scalar: 1 code byte per load, 8 scalar MACs per thread        26 GB/s
#      -- single-byte loads waste most of each memory transaction
#   3. wide loads + activations staged in shared memory                   40 GB/s
#      -- the stride-4 access pattern forced a 16-iteration SCALAR GATHER out of
#         shared memory, which cost more than the global traffic it removed
#
# All three sit far below the 245 GB/s streaming ceiling this GPU reaches on a plain
# read, so none of them is memory-bound: the limit is per-thread ALU work and the
# awkward stride-4 unpacking pattern. Making this fast is a genuine kernel
# optimisation project, not a port -- llama.cpp's Metal ternary path reaches an
# effective 281 GB/s and is presumably carefully tuned.
#
# Ideas not yet tried, for whoever picks this up: narrower accumulators (int16
# instead of int32, halving register pressure), restructuring the packed layout so
# codes unpack to CONSECUTIVE rather than stride-4 weights (a load-time repack,
# which would make both the weight loads and the activation reads contiguous), and
# processing several rows per thread to reuse the activation vector from registers.
#
# No `sdot`: it is an ARM CPU instruction, absent on this target.
# ---------------------------------------------------------------------------
def packed_kernel(
    codes: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    xperm: Pointer[Scalar[DType.int8], MutAnyOrigin],
    wscale: Pointer[Scalar[DType.float32], MutAnyOrigin],
    ascale: Pointer[Scalar[DType.float32], MutAnyOrigin],
    y: Pointer[Scalar[DType.float32], MutAnyOrigin],
    code_bytes_per_row: Int32,
    cols: Int32,
    ng: Int32,
):
    var r = block_idx.x
    var t = thread_idx.x
    var ngi = Int(ng)
    var crow = r * Int(code_bytes_per_row)

    var acc: Float32 = 0.0
    var g = t
    while g < ngi:
        var lanes = SIMD[DType.int32, 16](0)
        for chunk in range(2):
            var cv = codes.unsafe_load[width=16](crow + (g * 2 + chunk) * 16)
            for sft in range(4):
                var w = ((cv >> UInt8(2 * sft)) & 0x3).cast[DType.int32]() - 1
                var xv = xperm.unsafe_load[width=16](
                    g * TERNARY_GROUP + chunk * 64 + sft * 16
                ).cast[DType.int32]()
                lanes += w * xv
        acc += (
            Float32(lanes.reduce_add())
            * wscale[unsafe_offset = r * ngi + g]
            * ascale[unsafe_offset=g]
        )
        g += GPU_THREADS

    var sh = stack_allocation[
        GPU_THREADS, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    sh[unsafe_offset=t] = acc
    barrier()
    var active = GPU_THREADS
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
    var nelem = rows * cols
    var ngroups = cols // TERNARY_GROUP

    print("packed ternary GEMV — weights never expanded")
    print("  tensor :", tname)
    print("  shape  :", rows, "x", cols, "=", nelem, "weights")

    # int8 reference (the current engine's layout)
    var t8 = take_q(g, path, tname, rows, cols, workers)

    # packed layout, straight from the file's blocks
    var blob = read_blob(
        path,
        g.data_start + g.offsets[ti],
        (nelem // TERNARY_GROUP) * TERNARY_BLOCK_BYTES,
    )
    var tp = PackedTensor(blob, rows, cols)

    var int8_bytes = Float64(nelem) + Float64(nelem // TERNARY_GROUP) * 4.0
    var pack_bytes = Float64(nelem) / 4.0 + Float64(nelem // TERNARY_GROUP) * 4.0
    print("  int8 layout :", int8_bytes / 1e6, "MB  (", int8_bytes * 8.0 / Float64(nelem), "bits/weight )")
    print("  packed      :", pack_bytes / 1e6, "MB  (", pack_bytes * 8.0 / Float64(nelem), "bits/weight )")
    print("  reduction   :", int8_bytes / pack_bytes, "x fewer bytes per token")

    # activations
    var x = List[F32](length=cols, fill=0)
    for i in range(cols):
        x[i] = F32((i * 37) % 211) * 0.0094 - 1.0
    var xq = List[I8](length=cols, fill=0)
    var ascale = List[F32](length=ngroups, fill=0)
    quantize_groups(x, cols, xq, ascale)
    var xperm = List[I8](length=cols, fill=0)
    permute_activations(xq, cols, xperm)

    var y8 = List[F32](length=rows, fill=0)
    var yp = List[F32](length=rows, fill=0)
    gemv_sdot(t8, x, xq, ascale, y8, workers)
    gemv_packed(tp, xperm, ascale, yp, workers)

    var bad = 0
    for i in range(rows):
        if y8[i] != yp[i]:
            bad += 1
    print("")
    print("=== correctness (must be BIT-EXACT vs the int8 kernel) ===")
    print("  mismatched rows:", bad, "of", rows)
    print("  sample int8  :", y8[0], y8[1], y8[rows - 1])
    print("  sample packed:", yp[0], yp[1], yp[rows - 1])
    if bad != 0:
        raise Error("packed kernel disagrees with the int8 kernel on " + String(bad) + " rows")

    var best8 = 0.0
    var bestp = 0.0
    for rep in range(REPS):
        var a0 = perf_counter_ns()
        gemv_sdot(t8, x, xq, ascale, y8, workers)
        var a = Float64(perf_counter_ns() - a0) / 1e6
        if rep == 0 or a < best8:
            best8 = a
        var b0 = perf_counter_ns()
        gemv_packed(tp, xperm, ascale, yp, workers)
        var b = Float64(perf_counter_ns() - b0) / 1e6
        if rep == 0 or b < bestp:
            bestp = b

    # --- GPU packed -------------------------------------------------------
    var ctx = DeviceContext()
    # (chunk count no longer needed — the GPU-native kernel indexes by weight)
    var d_codes = ctx.enqueue_create_buffer[DType.uint8](nelem // 4)
    var d_xp = ctx.enqueue_create_buffer[DType.int8](cols)
    var d_ws = ctx.enqueue_create_buffer[DType.float32](rows * ngroups)
    var d_as = ctx.enqueue_create_buffer[DType.float32](ngroups)
    var d_y = ctx.enqueue_create_buffer[DType.float32](rows)
    with d_codes.map_to_host() as m:
        var q = m.unsafe_ptr()
        for i in range(nelem // 4):
            q[unsafe_offset=i] = tp.codes[i]
    with d_xp.map_to_host() as m:
        var q = m.unsafe_ptr()
        for i in range(cols):
            q[unsafe_offset=i] = xperm[i]     # stride-4 order, matching the unpack
    with d_ws.map_to_host() as m:
        var q = m.unsafe_ptr()
        for i in range(rows * ngroups):
            q[unsafe_offset=i] = tp.scales[i]
    with d_as.map_to_host() as m:
        var q = m.unsafe_ptr()
        for i in range(ngroups):
            q[unsafe_offset=i] = ascale[i]

    var bestg = 0.0
    for rep in range(REPS):
        var g0 = perf_counter_ns()
        ctx.enqueue_function[packed_kernel](
            d_codes.unsafe_ptr(), d_xp.unsafe_ptr(), d_ws.unsafe_ptr(),
            d_as.unsafe_ptr(), d_y.unsafe_ptr(),
            Int32(cols // 4), Int32(cols), Int32(ngroups),
            grid_dim=rows, block_dim=GPU_THREADS,
        )
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - g0) / 1e6
        if rep == 0 or ms < bestg:
            bestg = ms

    var yg = List[F32](length=rows, fill=0)
    with d_y.map_to_host() as m:
        for i in range(rows):
            yg[i] = m[i]

    var peak: F32 = 0.0
    for i in range(rows):
        var a = y8[i] if y8[i] >= 0.0 else -y8[i]
        if a > peak:
            peak = a
    var gbad: F32 = 0.0
    for i in range(rows):
        var m2 = y8[i] if y8[i] >= 0.0 else -y8[i]
        if m2 < peak * 0.01:
            continue
        var d = yg[i] - y8[i]
        if d < 0.0:
            d = -d
        if d / m2 > gbad:
            gbad = d / m2
    print("  GPU packed vs int8 CPU, max rel:", gbad, " (fp32 reduction order differs)")

    print("")
    print("=== speed (min of", REPS, ") ===")
    print("  CPU int8   :", best8, "ms  ", int8_bytes / best8 / 1e6, "GB/s of own bytes")
    print("  CPU packed :", bestp, "ms  ", pack_bytes / bestp / 1e6, "GB/s of own bytes")
    print("  GPU packed :", bestg, "ms  ", pack_bytes / bestg / 1e6, "GB/s of own bytes")
    print("")
    print("  CPU packed vs CPU int8:", best8 / bestp, "x")
    print("  GPU packed vs CPU int8:", best8 / bestg, "x   <-- the number that matters")
    if gbad > 1e-3:
        raise Error("GPU packed disagrees with the CPU int8 kernel by " + String(gbad))
