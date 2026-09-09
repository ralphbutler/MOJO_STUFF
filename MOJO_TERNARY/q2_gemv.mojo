# q2_gemv.mojo — M1: the real kernel, on real weights.
#
# Takes an actual ternary tensor out of Ternary-Bonsai-1.7B-Q2_0.gguf, unpacks it
# once into int8, and runs a GEMV four ways. This is where the M0 reader and the
# q2_dot.mojo kernel spike finally meet on real data.
#
#   scalar        one weight at a time, int32 accumulate     — the honest reference
#   widen         SIMD, cast int8 -> int32 then multiply     — what portable Mojo lowers to
#   sdot          NEON SDOT via llvm_intrinsic               — the actual M1 kernel
#   sdot+threads  the same, rows split across cores
#
# THREE references, because "is it correct" and "is it accurate" are different
# questions and conflating them hides bugs:
#
#   ref_exact  fp32 weights (from the Q2_0 file) . fp32 activations
#              The true answer. Distance from this is *quantization error* and is
#              expected to be nonzero.
#   ref_actq   fp32 weights . dequantized-int8 activations
#              Same weights, same quantized activations as our kernel, but done in
#              fp32. Our int8 kernels must match this *tightly* — the only
#              remaining difference is accumulation order. This is the bug detector.
#   ours       the int8 pipeline under test
#
# Note on the plan: IDEA_CURR.md's M1 calls for "zero-point correction via
# precomputed column sums". That is not needed here. Both sides are symmetric —
# weights are {-1,0,+1} and activations are absmax int8 — so both zero-points are
# 0 and the correction terms vanish. Deleted rather than implemented.
#
# Usage: mojo run q2_gemv.mojo [tensor-name] [q2_path]

from gguf import (
    GGUF,
    TERNARY_BLOCK_BYTES,
    TERNARY_GROUP,
    dequant_q2_g128,
    load_header,
    read_blob,
)
from max.algorithm import parallelize
from std.memory import bitcast
from std.sys import argv, llvm_intrinsic, num_physical_cores, simd_width_of
from std.time import perf_counter_ns

comptime W = simd_width_of[DType.int8]()      # 16 on NEON
comptime REPS = 20
comptime ROW_BLOCK = 32

comptime I8 = Scalar[DType.int8]
comptime F32 = Scalar[DType.float32]
comptime I32x4 = SIMD[DType.int32, 4]
comptime I8x16 = SIMD[DType.int8, 16]


# ---------------------------------------------------------------------------
# Load-time unpack: 2-bit codes -> int8 in {-1,0,+1}, plus the FP16 scales as
# fp32. Done once (IDEA_CURR.md Correction #2), never inside a kernel.
# ---------------------------------------------------------------------------
struct TernaryTensor(Movable):
    var w: List[I8]             # rows * cols, row-major
    var scales: List[F32]       # rows * (cols // 128)
    var rows: Int
    var cols: Int

    def __init__(out self, var blob: List[UInt8], rows: Int, cols: Int) raises:
        if cols % TERNARY_GROUP != 0:
            raise Error(
                "cols must be a multiple of 128 so blocks never straddle a row; got "
                + String(cols)
            )
        self.rows = rows
        self.cols = cols
        var nelem = rows * cols
        var nblocks = nelem // TERNARY_GROUP
        self.w = List[I8](length=nelem, fill=0)
        self.scales = List[F32](length=nblocks, fill=0)

        for b in range(nblocks):
            var base = b * TERNARY_BLOCK_BYTES
            var raw = UInt16(blob[base]) | (UInt16(blob[base + 1]) << 8)
            self.scales[b] = bitcast[DType.float16](raw).cast[DType.float32]()
            for j in range(32):
                var byte = blob[base + 2 + j]
                for k in range(4):
                    var q = (byte >> UInt8(2 * k)) & 0x3
                    self.w[b * TERNARY_GROUP + j * 4 + k] = (
                        q.cast[DType.int8]() - 1
                    )

    def groups_per_row(self) -> Int:
        return self.cols // TERNARY_GROUP


def quantize_activations(x: List[F32], mut xq: List[I8]) -> F32:
    var amax: F32 = 0.0
    for i in range(len(x)):
        var a = x[i]
        if a < 0.0:
            a = -a
        if a > amax:
            amax = a
    if amax == 0.0:
        return 0.0
    var scale = amax / 127.0
    var inv = 1.0 / scale
    for i in range(len(x)):
        var v = x[i] * inv
        var r = (v + 0.5) if v >= 0.0 else (v - 0.5)
        var ri = Int(r)
        if ri > 127:
            ri = 127
        if ri < -127:
            ri = -127
        xq[i] = I8(ri)
    return scale


# --- kernel 1: scalar reference --------------------------------------------
def gemv_scalar(t: TernaryTensor, xq: List[I8], act_scale: F32, mut y: List[F32]):
    var ng = t.groups_per_row()
    for r in range(t.rows):
        var row = r * t.cols
        var acc: F32 = 0.0
        for g in range(ng):
            var isum: Int32 = 0
            for i in range(TERNARY_GROUP):
                isum += (
                    t.w[row + g * TERNARY_GROUP + i].cast[DType.int32]()
                    * xq[g * TERNARY_GROUP + i].cast[DType.int32]()
                )
            acc += F32(isum) * t.scales[r * ng + g]
        y[r] = acc * act_scale


# --- kernel 2: portable SIMD (widen then multiply) --------------------------
def gemv_widen(t: TernaryTensor, xq: List[I8], act_scale: F32, mut y: List[F32]):
    var wp = t.w.unsafe_ptr()
    var xp = xq.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var ng = t.groups_per_row()
    for r in range(t.rows):
        var row = r * t.cols
        var acc: F32 = 0.0
        for g in range(ng):
            var lanes = SIMD[DType.int32, W](0)
            var i = 0
            while i < TERNARY_GROUP:
                lanes += (
                    wp.unsafe_load[width=W](row + g * TERNARY_GROUP + i)
                    .cast[DType.int32]()
                    * xp.unsafe_load[width=W](g * TERNARY_GROUP + i)
                    .cast[DType.int32]()
                )
                i += W
            acc += F32(lanes.reduce_add()) * sp[unsafe_offset = r * ng + g]
        y[r] = acc * act_scale


# --- kernel 3: NEON SDOT ----------------------------------------------------
# sdot takes an int32x4 accumulator and two int8x16 vectors, and adds four
# 4-element dot products into the four int32 lanes: 16 MACs per instruction,
# with the int32 accumulation free. Dense — every weight participates, zeros
# included (Correction #1).
def gemv_sdot(t: TernaryTensor, xq: List[I8], act_scale: F32, mut y: List[F32]):
    var wp = t.w.unsafe_ptr()
    var xp = xq.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var ng = t.groups_per_row()
    for r in range(t.rows):
        var row = r * t.cols
        var acc: F32 = 0.0
        for g in range(ng):
            var lanes = I32x4(0)
            var i = 0
            while i < TERNARY_GROUP:
                lanes = llvm_intrinsic["llvm.aarch64.neon.sdot", I32x4](
                    lanes,
                    wp.unsafe_load[width=16](row + g * TERNARY_GROUP + i),
                    xp.unsafe_load[width=16](g * TERNARY_GROUP + i),
                )
                i += 16
            acc += F32(lanes.reduce_add()) * sp[unsafe_offset = r * ng + g]
        y[r] = acc * act_scale


# --- kernel 4: SDOT across cores -------------------------------------------
def gemv_sdot_par(
    t: TernaryTensor, xq: List[I8], act_scale: F32, mut y: List[F32], workers: Int
):
    var wp = t.w.unsafe_ptr()
    var xp = xq.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var ng = t.groups_per_row()
    var cols = t.cols
    var rows = t.rows

    def row_block(blk: Int) capturing:
        var r0 = blk * ROW_BLOCK
        var r1 = min(r0 + ROW_BLOCK, rows)
        for r in range(r0, r1):
            var row = r * cols
            var acc: F32 = 0.0
            for g in range(ng):
                var lanes = I32x4(0)
                var i = 0
                while i < TERNARY_GROUP:
                    lanes = llvm_intrinsic["llvm.aarch64.neon.sdot", I32x4](
                        lanes,
                        wp.unsafe_load[width=16](row + g * TERNARY_GROUP + i),
                        xp.unsafe_load[width=16](g * TERNARY_GROUP + i),
                    )
                    i += 16
                acc += F32(lanes.reduce_add()) * sp[unsafe_offset = r * ng + g]
            yp[unsafe_offset=r] = acc * act_scale

    parallelize[row_block]((rows + ROW_BLOCK - 1) // ROW_BLOCK, workers)


# Comparing a GEMV output against a reference needs care on two counts:
#
#   1. The reference must be MORE accurate than the thing under test, or the test
#      measures the reference's error. Ours accumulates exactly in int32 per block
#      of 128, so an fp32 reference summing 2048 terms sequentially is the *less*
#      accurate of the two. References here therefore accumulate in float64.
#   2. Relative error is meaningless where the true output is near zero — rows
#      whose terms cancel will show enormous relative error from a tiny absolute
#      one. So the floor is data-driven: 1% of the largest |reference| value,
#      rather than an arbitrary constant.
def rel_floor(expected: List[Float64]) -> Float64:
    var m: Float64 = 0.0
    for i in range(len(expected)):
        var a = expected[i] if expected[i] >= 0.0 else -expected[i]
        if a > m:
            m = a
    return m * 0.01


def max_rel_diff(expected: List[Float64], got: List[F32], floor: Float64) -> Float64:
    var worst: Float64 = 0.0
    for i in range(len(expected)):
        var m = expected[i] if expected[i] >= 0.0 else -expected[i]
        if m < floor:
            continue
        var d = Float64(got[i]) - expected[i]
        if d < 0.0:
            d = -d
        var rel = d / m
        if rel > worst:
            worst = rel
    return worst


def max_abs_diff(expected: List[Float64], got: List[F32]) -> Float64:
    var worst: Float64 = 0.0
    for i in range(len(expected)):
        var d = Float64(got[i]) - expected[i]
        if d < 0.0:
            d = -d
        if d > worst:
            worst = d
    return worst


def main() raises:
    var args = argv()
    comptime SNAP = "/Users/rbutler/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-gguf/snapshots/983b5dec2ff16aab79990711ba0f828a499a7e6a/"
    var tname = String("blk.0.ffn_gate.weight")
    var q2_path = String(SNAP) + "Ternary-Bonsai-1.7B-Q2_0.gguf"
    if len(args) > 1:
        tname = String(args[1])
    if len(args) > 2:
        q2_path = String(args[2])

    var workers = num_physical_cores()
    var g = load_header(q2_path.copy(), False)
    var ti = g.index_of(tname)
    if ti < 0:
        raise Error("tensor not found: " + tname)

    var cols = g.ne0[ti]
    var rows = g.ne1[ti]
    var nelem = g.counts[ti]
    print("M1 — real ternary GEMV from the GGUF")
    print("  tensor :", tname)
    print("  shape  :", rows, "rows x", cols, "cols  =", nelem, "weights")
    print("  simd   : width", W, " workers", workers)
    print("")

    # --- load + unpack once -------------------------------------------------
    var nblocks = nelem // TERNARY_GROUP
    var blob = read_blob(
        q2_path, g.data_start + g.offsets[ti], nblocks * TERNARY_BLOCK_BYTES
    )
    var t0 = perf_counter_ns()
    var t = TernaryTensor(blob.copy(), rows, cols)
    print("  unpack :", Float64(perf_counter_ns() - t0) / 1e6, "ms  ->",
          Float64(nelem) / 1e6, "MB int8 resident")

    # Sanity: the unpacked weights must be ternary.
    var nneg = 0
    var nzero = 0
    var npos = 0
    for i in range(nelem):
        var v = Int(t.w[i])
        if v == -1:
            nneg += 1
        elif v == 0:
            nzero += 1
        elif v == 1:
            npos += 1
        else:
            raise Error("non-ternary weight after unpack: " + String(v))
    print("  ternary: -1:", nneg, " 0:", nzero, " +1:", npos)
    print("")

    # --- activations --------------------------------------------------------
    var x = List[F32](length=cols, fill=0)
    var xq = List[I8](length=cols, fill=0)
    for i in range(cols):
        x[i] = F32((i * 37) % 211) * 0.0094 - 1.0
    var act_scale = quantize_activations(x, xq)

    # --- references ---------------------------------------------------------
    var wf = dequant_q2_g128(blob, nelem)      # exactly our int8 weights * scale
    var ref_exact = List[Float64](length=rows, fill=0)
    var ref_actq = List[Float64](length=rows, fill=0)
    for r in range(rows):
        var a: Float64 = 0.0
        var b: Float64 = 0.0
        for cc in range(cols):
            var wv = Float64(wf[r * cols + cc])
            a += wv * Float64(x[cc])
            b += wv * (Float64(xq[cc]) * Float64(act_scale))
        ref_exact[r] = a
        ref_actq[r] = b

    # --- the four kernels ---------------------------------------------------
    var y_scalar = List[F32](length=rows, fill=0)
    var y_widen = List[F32](length=rows, fill=0)
    var y_sdot = List[F32](length=rows, fill=0)
    var y_par = List[F32](length=rows, fill=0)
    gemv_scalar(t, xq, act_scale, y_scalar)
    gemv_widen(t, xq, act_scale, y_widen)
    gemv_sdot(t, xq, act_scale, y_sdot)
    gemv_sdot_par(t, xq, act_scale, y_par, workers)

    print("=== correctness ===")
    var floor_q = rel_floor(ref_actq)
    var floor_e = rel_floor(ref_exact)
    print("  relative-error floor (1% of peak |output|):", floor_q)
    # Against ref_actq: identical weights, identical quantized activations, but
    # accumulated in float64. Only arithmetic differs, so this is the bug detector.
    print("  vs ref_actq in f64 (bug detector — max rel / max abs):")
    print("    scalar :", max_rel_diff(ref_actq, y_scalar, floor_q),
          "/", max_abs_diff(ref_actq, y_scalar))
    print("    widen  :", max_rel_diff(ref_actq, y_widen, floor_q),
          "/", max_abs_diff(ref_actq, y_widen))
    print("    sdot   :", max_rel_diff(ref_actq, y_sdot, floor_q),
          "/", max_abs_diff(ref_actq, y_sdot))
    print("    par    :", max_rel_diff(ref_actq, y_par, floor_q),
          "/", max_abs_diff(ref_actq, y_par))

    # The int8 kernels should agree with each other exactly -- identical integer
    # math, identical order.
    var exact_pairs = 0
    for r in range(rows):
        if y_scalar[r] == y_widen[r] and y_scalar[r] == y_sdot[r] and y_scalar[r] == y_par[r]:
            exact_pairs += 1
    print("  all four int8 kernels bit-identical on", exact_pairs, "of", rows, "rows")

    print("  vs ref_exact (activation quantization cost, NOT a bug):")
    print("    sdot   :", max_rel_diff(ref_exact, y_sdot, floor_e),
          "/", max_abs_diff(ref_exact, y_sdot))
    print("")

    if exact_pairs != rows:
        raise Error("the int8 kernels disagree with each other -- real bug")
    # Our int32-per-block accumulation is *more* accurate than a sequential fp32
    # sum, so the only slack expected here is fp32 rounding of 16 block terms.
    var bug = max_rel_diff(ref_actq, y_sdot, floor_q)
    if bug > 1e-5:
        raise Error(
            "sdot differs from the f64 reference by " + String(bug)
            + " relative -- that is a bug, not rounding"
        )

    # --- timing -------------------------------------------------------------
    var best_scalar = 0.0
    var best_widen = 0.0
    var best_sdot = 0.0
    var best_par = 0.0
    for rep in range(REPS):
        var a0 = perf_counter_ns()
        gemv_scalar(t, xq, act_scale, y_scalar)
        var a = Float64(perf_counter_ns() - a0) / 1e6
        if rep == 0 or a < best_scalar:
            best_scalar = a

        var b0 = perf_counter_ns()
        gemv_widen(t, xq, act_scale, y_widen)
        var b = Float64(perf_counter_ns() - b0) / 1e6
        if rep == 0 or b < best_widen:
            best_widen = b

        var c0 = perf_counter_ns()
        gemv_sdot(t, xq, act_scale, y_sdot)
        var c = Float64(perf_counter_ns() - c0) / 1e6
        if rep == 0 or c < best_sdot:
            best_sdot = c

        var d0 = perf_counter_ns()
        gemv_sdot_par(t, xq, act_scale, y_par, workers)
        var d = Float64(perf_counter_ns() - d0) / 1e6
        if rep == 0 or d < best_par:
            best_par = d

    var mb = Float64(nelem)
    print("=== speed (min of", REPS, "runs) ===")
    print("             time (ms)      GB/s      vs scalar")
    print("  scalar  :", best_scalar, "  ", mb / best_scalar / 1e6, "   1.0x")
    print("  widen   :", best_widen, "  ", mb / best_widen / 1e6, "  ",
          best_scalar / best_widen, "x")
    print("  sdot    :", best_sdot, "  ", mb / best_sdot / 1e6, "  ",
          best_scalar / best_sdot, "x")
    print("  sdot+par:", best_par, "  ", mb / best_par / 1e6, "  ",
          best_scalar / best_par, "x")
    print("")
    print("  sdot vs widen (does the intrinsic earn its keep?):",
          best_widen / best_sdot, "x")
    print("")
    if best_scalar / best_sdot > 10.0:
        print("  M1 DONE: matches the fp32 reference, and sdot beats scalar by >10x.")
    else:
        print("  M1 INCOMPLETE: sdot is only", best_scalar / best_sdot,
              "x over scalar; the bar is 10x.")
