# q2_dot.mojo — the smallest honest slice of this engine, end to end.
#
# It walks the whole data path from IDEA_CURR.md's diagram on one synthetic
# tensor, with no GGUF file and no model:
#
#   packed 2-bit codes + one scale per 128        (what sits on disk)
#     -> load-time unpack to int8 in {-1,0,+1}, w = q - 1   (Correction #2)
#       -> per-token absmax int8 activation quant
#         -> dense int8 GEMV, no zero-skipping              (Correction #1)
#           -> group-128 scale applied to int32 sums
#             -> fp32 output
#
# Two implementations of that GEMV run against each other:
#   scalar — one weight at a time, the M0 reference
#   simd   — W lanes at a time, dense, the seed of the M1 NEON kernel
#
# Both accumulate in int32, so agreement is *exact*, not "within tolerance".
# A mismatch of even one count is a real bug, and the check says so.
#
# Two things this file is deliberately not yet:
#
#   1. It does not emit `sdot`. Widen-then-multiply is what a portable Mojo
#      cast lowers to. Reaching SDOT's 16-int8-pairs-per-instruction is the
#      actual work of M1; this establishes the answer that kernel must
#      reproduce and the baseline it must beat.
#   2. Scales here are fp32, not the fp16 the real Q2_0_g128 block carries.
#      Nothing in this file depends on the width, and M0's GGUF reader is
#      where fp16 has to become real.
#
# Toolchain: Mojo 1.0.0. Storage is `List`, not raw `alloc` — List owns its
# memory, so there is no destructor and no `free()` in this file, and the
# kernel still gets a raw `Pointer` via `unsafe_ptr()` for wide SIMD loads.

# `parallelize` left the stdlib in Mojo 1.0 -- it now lives in `max.algorithm`,
# which is why this project depends on `max` (+293MB in .venv) for what is
# otherwise a from-scratch engine. IDEA_CURR.md's M3 names `parallelize`
# explicitly, so this is the sanctioned path rather than a hand-rolled pool.
from max.algorithm import parallelize
from std.sys import simd_width_of, num_physical_cores
from std.time import perf_counter_ns

comptime GROUP = 128                          # Q2_0_g128: one scale per 128 weights
comptime COLS = 16384                         # weights per row  (128 groups)
comptime ROWS = 16384                         # output features
comptime REPS = 5                             # timed repetitions, min taken

# 16384^2 int8 = 268 MB of weights, chosen deliberately: it is an order of
# magnitude past this machine's caches, so the GEMV number is a real DRAM
# bandwidth measurement rather than an L2/SLC residency artifact. At 4096^2
# (17 MB) the matrix largely sat in cache and the GB/s figure flattered itself.

# Rows are independent, so they parallelize with no locks. Blocking them keeps
# scheduling overhead off the critical path -- one work item per row would hand
# the scheduler 16384 tiny tasks.
comptime ROW_BLOCK = 64

comptime W = simd_width_of[DType.int8]()      # int8 lanes per SIMD register (NEON: 16)

comptime I8 = Scalar[DType.int8]
comptime U8 = Scalar[DType.uint8]
comptime F32 = Scalar[DType.float32]


# ---------------------------------------------------------------------------
# Synthetic Q2_0_g128 tensor: packed 2-bit codes, 4 to a byte, + scales.
# Deterministic and asymmetric, so a row/column mix-up shows up as wrong
# numbers rather than a plausible-looking answer.
# ---------------------------------------------------------------------------
struct PackedQ2(Movable):
    var codes: List[U8]                       # 2-bit codes, 4 per byte
    var scales: List[F32]                     # one per group of 128
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.codes = List[U8](length=rows * cols // 4, fill=0)
        self.scales = List[F32](length=rows * cols // GROUP, fill=0)

        # Fill: q cycles through {0,1,2} with a row-dependent phase, so every
        # row has a different ternary pattern and roughly a third are zeros.
        var bytes_per_row = cols // 4
        for r in range(rows):
            for b in range(bytes_per_row):
                var byte: U8 = 0
                for k in range(4):
                    var i = b * 4 + k
                    var q = U8((i * 7 + r * 3) % 3)   # {0,1,2} -> w in {-1,0,+1}
                    byte |= q << U8(2 * k)
                self.codes[r * bytes_per_row + b] = byte
            for g in range(cols // GROUP):
                self.scales[r * (cols // GROUP) + g] = F32((r + g) % 11) * 0.01 + 0.02


# ---------------------------------------------------------------------------
# Unpacked, RAM-resident int8 weights. This is the shape the kernel sees.
# The 4x memory inflation happens here, once, and never again.
# ---------------------------------------------------------------------------
struct TernaryMatrix(Movable):
    var w: List[I8]                           # int8 in {-1, 0, +1}, row-major
    var scales: List[F32]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.w = List[I8](length=rows * cols, fill=0)
        self.scales = List[F32](length=rows * cols // GROUP, fill=0)

    def groups_per_row(self) -> Int:
        return self.cols // GROUP


# Load-time unpack: 2-bit code q -> int8 weight w = q - 1. Done once, at load.
# Doing this inside the matmul instead is the single biggest way to throw the
# whole performance argument away (IDEA_CURR.md Correction #2).
# Rows are independent here too, and at 27B this is a ~27GB one-time cost that
# is worth every core we can throw at it.
def unpack(packed: PackedQ2, mut out_m: TernaryMatrix, workers: Int):
    var bytes_per_row = packed.cols // 4
    var cols = packed.cols
    var cp = packed.codes.unsafe_ptr()
    var wp = out_m.w.unsafe_ptr()

    def unpack_block(blk: Int) capturing:
        var r0 = blk * ROW_BLOCK
        var r1 = min(r0 + ROW_BLOCK, packed.rows)
        for r in range(r0, r1):
            for b in range(bytes_per_row):
                var byte = cp[unsafe_offset = r * bytes_per_row + b]
                for k in range(4):
                    var q = (byte >> U8(2 * k)) & 0x3
                    # q - 1 maps {0,1,2} -> {-1,0,+1}. Cast before the subtract:
                    # uint8 0 - 1 wraps to 255.
                    wp[unsafe_offset = r * cols + b * 4 + k] = (
                        q.cast[DType.int8]() - 1
                    )

    parallelize[unpack_block]((packed.rows + ROW_BLOCK - 1) // ROW_BLOCK, workers)

    for i in range(packed.rows * (packed.cols // GROUP)):
        out_m.scales[i] = packed.scales[i]


# ---------------------------------------------------------------------------
# Activation quantization: per-token absmax, symmetric, into int8.
# Returns the scale that puts the int32 dot product back into fp32.
# ---------------------------------------------------------------------------
def quantize_activations(x: List[F32], mut xq: List[I8]) -> F32:
    var n = len(x)
    var amax: F32 = 0.0
    for i in range(n):
        var a = x[i]
        if a < 0.0:
            a = -a
        if a > amax:
            amax = a
    if amax == 0.0:
        for i in range(n):
            xq[i] = 0
        return 0.0

    var scale = amax / 127.0
    var inv = 1.0 / scale
    for i in range(n):
        var v = x[i] * inv
        # round-half-away-from-zero, then clamp into int8 range
        var r = (v + 0.5) if v >= 0.0 else (v - 0.5)
        var ri = Int(r)
        if ri > 127:
            ri = 127
        if ri < -127:
            ri = -127
        xq[i] = I8(ri)
    return scale


# ---------------------------------------------------------------------------
# GEMV, scalar. The M0 reference: obviously correct, no cleverness.
# ---------------------------------------------------------------------------
def gemv_scalar(m: TernaryMatrix, xq: List[I8], act_scale: F32, mut y: List[F32]):
    var ngroups = m.groups_per_row()
    for r in range(m.rows):
        var row = r * m.cols
        var acc: F32 = 0.0
        for g in range(ngroups):
            # Two different offsets, and mixing them up is the easy bug here:
            # weights are row-major over the whole matrix, activations are one
            # row long and get re-read for every output feature.
            var base = row + g * GROUP        # into m.w   (row-global)
            var col = g * GROUP               # into xq    (column-local)
            var isum: Int32 = 0
            for i in range(GROUP):
                # Dense. Every weight participates, zeros included. Branching to
                # skip the zeros costs more than the multiply (Correction #1).
                isum += m.w[base + i].cast[DType.int32]() * xq[col + i].cast[
                    DType.int32
                ]()
            acc += F32(isum) * m.scales[r * ngroups + g]
        y[r] = acc * act_scale


# ---------------------------------------------------------------------------
# GEMV, SIMD. Same arithmetic, W int8 pairs per step, int32 lane accumulators.
# GROUP (128) is a whole number of SIMD registers on every plausible target,
# so there is no tail loop to get wrong -- asserted at compile time below.
# ---------------------------------------------------------------------------
def gemv_simd(m: TernaryMatrix, xq: List[I8], act_scale: F32, mut y: List[F32]):
    comptime assert GROUP % W == 0, "GROUP must be a whole number of SIMD registers"

    # Raw pointers for wide loads. List owns the memory; these just read it.
    var wp = m.w.unsafe_ptr()
    var xp = xq.unsafe_ptr()

    var ngroups = m.groups_per_row()
    for r in range(m.rows):
        var row = r * m.cols
        var acc: F32 = 0.0
        for g in range(ngroups):
            var base = row + g * GROUP        # into m.w   (row-global)
            var col = g * GROUP               # into xq    (column-local)
            var lanes = SIMD[DType.int32, W](0)
            var i = 0
            while i < GROUP:
                var wv = wp.unsafe_load[width=W](base + i).cast[DType.int32]()
                var xv = xp.unsafe_load[width=W](col + i).cast[DType.int32]()
                lanes += wv * xv
                i += W
            # One horizontal reduction per group of 128, not per element.
            acc += F32(lanes.reduce_add()) * m.scales[r * ngroups + g]
        y[r] = acc * act_scale


# ---------------------------------------------------------------------------
# GEMV, SIMD + threads. Identical arithmetic to gemv_simd, so it stays
# bit-exact -- each row's accumulation order is untouched, only *which thread*
# walks the row changes. That is what makes the parity check meaningful here:
# a race would show up as a mismatch, not as drift.
# ---------------------------------------------------------------------------
def gemv_parallel(
    m: TernaryMatrix, xq: List[I8], act_scale: F32, mut y: List[F32], workers: Int
):
    var wp = m.w.unsafe_ptr()
    var xp = xq.unsafe_ptr()
    var sp = m.scales.unsafe_ptr()
    var yp = y.unsafe_ptr()          # each row is written by exactly one thread
    var ngroups = m.groups_per_row()
    var cols = m.cols
    var rows = m.rows

    def row_block(blk: Int) capturing:
        var r0 = blk * ROW_BLOCK
        var r1 = min(r0 + ROW_BLOCK, rows)
        for r in range(r0, r1):
            var row = r * cols
            var acc: F32 = 0.0
            for g in range(ngroups):
                var base = row + g * GROUP
                var col = g * GROUP
                var lanes = SIMD[DType.int32, W](0)
                var i = 0
                while i < GROUP:
                    var wv = wp.unsafe_load[width=W](base + i).cast[DType.int32]()
                    var xv = xp.unsafe_load[width=W](col + i).cast[DType.int32]()
                    lanes += wv * xv
                    i += W
                acc += F32(lanes.reduce_add()) * sp[unsafe_offset = r * ngroups + g]
            yp[unsafe_offset=r] = acc * act_scale

    parallelize[row_block]((rows + ROW_BLOCK - 1) // ROW_BLOCK, workers)


# ---------------------------------------------------------------------------
# Roofline reference: stream the whole weight tensor doing almost no work.
# One widening add per SIMD register is the least compute per byte we can do
# and still be forced to actually read every byte. Whatever this reaches is the
# practical ceiling for the GEMV -- it says whether the kernel is bandwidth-
# bound or still instruction-bound.
# ---------------------------------------------------------------------------
def stream_read(m: TernaryMatrix, workers: Int) -> Int:
    var wp = m.w.unsafe_ptr()
    var cols = m.cols
    var rows = m.rows
    var partials = List[Int32](length=(rows + ROW_BLOCK - 1) // ROW_BLOCK, fill=0)
    var pp = partials.unsafe_ptr()

    def block(blk: Int) capturing:
        var r0 = blk * ROW_BLOCK
        var r1 = min(r0 + ROW_BLOCK, rows)
        var lanes = SIMD[DType.int32, W](0)
        for r in range(r0, r1):
            var row = r * cols
            var i = 0
            while i < cols:
                lanes += wp.unsafe_load[width=W](row + i).cast[DType.int32]()
                i += W
        pp[unsafe_offset=blk] = lanes.reduce_add()

    parallelize[block]((rows + ROW_BLOCK - 1) // ROW_BLOCK, workers)

    var total: Int = 0
    for i in range(len(partials)):
        total += Int(partials[i])
    return total


def main() raises:
    var workers = num_physical_cores()
    print("ternary GEMV --", ROWS, "x", COLS, " group =", GROUP, " simd width =", W)
    print("  row block =", ROW_BLOCK, " workers =", workers)
    print("")

    # --- build + unpack -----------------------------------------------------
    var packed = PackedQ2(ROWS, COLS)
    var m = TernaryMatrix(ROWS, COLS)

    var t0 = perf_counter_ns()
    unpack(packed, m, workers)
    var t_unpack = Float64(perf_counter_ns() - t0) / 1e6

    var on_disk = Float64(ROWS * COLS) / 4.0 + Float64(ROWS * COLS // GROUP) * 2.0
    var in_ram = Float64(ROWS * COLS) + Float64(ROWS * COLS // GROUP) * 4.0
    print("unpack        :", t_unpack, "ms  (parallel)")
    print("  2-bit packed:", on_disk / 1e6, "MB")
    print("  int8 in RAM :", in_ram / 1e6, "MB   (the 4x we deliberately pay)")
    # Extrapolate the one-time load cost up the ladder: 27B unpacks ~27GB.
    print(
        "  -> at 27GB that rate is",
        27.0e9 / (in_ram / (t_unpack / 1e3)) ,
        "s of load-time unpack",
    )

    # Sanity: every unpacked weight must be in {-1, 0, +1}.
    var nneg = 0
    var nzero = 0
    var npos = 0
    for i in range(ROWS * COLS):
        var v = Int(m.w[i])
        if v == -1:
            nneg += 1
        elif v == 0:
            nzero += 1
        elif v == 1:
            npos += 1
        else:
            raise Error("unpack produced a non-ternary weight: " + String(v))
    print("  ternary check: -1:", nneg, " 0:", nzero, " +1:", npos)
    print("")

    # --- activations --------------------------------------------------------
    var x = List[F32](length=COLS, fill=0)
    var xq = List[I8](length=COLS, fill=0)
    for i in range(COLS):
        x[i] = F32((i * 13) % 97) * 0.031 - 1.4      # spans negative and positive
    var act_scale = quantize_activations(x, xq)
    print("activation quant: absmax scale =", act_scale)
    print("")

    # --- correctness: all three agree, exactly ------------------------------
    var y_ref = List[F32](length=ROWS, fill=0)
    var y_simd = List[F32](length=ROWS, fill=0)
    var y_par = List[F32](length=ROWS, fill=0)
    gemv_scalar(m, xq, act_scale, y_ref)
    gemv_simd(m, xq, act_scale, y_simd)
    gemv_parallel(m, xq, act_scale, y_par, workers)

    var bad_simd = 0
    var bad_par = 0
    for r in range(ROWS):
        if y_ref[r] != y_simd[r]:
            bad_simd += 1
        if y_ref[r] != y_par[r]:
            bad_par += 1
    if bad_simd != 0 or bad_par != 0:
        raise Error(
            "parity failed -- simd: " + String(bad_simd)
            + " rows, parallel: " + String(bad_par) + " rows"
        )
    print("parity        : scalar == simd == parallel on all", ROWS, "rows (bit-exact)")
    print("  y[0] =", y_ref[0], " y[last] =", y_ref[ROWS - 1])
    print("")

    # --- timing: min of REPS, the least noisy summary of a short kernel ------
    var best_scalar = 0.0
    var best_simd = 0.0
    var best_par = 0.0
    for rep in range(REPS):
        var a0 = perf_counter_ns()
        gemv_scalar(m, xq, act_scale, y_ref)
        var a = Float64(perf_counter_ns() - a0) / 1e6
        if rep == 0 or a < best_scalar:
            best_scalar = a

        var b0 = perf_counter_ns()
        gemv_simd(m, xq, act_scale, y_simd)
        var b = Float64(perf_counter_ns() - b0) / 1e6
        if rep == 0 or b < best_simd:
            best_simd = b

        var c0 = perf_counter_ns()
        gemv_parallel(m, xq, act_scale, y_par, workers)
        var c = Float64(perf_counter_ns() - c0) / 1e6
        if rep == 0 or c < best_par:
            best_par = c

    # Decode is bandwidth-bound, so weight bytes streamed per second is the
    # number that matters -- not FLOP/s. One GEMV reads the whole matrix once.
    var wbytes = Float64(ROWS * COLS)
    print("             time (ms)     GB/s     vs scalar")
    print("scalar   :", best_scalar, "  ", wbytes / best_scalar / 1e6, "   1.0x")
    print(
        "simd     :", best_simd, "  ", wbytes / best_simd / 1e6,
        "  ", best_scalar / best_simd, "x",
    )
    print(
        "parallel :", best_par, "  ", wbytes / best_par / 1e6,
        "  ", best_scalar / best_par, "x",
    )
    print("")

    # --- roofline: how fast can we merely READ the weights? -----------------
    var best_stream = 0.0
    var checksum: Int = 0
    for rep in range(REPS):
        var d0 = perf_counter_ns()
        checksum = stream_read(m, workers)
        var d = Float64(perf_counter_ns() - d0) / 1e6
        if rep == 0 or d < best_stream:
            best_stream = d
    # The checksum must equal (+1 count) - (-1 count); printing it stops the
    # compiler dead-code-eliminating the whole streaming loop.
    print(
        "stream   :", best_stream, "  ", wbytes / best_stream / 1e6,
        "   (read-only roofline, checksum", checksum, ")",
    )
    print("")
    print("  simd -> parallel scaling:", best_simd / best_par, "x on", workers, "cores")
    print(
        "  parallel GEMV reaches",
        100.0 * (best_stream / best_par), "% of the read-only roofline",
    )
