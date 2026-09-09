# model.mojo — M2 stage A: load the whole Bonsai-1.7B into memory.
#
# Reads every tensor out of the GGUF, unpacks the ternary ones once into int8
# (Correction #2), keeps the norms as fp32, and checks each tensor's shape against
# what the metadata says the architecture should be. A shape that disagrees is a
# loud error here rather than silent garbage in the forward pass.
#
# Architecture, confirmed from both the GGUF metadata and the upstream config.json:
#   Qwen3ForCausalLM, 28 blocks, all full attention (no sliding window)
#   hidden 2048, FFN 6144, 16 query heads over 8 KV heads, head_dim 128
#   RMSNorm eps 1e-6, SwiGLU, no biases anywhere
#   QK-norm: a per-head RMSNorm on q and k, weight length 128 = head_dim
#   tied embeddings -- there is no output.weight, the LM head reuses token_embd

from gguf import (
    GGUF,
    TERNARY_BLOCK_BYTES,
    TERNARY_GROUP,
    load_header,
    read_blob,
)
from max.algorithm import parallelize
from std.math import ceil, floor, log
from std.memory import bitcast
from std.sys import llvm_intrinsic, num_physical_cores
from std.time import perf_counter_ns

# Two ggml type ids carry the same 34-byte/128-weight ternary block:
#   42  = legacy Prism Q2_0. What the published 1.7B/4B/8B/27B `*-Q2_0.gguf` files use.
#         Current llama.cpp HEAD reads id 42 as the *official* group-64 layout instead and
#         REFUSES these files outright ("this file matches the legacy Prism Q2_0 layout").
#   142 = PQ2_0, the same group-128 layout under its own id. What `*-PQ2_0.gguf` uses.
# Accept both: the block layout is identical, only the label moved.
comptime TERNARY_TYPE = 42
comptime TERNARY_TYPE_PQ = 142
comptime F32_TYPE = 0

comptime I8 = Scalar[DType.int8]
comptime F32 = Scalar[DType.float32]


# A ternary weight matrix: int8 in {-1,0,+1} plus one fp32 scale per 128 weights.
# GGUF stores dims as [ne0, ne1] with ne0 the fastest-varying axis, which for these
# weight matrices is the INPUT features. So rows = ne1 = outputs, cols = ne0 = inputs,
# and the matrix is row-major with each row holding cols contiguous weights.
struct QTensor(Movable):
    var w: List[I8]
    var scales: List[F32]
    var rows: Int
    var cols: Int

    def __init__(out self):
        self.w = List[I8]()
        self.scales = List[F32]()
        self.rows = 0
        self.cols = 0

    def groups_per_row(self) -> Int:
        return self.cols // TERNARY_GROUP


struct FTensor(Movable):
    var v: List[F32]

    def __init__(out self):
        self.v = List[F32]()


struct Layer(Movable):
    var attn_norm: FTensor
    var attn_q: QTensor
    var attn_k: QTensor
    var attn_v: QTensor
    var attn_output: QTensor
    var attn_q_norm: FTensor
    var attn_k_norm: FTensor
    var ffn_norm: FTensor
    var ffn_gate: QTensor
    var ffn_up: QTensor
    var ffn_down: QTensor

    def __init__(out self):
        self.attn_norm = FTensor()
        self.attn_q = QTensor()
        self.attn_k = QTensor()
        self.attn_v = QTensor()
        self.attn_output = QTensor()
        self.attn_q_norm = FTensor()
        self.attn_k_norm = FTensor()
        self.ffn_norm = FTensor()
        self.ffn_gate = QTensor()
        self.ffn_up = QTensor()
        self.ffn_down = QTensor()


struct Model(Movable):
    var token_embd: QTensor
    var output_head: QTensor     # separate LM head when weights are NOT tied
    var tied_head: Bool          # True -> logits come from token_embd
    var output_norm: FTensor
    var layers: List[Layer]
    # architecture, read from metadata rather than hardcoded
    var n_layer: Int
    var d_model: Int
    var d_ffn: Int
    var n_head: Int
    var n_head_kv: Int
    var head_dim: Int
    var vocab: Int
    var rms_eps: F32
    var rope_base: F32
    var rope_factor: F32         # YaRN scaling factor (1.0 = plain RoPE)
    var rope_orig_ctx: Int       # YaRN original_max_position_embeddings
    var arch: String

    def __init__(out self):
        self.token_embd = QTensor()
        self.output_head = QTensor()
        self.tied_head = True
        self.output_norm = FTensor()
        self.layers = List[Layer]()
        self.n_layer = 0
        self.d_model = 0
        self.d_ffn = 0
        self.n_head = 0
        self.n_head_kv = 0
        self.head_dim = 0
        self.vocab = 0
        self.rms_eps = 1e-6
        self.rope_base = 1000000.0
        self.rope_factor = 1.0
        self.rope_orig_ctx = 0
        self.arch = String("")


# Unpack one ternary tensor straight out of the file buffer. Parallel over blocks:
# at 1.7B this runs 197 times over 1.72 billion weights, so it is worth the threads.
def unpack_qtensor(
    blob: List[UInt8], rows: Int, cols: Int, workers: Int
) raises -> QTensor:
    if cols % TERNARY_GROUP != 0:
        raise Error(
            "cols must be a multiple of 128 so blocks never straddle a row; got "
            + String(cols)
        )
    var t = QTensor()
    t.rows = rows
    t.cols = cols
    var nelem = rows * cols
    var nblocks = nelem // TERNARY_GROUP
    t.w = List[I8](length=nelem, fill=0)
    t.scales = List[F32](length=nblocks, fill=0)

    var rp = blob.unsafe_ptr()
    var wp = t.w.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()

    comptime CHUNK = 256                       # blocks per work item

    def do_chunk(c: Int) capturing:
        var b0 = c * CHUNK
        var b1 = min(b0 + CHUNK, nblocks)
        for b in range(b0, b1):
            var base = b * TERNARY_BLOCK_BYTES
            var lo = UInt16(rp[unsafe_offset=base])
            var hi = UInt16(rp[unsafe_offset = base + 1])
            sp[unsafe_offset=b] = (
                bitcast[DType.float16](lo | (hi << 8)).cast[DType.float32]()
            )
            for j in range(32):
                var byte = rp[unsafe_offset = base + 2 + j]
                for k in range(4):
                    var q = (byte >> UInt8(2 * k)) & 0x3
                    wp[unsafe_offset = b * TERNARY_GROUP + j * 4 + k] = (
                        q.cast[DType.int8]() - 1
                    )

    parallelize[do_chunk]((nblocks + CHUNK - 1) // CHUNK, workers)
    return t^


def read_ftensor(blob: List[UInt8], n: Int) raises -> FTensor:
    var t = FTensor()
    t.v = List[F32](length=n, fill=0)
    var rp = blob.unsafe_ptr()
    for i in range(n):
        var b = i * 4
        var bits: UInt32 = 0
        for k in range(4):
            bits |= UInt32(rp[unsafe_offset = b + k]) << UInt32(8 * k)
        t.v[i] = bitcast[DType.float32](bits)
    return t^


# Pull one named tensor, checking its type and shape against what we expect.
def take_q(
    g: GGUF, path: String, name: String, rows: Int, cols: Int, workers: Int
) raises -> QTensor:
    var i = g.index_of(name)
    if i < 0:
        raise Error("missing tensor: " + name)
    if g.types[i] != TERNARY_TYPE and g.types[i] != TERNARY_TYPE_PQ:
        raise Error(
            name + ": expected ternary type 42 or 142, got " + String(g.types[i])
        )
    if g.ne0[i] != cols or g.ne1[i] != rows:
        raise Error(
            name + ": expected [" + String(cols) + ", " + String(rows)
            + "] got [" + String(g.ne0[i]) + ", " + String(g.ne1[i]) + "]"
        )
    var nbytes = (rows * cols // TERNARY_GROUP) * TERNARY_BLOCK_BYTES
    var blob = read_blob(path, g.data_start + g.offsets[i], nbytes)
    return unpack_qtensor(blob, rows, cols, workers)


def take_f(g: GGUF, path: String, name: String, n: Int) raises -> FTensor:
    var i = g.index_of(name)
    if i < 0:
        raise Error("missing tensor: " + name)
    if g.types[i] != F32_TYPE:
        raise Error(name + ": expected F32, got type " + String(g.types[i]))
    if g.counts[i] != n:
        raise Error(
            name + ": expected " + String(n) + " elements, got " + String(g.counts[i])
        )
    return read_ftensor(read_blob(path, g.data_start + g.offsets[i], n * 4), n)


def load_model(var path: String, workers: Int, verbose: Bool) raises -> Model:
    var g = load_header(path.copy(), False)

    var m = Model()

    # Architecture comes from the file, not from constants. Hardcoding 1.7B's numbers
    # is exactly what would break silently on another rung.
    m.arch = g.meta.get_str("general.architecture", String("qwen3"))
    var a = m.arch + "."
    m.n_layer = g.meta.require_int(a + "block_count")
    m.d_model = g.meta.require_int(a + "embedding_length")
    m.d_ffn = g.meta.require_int(a + "feed_forward_length")
    m.n_head = g.meta.require_int(a + "attention.head_count")
    m.n_head_kv = g.meta.require_int(a + "attention.head_count_kv")
    m.head_dim = g.meta.get_int(a + "attention.key_length", 0)
    m.rms_eps = F32(g.meta.get_f64(a + "attention.layer_norm_rms_epsilon", 1e-6))
    m.rope_base = F32(g.meta.get_f64(a + "rope.freq_base", 1000000.0))
    m.rope_factor = F32(g.meta.get_f64(a + "rope.scaling.factor", 1.0))
    m.rope_orig_ctx = g.meta.get_int(a + "rope.scaling.original_context_length", 0)

    # head_dim is optional in GGUF; derive it when absent.
    if m.head_dim == 0:
        m.head_dim = m.d_model // m.n_head
    if m.n_head_kv == 0 or m.n_head % m.n_head_kv != 0:
        raise Error("head_count must be a multiple of head_count_kv")

    # Vocabulary is the embedding tensor's row count -- no need to walk the token array.
    var te = g.index_of("token_embd.weight")
    if te < 0:
        raise Error("missing token_embd.weight")
    m.vocab = g.ne1[te]

    var d_kv = m.n_head_kv * m.head_dim

    m.token_embd = take_q(g, path, "token_embd.weight", m.vocab, m.d_model, workers)
    m.output_norm = take_f(g, path, "output_norm.weight", m.d_model)

    # Tied embeddings (1.7B) reuse token_embd for the logits; untied (8B) ships a
    # separate output.weight. Detect from the file rather than from a config flag.
    var oh = g.index_of("output.weight")
    if oh >= 0:
        m.tied_head = False
        m.output_head = take_q(g, path, "output.weight", m.vocab, m.d_model, workers)
    else:
        m.tied_head = True

    for l in range(m.n_layer):
        var p = "blk." + String(l) + "."
        var lay = Layer()
        lay.attn_norm = take_f(g, path, p + "attn_norm.weight", m.d_model)
        lay.attn_q = take_q(g, path, p + "attn_q.weight", m.d_model, m.d_model, workers)
        lay.attn_k = take_q(g, path, p + "attn_k.weight", d_kv, m.d_model, workers)
        lay.attn_v = take_q(g, path, p + "attn_v.weight", d_kv, m.d_model, workers)
        lay.attn_output = take_q(
            g, path, p + "attn_output.weight", m.d_model, m.d_model, workers
        )
        lay.attn_q_norm = take_f(g, path, p + "attn_q_norm.weight", m.head_dim)
        lay.attn_k_norm = take_f(g, path, p + "attn_k_norm.weight", m.head_dim)
        lay.ffn_norm = take_f(g, path, p + "ffn_norm.weight", m.d_model)
        lay.ffn_gate = take_q(g, path, p + "ffn_gate.weight", m.d_ffn, m.d_model, workers)
        lay.ffn_up = take_q(g, path, p + "ffn_up.weight", m.d_ffn, m.d_model, workers)
        lay.ffn_down = take_q(g, path, p + "ffn_down.weight", m.d_model, m.d_ffn, workers)
        m.layers.append(lay^)
        if verbose and (l == 0 or l == m.n_layer - 1):
            print("   loaded block", l)

    return m^


# ---------------------------------------------------------------------------
# YaRN RoPE inverse frequencies, computed from config.
#
# M2 read these from the oracle's dump rather than reimplementing YaRN's ramp. That
# was the right call then, but it does not survive changing rungs: the 8B uses
# original_max_position_embeddings = 16384 against the 1.7B's 8192, which moves the
# ramp (low/high 20/37 vs 17/34) and so changes every frequency.
#
# Verified against the 1.7B's dumped inv_freq to 1.1e-07 max relative -- i.e. fp32
# rounding. `attention_scaling` is 0.1*ln(factor) + 1.
#
# beta_fast / beta_slow are transformers' defaults (32 / 1); the Bonsai configs do
# not override them.
def yarn_inv_freq(
    head_dim: Int, base: F32, factor: F32, orig_ctx: Int
) raises -> List[F32]:
    var half = head_dim // 2
    var out = List[F32](length=half, fill=0)
    if factor <= 1.0 or orig_ctx <= 0:
        # Plain RoPE, no scaling.
        for i in range(half):
            out[i] = 1.0 / (base ** (F32(2 * i) / F32(head_dim)))
        return out^

    comptime BETA_FAST: Float64 = 32.0
    comptime BETA_SLOW: Float64 = 1.0
    comptime TWO_PI: Float64 = 6.283185307179586

    var d = Float64(head_dim)
    var b = Float64(base)
    var oc = Float64(orig_ctx)

    # Which rotary dimension completes `rot` full turns over the original context.
    # Inlined rather than a helper closure: Mojo cannot infer capture conventions here.
    var logb2 = 2.0 * log(b)
    var corr_fast = (d * log(oc / (BETA_FAST * TWO_PI))) / logb2
    var corr_slow = (d * log(oc / (BETA_SLOW * TWO_PI))) / logb2

    var lo = Int(floor(corr_fast))
    var hi = Int(ceil(corr_slow))
    if lo < 0:
        lo = 0
    if hi > half - 1:
        hi = half - 1

    for i in range(half):
        var p = b ** (Float64(2 * i) / d)
        var extrapolation = 1.0 / p                    # keep the original frequency
        var interpolation = 1.0 / (Float64(factor) * p)  # stretch it by the factor
        var ramp: Float64
        if hi == lo:
            ramp = 0.0 if i < lo else 1.0
        else:
            ramp = (Float64(i) - Float64(lo)) / (Float64(hi) - Float64(lo))
        if ramp < 0.0:
            ramp = 0.0
        if ramp > 1.0:
            ramp = 1.0
        var keep = 1.0 - ramp        # 1 -> pure extrapolation, 0 -> pure interpolation
        out[i] = F32(interpolation * (1.0 - keep) + extrapolation * keep)
    return out^


def yarn_attention_scaling(factor: F32) -> F32:
    if factor <= 1.0:
        return 1.0
    return F32(0.1 * log(Float64(factor)) + 1.0)


# ---------------------------------------------------------------------------
# fp32 GEMV against a ternary weight matrix, parallel over output rows.
#
# Exact, not approximate: each weight is exactly {-1,0,+1} times its group's FP16
# scale, so accumulating (w * x) in fp32 gives the same value the dequantized matrix
# would. Activations stay fp32 here on purpose — M2 is about proving the *graph* is
# right, and quantizing activations would blur the comparison against the oracle.
# ---------------------------------------------------------------------------
def gemv_f32(t: QTensor, x: List[F32], mut y: List[F32], workers: Int):
    var wp = t.w.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var ng = t.groups_per_row()
    var cols = t.cols
    var rows = t.rows

    comptime RB = 16

    def row_block(blk: Int) capturing:
        var r0 = blk * RB
        var r1 = min(r0 + RB, rows)
        for r in range(r0, r1):
            var row = r * cols
            var acc: F32 = 0.0
            for g in range(ng):
                var gsum: F32 = 0.0
                var base = g * TERNARY_GROUP
                for i in range(TERNARY_GROUP):
                    var wv = wp[unsafe_offset = row + base + i]
                    if wv == 1:
                        gsum += xp[unsafe_offset = base + i]
                    elif wv == -1:
                        gsum -= xp[unsafe_offset = base + i]
                acc += gsum * sp[unsafe_offset = r * ng + g]
            yp[unsafe_offset=r] = acc

    parallelize[row_block]((rows + RB - 1) // RB, workers)


# Dequantize one row of a ternary matrix into fp32 — used for embedding lookup.
def embed_row(t: QTensor, row: Int, mut out: List[F32]):
    var ng = t.groups_per_row()
    for g in range(ng):
        var s = t.scales[row * ng + g]
        for i in range(TERNARY_GROUP):
            var c = g * TERNARY_GROUP + i
            out[c] = F32(Int(t.w[row * t.cols + c])) * s


# ---------------------------------------------------------------------------
# M3: the fast path. Same GEMV, int8 activations, NEON sdot.
#
# `sdot` takes an int32x4 accumulator and two int8x16 vectors and adds four
# 4-element dot products into the four lanes — 16 multiply-accumulates per
# instruction, with int32 accumulation free. Dense: every weight participates,
# zeros included (Correction #1).
#
# The activation vector is quantized once per call, which is O(cols) against the
# GEMV's O(rows*cols) — negligible, and it is why `xq` is passed in as scratch
# rather than allocated here.
#
# Accuracy: activations become int8, so this is ~0.3% off the fp32 path rather
# than ~0.005%. Weights are untouched — they were already exactly {-1,0,+1}.
# ---------------------------------------------------------------------------
comptime I32x4 = SIMD[DType.int32, 4]

# Below this many weights, a GEMV is faster single-threaded than paying one
# `parallelize` dispatch (~300us fixed, measured).
#
# Swept on the 1.7B, median of 3 runs, ms/token:
#   2M -> 54.2    everything parallel; dispatch swamps the token
#   4M -> 49.9
#   8M -> 31.1    (single-run readings ranged 29.9-32.6 -- noisy)
#   18M -> 29.6   best, and by far the most repeatable (29.6/29.6/29.8)
#   all serial -> 32.0
#
# At 18M only the 310MB tied LM head parallelizes; every per-layer GEMV runs serial.
# That is the honest shape of single-stream decode here: 1.72GB of weights split into
# ~197 sequential dependent GEMVs, none individually big enough to amortise a
# dispatch. Re-measure per rung -- 4B/8B have larger tensors and the balance moves.
#
# Take three samples before believing a change: a single run once made 8M look like
# 27.8ms and it does not reproduce.
comptime MIN_PARALLEL_ELEMS = 18 * 1024 * 1024


# Quantize activations to int8 in GROUPS OF 128, one scale each.
#
# A single absmax over the whole vector is not good enough, and the reason is worth
# recording: transformer residual streams carry a handful of outlier channels with
# huge magnitudes. One global scale is set by those outliers, and every ordinary
# channel is then crushed toward zero — measured cost was up to 0.76 relative error
# at a checkpoint, with the 3rd and 4th ranked tokens swapping places.
#
# Group-of-128 scales cost almost nothing here because this kernel already reduces
# once per 128 weights to apply the *weight* scale. The activation scale rides along
# in the same multiply.
def quantize_groups(
    x: List[F32], n: Int, mut xq: List[I8], mut ascale: List[F32]
) raises:
    if n % TERNARY_GROUP != 0:
        raise Error("activation length must be a multiple of 128")
    var ngroups = n // TERNARY_GROUP
    for g in range(ngroups):
        var base = g * TERNARY_GROUP
        var amax: F32 = 0.0
        for i in range(TERNARY_GROUP):
            var a = x[base + i]
            if a < 0.0:
                a = -a
            if a > amax:
                amax = a
        if amax == 0.0:
            ascale[g] = 0.0
            for i in range(TERNARY_GROUP):
                xq[base + i] = 0
            continue
        var scale = amax / 127.0
        var inv = 1.0 / scale
        ascale[g] = scale
        for i in range(TERNARY_GROUP):
            var v = x[base + i] * inv
            var r = (v + 0.5) if v >= 0.0 else (v - 0.5)
            var ri = Int(r)
            if ri > 127:
                ri = 127
            if ri < -127:
                ri = -127
            xq[base + i] = I8(ri)


def gemv_sdot(
    t: QTensor,
    x: List[F32],
    mut xq: List[I8],
    mut ascale: List[F32],
    mut y: List[F32],
    workers: Int,
) raises:
    quantize_groups(x, t.cols, xq, ascale)
    var ap = ascale.unsafe_ptr()
    var wp = t.w.unsafe_ptr()
    var sp = t.scales.unsafe_ptr()
    var xp = xq.unsafe_ptr()
    var yp = y.unsafe_ptr()
    var ng = t.groups_per_row()
    var cols = t.cols
    var rows = t.rows

    # One row's dot product, shared by the serial and parallel paths.
    def do_rows(r0: Int, r1: Int) capturing:
        for r in range(r0, r1):
            var row = r * cols
            var acc: F32 = 0.0
            for g in range(ng):
                var base = g * TERNARY_GROUP
                var lanes = I32x4(0)
                var i = 0
                while i < TERNARY_GROUP:
                    lanes = llvm_intrinsic["llvm.aarch64.neon.sdot", I32x4](
                        lanes,
                        wp.unsafe_load[width=16](row + base + i),
                        xp.unsafe_load[width=16](base + i),
                    )
                    i += 16
                # One reduction per group of 128, carrying both the weight scale
                # and this group's activation scale.
                acc += (
                    F32(lanes.reduce_add())
                    * sp[unsafe_offset = r * ng + g]
                    * ap[unsafe_offset=g]
                )
            yp[unsafe_offset=r] = acc

    # A `parallelize` dispatch costs ~300us regardless of how much work it carries.
    # Measured, from the M3 profile: the one 310MB GEMV reached 256 GB/s, while the
    # 2-4MB attention projections managed 9.7 GB/s across 112 calls -- the overhead
    # WAS the token. Single-threaded sdot runs at ~60 GB/s, so any tensor that takes
    # less than the dispatch cost to stream serially is faster left alone. 300us at
    # 60 GB/s is ~18MB, which is the threshold below.
    if rows * cols < MIN_PARALLEL_ELEMS:
        do_rows(0, rows)
        return

    var rb = (rows + workers - 1) // workers
    def row_block(blk: Int) capturing:
        do_rows(blk * rb, min(blk * rb + rb, rows))

    parallelize[row_block]((rows + rb - 1) // rb, workers)
