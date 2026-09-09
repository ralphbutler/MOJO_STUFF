# m2.mojo — M2: forward pass for Ternary-Bonsai-1.7B, checked against transformers.
#
# Loads the model, runs the full 28-block graph in fp32, and diffs four checkpoints
# against reference activations dumped by oracle/dump_reference.py.
#
# WHY THE ORACLE IS EXTERNAL: transformers' Qwen3 implementation is authored by
# someone else and is what the model was released against. A numpy reimplementation
# of my own would repeat whatever I misunderstood, and agree with the bug.
#
# Activations are fp32 throughout. M2's job is proving the *graph* is right; int8
# activation quantization would blur the comparison and is M3's business.
#
# Architecture (GGUF metadata + upstream config.json, both checked):
#   28 blocks, all full attention. hidden 2048, FFN 6144.
#   16 query heads over 8 KV heads (GQA 2:1), head_dim 128.
#   Pre-norm RMSNorm (eps 1e-6), SwiGLU, no biases.
#   QK-norm: per-head RMSNorm on q and k, weight length 128.
#   Tied embeddings — no output.weight; the LM head reuses token_embd.
#   RoPE: YaRN. inv_freq and attention_scaling are READ FROM THE ORACLE rather
#   than recomputed, so YaRN's ramp math cannot be a source of divergence here.
#
# Usage: mojo run m2.mojo [gguf-path]

from gguf import load_tokens
from tokenizer import ByteDecoder, decode
from model import (
    F32,
    I8,
    Model,
    QTensor,
    TERNARY_GROUP,
    embed_row,
    gemv_f32,
    gemv_sdot,
    load_model,
    yarn_attention_scaling,
    yarn_inv_freq,
)
from std.math import cos, exp, sin, sqrt
from std.memory import bitcast
from std.sys import argv, num_physical_cores
from std.time import perf_counter_ns

comptime ORACLE = "/Users/rbutler/Desktop/DEMO/MOJO_TERNARY/oracle/ref/"


def read_f32_bin(path: String, n: Int) raises -> List[F32]:
    var f = open(path, "r")
    var b = f.read_bytes(n * 4)
    if len(b) != n * 4:
        raise Error(
            "expected " + String(n * 4) + " bytes from " + path
            + ", got " + String(len(b)) + " -- run oracle/dump_reference.py first"
        )
    var out = List[F32](length=n, fill=0)
    var bp = b.unsafe_ptr()
    for i in range(n):
        var bits: UInt32 = 0
        for k in range(4):
            bits |= UInt32(bp[unsafe_offset = i * 4 + k]) << UInt32(8 * k)
        out[i] = bitcast[DType.float32](bits)
    return out^


# RMSNorm as Qwen3 computes it: normalise by RMS over the vector, then scale by the
# learned weight. No mean subtraction, no bias.
def rmsnorm(x: List[F32], w: List[F32], eps: F32, mut out: List[F32]):
    var n = len(x)
    var ss: F32 = 0.0
    for i in range(n):
        ss += x[i] * x[i]
    var inv = 1.0 / sqrt(ss / F32(n) + eps)
    for i in range(n):
        out[i] = x[i] * inv * w[i]


# In-place RMSNorm over one head's slice — this is the QK-norm.
def rmsnorm_head(mut v: List[F32], off: Int, n: Int, w: List[F32], eps: F32):
    var ss: F32 = 0.0
    for i in range(n):
        ss += v[off + i] * v[off + i]
    var inv = 1.0 / sqrt(ss / F32(n) + eps)
    for i in range(n):
        v[off + i] = v[off + i] * inv * w[i]


def silu(x: F32) -> F32:
    return x / (1.0 + exp(-x))


# transformers' "rotate_half" convention: the vector splits in two halves and the
# second half rotates into the first. NOT the interleaved GPT-NeoX layout — getting
# this backwards produces plausible-looking garbage.
def apply_rope(
    mut v: List[F32],
    off: Int,
    head_dim: Int,
    cos_tab: List[F32],
    sin_tab: List[F32],
    pos: Int,
):
    var half = head_dim // 2
    var c0 = pos * head_dim
    for i in range(half):
        var a = v[off + i]
        var b = v[off + half + i]
        v[off + i] = a * cos_tab[c0 + i] - b * sin_tab[c0 + i]
        v[off + half + i] = (
            b * cos_tab[c0 + half + i] + a * sin_tab[c0 + half + i]
        )


def max_abs_rel(expected: List[F32], got: List[F32], floor: F32) -> F32:
    var worst: F32 = 0.0
    for i in range(len(expected)):
        var m = expected[i] if expected[i] >= 0.0 else -expected[i]
        if m < floor:
            continue
        var d = got[i] - expected[i]
        if d < 0.0:
            d = -d
        var rel = d / m
        if rel > worst:
            worst = rel
    return worst


def peak(v: List[F32]) -> F32:
    var m: F32 = 0.0
    for i in range(len(v)):
        var a = v[i] if v[i] >= 0.0 else -v[i]
        if a > m:
            m = a
    return m


def main() raises:
    var args = argv()
    comptime SNAP = "/Users/rbutler/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-gguf/snapshots/983b5dec2ff16aab79990711ba0f828a499a7e6a/"
    var path = String(SNAP) + "Ternary-Bonsai-1.7B-Q2_0.gguf"
    if len(args) > 1:
        path = String(args[1])
    var workers = num_physical_cores()
    # M3: int8 sdot path by default; pass "fp32" as the second argument to run the
    # slow reference graph instead. Both are kept so the accuracy cost of int8
    # activations can be measured rather than assumed.
    var use_sdot = True
    if len(args) > 2:
        use_sdot = String(args[2]) != "fp32"

    # The prompt "The capital of France is", tokenized by the oracle. Encoding is
    # the half of the tokenizer that needs a regex Mojo does not have, so prompt ids
    # come from outside for now; decoding below is pure Mojo.
    var seq: List[Int] = [785, 6722, 315, 9625, 374]
    var T = len(seq)
    comptime N_GEN = 24
    comptime EOS = 151645
    var MAXLEN = T + N_GEN

    print("M2 — forward pass, Ternary-Bonsai-1.7B")
    print("  prompt: \"The capital of France is\"  ->", T, "tokens")
    var t0 = perf_counter_ns()
    var m = load_model(path.copy(), workers, False)
    print("  loaded in", Float64(perf_counter_ns() - t0) / 1e6, "ms")
    print("  kernel :", "int8 sdot" if use_sdot else "fp32 scalar reference")
    print(
        "  arch   :", m.arch, " layers", m.n_layer, " d_model", m.d_model,
        " d_ffn", m.d_ffn,
    )
    print(
        "  heads  :", m.n_head, "q /", m.n_head_kv, "kv  (GQA",
        m.n_head // m.n_head_kv, ": 1)  head_dim", m.head_dim,
    )
    print(
        "  head   :", "tied to token_embd" if m.tied_head else "separate output.weight",
        " vocab", m.vocab,
    )
    print(
        "  rope   : base", m.rope_base, " yarn factor", m.rope_factor,
        " orig_ctx", m.rope_orig_ctx,
    )

    var D = m.d_model
    var H = m.n_head
    var HKV = m.n_head_kv
    var HD = m.head_dim
    var GQA = H // HKV
    var DKV = HKV * HD
    var eps = m.rms_eps

    # --- RoPE tables, from YaRN computed in-engine ---------------------------
    # M2 read these from the oracle's dump. That does not survive changing rungs:
    # the 8B's original_max_position_embeddings is 16384 against the 1.7B's 8192,
    # which moves the YaRN ramp and so changes every frequency. Now computed from
    # metadata, verified against the 1.7B dump to 1.1e-07 (fp32 rounding).
    var inv_freq = yarn_inv_freq(HD, m.rope_base, m.rope_factor, m.rope_orig_ctx)
    var ATT_SCALING = yarn_attention_scaling(m.rope_factor)
    print("  yarn   : attention_scaling", ATT_SCALING)
    print("")
    var cos_tab = List[F32](length=MAXLEN * HD, fill=0)
    var sin_tab = List[F32](length=MAXLEN * HD, fill=0)
    for t in range(MAXLEN):
        for i in range(HD // 2):
            var ang = F32(t) * inv_freq[i]
            var c = cos(ang) * ATT_SCALING
            var s = sin(ang) * ATT_SCALING
            cos_tab[t * HD + i] = c
            cos_tab[t * HD + HD // 2 + i] = c
            sin_tab[t * HD + i] = s
            sin_tab[t * HD + HD // 2 + i] = s

    # --- KV cache: [layer][pos][kv_head * head_dim] ------------------------
    var kcache = List[F32](length=m.n_layer * MAXLEN * DKV, fill=0)
    var vcache = List[F32](length=m.n_layer * MAXLEN * DKV, fill=0)

    # checkpoints, to diff against the oracle
    var ck_embed = List[F32](length=T * D, fill=0)
    var ck_l0 = List[F32](length=T * D, fill=0)
    var ck_l1 = List[F32](length=T * D, fill=0)
    var ck_final = List[F32](length=T * D, fill=0)
    var last_logits = List[F32](length=m.vocab, fill=0)
    var first_logits = List[F32](length=m.vocab, fill=0)   # kept for the oracle diff

    # scratch
    var x = List[F32](length=D, fill=0)
    var xn = List[F32](length=D, fill=0)
    var q = List[F32](length=D, fill=0)
    var k = List[F32](length=DKV, fill=0)
    var v = List[F32](length=DKV, fill=0)
    var attn = List[F32](length=D, fill=0)
    var proj = List[F32](length=D, fill=0)
    var gate = List[F32](length=m.d_ffn, fill=0)
    var up = List[F32](length=m.d_ffn, fill=0)
    var scores = List[F32](length=MAXLEN, fill=0)
    # scratch for int8 activations — sized for the widest GEMV input (the FFN's 6144)
    var xq8 = List[I8](length=m.d_ffn, fill=0)
    var ascale = List[F32](length=m.d_ffn // TERNARY_GROUP, fill=0)

    # --- profile accumulators (ns) ------------------------------------------
    # "Profile before touching anything." These split the token into the pieces
    # that could plausibly dominate, so the next optimisation is aimed rather than
    # guessed at.
    var ns_gemv_attn: Int = 0      # q, k, v, output projections
    var ns_gemv_ffn: Int = 0       # gate, up, down
    var ns_gemv_logits: Int = 0    # the tied LM head, 310MB per token
    var ns_attention: Int = 0      # QK-norm, RoPE, softmax, KV gather
    var ns_norm_misc: Int = 0      # RMSNorm, SwiGLU elementwise, residuals

    var scale_qk = 1.0 / sqrt(F32(HD))
    var tf0 = perf_counter_ns()

    var stopped = False
    for t in range(MAXLEN):
        if stopped:
            break
        embed_row(m.token_embd, seq[t], x)
        if t < T:
            for i in range(D):
                ck_embed[t * D + i] = x[i]

        for l in range(m.n_layer):
            # ---- attention block ----
            var p0 = perf_counter_ns()
            rmsnorm(x, m.layers[l].attn_norm.v, eps, xn)
            ns_norm_misc += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            if use_sdot:
                gemv_sdot(m.layers[l].attn_q, xn, xq8, ascale, q, workers)
                gemv_sdot(m.layers[l].attn_k, xn, xq8, ascale, k, workers)
                gemv_sdot(m.layers[l].attn_v, xn, xq8, ascale, v, workers)
            else:
                gemv_f32(m.layers[l].attn_q, xn, q, workers)
                gemv_f32(m.layers[l].attn_k, xn, k, workers)
                gemv_f32(m.layers[l].attn_v, xn, v, workers)
            ns_gemv_attn += perf_counter_ns() - p0

            p0 = perf_counter_ns()
            # QK-norm per head, then RoPE. Order matters: normalise, then rotate.
            for h in range(H):
                rmsnorm_head(q, h * HD, HD, m.layers[l].attn_q_norm.v, eps)
                apply_rope(q, h * HD, HD, cos_tab, sin_tab, t)
            for h in range(HKV):
                rmsnorm_head(k, h * HD, HD, m.layers[l].attn_k_norm.v, eps)
                apply_rope(k, h * HD, HD, cos_tab, sin_tab, t)

            var kbase = l * MAXLEN * DKV + t * DKV
            for i in range(DKV):
                kcache[kbase + i] = k[i]
                vcache[kbase + i] = v[i]

            # ---- causal attention, GQA ----
            for h in range(H):
                var kvh = h // GQA
                var best: F32 = -1e30
                for j in range(t + 1):
                    var dot: F32 = 0.0
                    var koff = l * MAXLEN * DKV + j * DKV + kvh * HD
                    for d in range(HD):
                        dot += q[h * HD + d] * kcache[koff + d]
                    var sc = dot * scale_qk
                    scores[j] = sc
                    if sc > best:
                        best = sc
                var denom: F32 = 0.0
                for j in range(t + 1):
                    scores[j] = exp(scores[j] - best)
                    denom += scores[j]
                for d in range(HD):
                    attn[h * HD + d] = 0.0
                for j in range(t + 1):
                    var wgt = scores[j] / denom
                    var voff = l * MAXLEN * DKV + j * DKV + kvh * HD
                    for d in range(HD):
                        attn[h * HD + d] += wgt * vcache[voff + d]

            ns_attention += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            if use_sdot:
                gemv_sdot(m.layers[l].attn_output, attn, xq8, ascale, proj, workers)
            else:
                    gemv_f32(m.layers[l].attn_output, attn, proj, workers)
            ns_gemv_attn += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            for i in range(D):
                x[i] += proj[i]
            ns_norm_misc += perf_counter_ns() - p0

            # ---- FFN: SwiGLU ----
            p0 = perf_counter_ns()
            rmsnorm(x, m.layers[l].ffn_norm.v, eps, xn)
            ns_norm_misc += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            if use_sdot:
                gemv_sdot(m.layers[l].ffn_gate, xn, xq8, ascale, gate, workers)
                gemv_sdot(m.layers[l].ffn_up, xn, xq8, ascale, up, workers)
            else:
                gemv_f32(m.layers[l].ffn_gate, xn, gate, workers)
                gemv_f32(m.layers[l].ffn_up, xn, up, workers)
            ns_gemv_ffn += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            for i in range(m.d_ffn):
                gate[i] = silu(gate[i]) * up[i]
            ns_norm_misc += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            if use_sdot:
                gemv_sdot(m.layers[l].ffn_down, gate, xq8, ascale, proj, workers)
            else:
                gemv_f32(m.layers[l].ffn_down, gate, proj, workers)
            ns_gemv_ffn += perf_counter_ns() - p0
            p0 = perf_counter_ns()
            for i in range(D):
                x[i] += proj[i]
            ns_norm_misc += perf_counter_ns() - p0

            if t < T:
                if l == 0:
                    for i in range(D):
                        ck_l0[t * D + i] = x[i]
                elif l == 1:
                    for i in range(D):
                        ck_l1[t * D + i] = x[i]

        rmsnorm(x, m.output_norm.v, eps, xn)
        if t < T:
            for i in range(D):
                ck_final[t * D + i] = xn[i]

        # Logits are only needed once the prompt is consumed: from t = T-1 onward
        # every step predicts the next token. Greedy (argmax) so the run is
        # deterministic and can be diffed against the oracle.
        if t >= T - 1 and t < MAXLEN - 1:
            var pl0 = perf_counter_ns()
            if use_sdot:
                if m.tied_head:
                    gemv_sdot(m.token_embd, xn, xq8, ascale, last_logits, workers)
                else:
                    gemv_sdot(m.output_head, xn, xq8, ascale, last_logits, workers)
            else:
                if m.tied_head:
                    gemv_f32(m.token_embd, xn, last_logits, workers)
                else:
                    gemv_f32(m.output_head, xn, last_logits, workers)
            ns_gemv_logits += perf_counter_ns() - pl0
            var bi = 0
            var bv: F32 = -1e30
            for i in range(m.vocab):
                if last_logits[i] > bv:
                    bv = last_logits[i]
                    bi = i
            if t == T - 1:
                for i in range(m.vocab):
                    first_logits[i] = last_logits[i]
            seq.append(bi)
            if bi == EOS:
                stopped = True

    var gen_ms = Float64(perf_counter_ns() - tf0) / 1e6
    print("  forward:", gen_ms, "ms for", len(seq), "tokens  =",
          gen_ms / Float64(len(seq)), "ms/token")
    var tot = ns_gemv_attn + ns_gemv_ffn + ns_gemv_logits + ns_attention + ns_norm_misc
    print("")
    print("=== profile (per token) ===")
    var nt = Float64(len(seq))
    print("  GEMV attention (q,k,v,o) :", Float64(ns_gemv_attn) / 1e6 / nt, "ms")
    print("  GEMV ffn (gate,up,down)  :", Float64(ns_gemv_ffn) / 1e6 / nt, "ms")
    print(
        "  GEMV logits (" + ("tied" if m.tied_head else "untied") + " head)  :",
        Float64(ns_gemv_logits) / 1e6 / nt, "ms",
    )
    print("  attention (norm/rope/sm) :", Float64(ns_attention) / 1e6 / nt, "ms")
    print("  rmsnorm/swiglu/residual  :", Float64(ns_norm_misc) / 1e6 / nt, "ms")
    print("  ---- accounted           :", Float64(tot) / 1e6 / nt, "ms of",
          gen_ms / nt, "ms")
    # Weight bytes touched per token: every ternary weight is read exactly once.
    # Derived from the model, not hardcoded -- the 1.7B constant that used to live
    # here silently misreported the 8B by 5x.
    var wcount = m.token_embd.rows * m.token_embd.cols
    if not m.tied_head:
        wcount += m.output_head.rows * m.output_head.cols
    for l in range(len(m.layers)):
        wcount += m.layers[l].attn_q.rows * m.layers[l].attn_q.cols
        wcount += m.layers[l].attn_k.rows * m.layers[l].attn_k.cols
        wcount += m.layers[l].attn_v.rows * m.layers[l].attn_v.cols
        wcount += m.layers[l].attn_output.rows * m.layers[l].attn_output.cols
        wcount += m.layers[l].ffn_gate.rows * m.layers[l].ffn_gate.cols
        wcount += m.layers[l].ffn_up.rows * m.layers[l].ffn_up.cols
        wcount += m.layers[l].ffn_down.rows * m.layers[l].ffn_down.cols
    var wbytes = Float64(wcount)
    print("  weight bytes/token       :", wbytes / 1e9, "GB")
    print("  GEMV effective bandwidth :",
          wbytes / (Float64(ns_gemv_attn + ns_gemv_ffn + ns_gemv_logits) / nt), "GB/s")
    print("")

    # --- diff against the oracle, when one exists for this model ------------
    # The dumps in oracle/ref/ are 1.7B-shaped. Other rungs have no dump unless one
    # is generated, so a missing or wrong-sized file is a skip, not a failure.
    print("=== vs transformers reference ===")
    try:
        var ref_embed = read_f32_bin(String(ORACLE) + "embeddings.bin", T * D)
        var ref_l0 = read_f32_bin(String(ORACLE) + "after_layer0.bin", T * D)
        var ref_l1 = read_f32_bin(String(ORACLE) + "after_layer1.bin", T * D)
        var ref_fin = read_f32_bin(String(ORACLE) + "final_hidden.bin", T * D)
        print("  embeddings   max rel:", max_abs_rel(ref_embed, ck_embed, peak(ref_embed) * 0.01))
        print("  after layer0 max rel:", max_abs_rel(ref_l0, ck_l0, peak(ref_l0) * 0.01))
        print("  after layer1 max rel:", max_abs_rel(ref_l1, ck_l1, peak(ref_l1) * 0.01))
        print("  final hidden max rel:", max_abs_rel(ref_fin, ck_final, peak(ref_fin) * 0.01))
    except e:
        print("  skipped — no reference dump matching this model's shape.")
        print("  (oracle/ref/ holds 1.7B dumps; regenerate for another rung.)")
    print("")

    # --- top-5 at the first generated position, vs the oracle ---------------
    print("=== top-5 next token after the prompt (oracle: ' Paris' 17.697) ===")
    for rank in range(5):
        var bi = 0
        var bv: F32 = -1e30
        for i in range(m.vocab):
            if first_logits[i] > bv:
                bv = first_logits[i]
                bi = i
        print("   rank", rank, " id", bi, " logit", bv)
        first_logits[bi] = -1e31       # knock it out so the next pass finds the runner-up
    print("")

    # --- decode, in pure Mojo ----------------------------------------------
    var vocab = load_tokens(path.copy())
    var dec = ByteDecoder()
    var prompt_ids = List[Int]()
    for i in range(T):
        prompt_ids.append(seq[i])
    var gen_ids = List[Int]()
    for i in range(T, len(seq)):
        gen_ids.append(seq[i])
    print("=== generated ===")
    print("  prompt   :", repr(decode(dec, vocab, prompt_ids)))
    print("  continues:", repr(decode(dec, vocab, gen_ids)))
    print("")
    print("  full     :", decode(dec, vocab, seq))
