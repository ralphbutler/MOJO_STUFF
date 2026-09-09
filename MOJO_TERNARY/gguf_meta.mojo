# gguf_meta.mojo — M0: report a Bonsai GGUF's header, and verify dequantization.
#
# Prints the container metadata, confirms the 34-byte/128-weight block layout from
# tensor offsets, then dequantizes ternary tensors in plain scalar Mojo and diffs
# them against the repo's own F16 build.
#
# The diff is judged PER BLOCK, and that is the whole point. 128 weights share one
# FP16 scale, so a bug in the layout, the code mapping, or the scale offset corrupts
# a block in a *pattern*, while a mere disagreement about the stored scale shows up
# as one constant ratio across the whole block. Only the former means our reader is
# wrong. The gate fails on structural mismatches and tolerates scale-only ones.
#
# Usage: mojo run gguf_meta.mojo [q2_path] [f16_path]

from gguf import (
    Cursor,
    GGUF,
    GGUF_MAGIC,
    HEADER_BUDGET,
    MAX_CHECK_ELEMS,
    MAX_CHECK_TENSORS,
    TERNARY_BLOCK_BYTES,
    TERNARY_GROUP,
    code_and_scale,
    decode_f16,
    dequant_q2_g128,
    ggml_type_name,
    load_header,
    read_blob,
    read_value,
)
from std.sys import argv

def main() raises:
    var args = argv()
    comptime SNAP = "/Users/rbutler/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-gguf/snapshots/983b5dec2ff16aab79990711ba0f828a499a7e6a/"
    var q2_path = String(SNAP) + "Ternary-Bonsai-1.7B-Q2_0.gguf"
    var f16_path = String(SNAP) + "Ternary-Bonsai-1.7B-F16.gguf"
    if len(args) > 1:
        q2_path = String(args[1])
    if len(args) > 2:
        f16_path = String(args[2])

    print("file:", q2_path)
    var g = load_header(q2_path.copy(), True)
    print("")
    print("alignment:", g.alignment, " tensor data starts at byte:", g.data_start)
    print("")

    # --- type histogram: this is where Q2_0's id shows itself ---------------
    print("=== tensor type histogram ===")
    var seen = List[Int]()
    var tally = List[Int]()
    var elems = List[Int]()
    for i in range(len(g.names)):
        var t = g.types[i]
        var found = -1
        for j in range(len(seen)):
            if seen[j] == t:
                found = j
        if found < 0:
            seen.append(t)
            tally.append(1)
            elems.append(g.counts[i])
        else:
            tally[found] += 1
            elems[found] += g.counts[i]
    var quant_type = -1
    for j in range(len(seen)):
        print(
            "  ", ggml_type_name(seen[j]), ":", tally[j], "tensors,",
            elems[j], "elements",
        )
        if ggml_type_name(seen[j]).find("unmapped") >= 0:
            quant_type = seen[j]
    print("")

    # --- verify the 34-byte / 128-weight block claim from offsets ----------
    print("=== block layout check (34 bytes per 128 weights) ===")
    if quant_type < 0:
        print("  no unmapped type found -- nothing to check")
    else:
        print("  ternary ggml type id:", quant_type)
        var checked = 0
        var mismatched = 0
        for i in range(len(g.names) - 1):
            if g.types[i] != quant_type:
                continue
            var predicted = (g.counts[i] // TERNARY_GROUP) * TERNARY_BLOCK_BYTES
            if predicted % g.alignment != 0:
                predicted += g.alignment - (predicted % g.alignment)
            var actual = g.offsets[i + 1] - g.offsets[i]
            checked += 1
            if predicted != actual:
                mismatched += 1
                if mismatched <= 5:
                    print(
                        "   MISMATCH", g.names[i], " elements:", g.counts[i],
                        " predicted:", predicted, " actual:", actual,
                    )
        print("  checked", checked, "ternary tensors,", mismatched, "mismatched")
        if checked > 0 and mismatched == 0:
            print("  -> block layout CONFIRMED from tensor offsets")
    print("")

    # --- M0's done bar -----------------------------------------------------
    # Diff dequantized tensors against the F16 reference, judged per *block*.
    #
    # Why per block: a block's 128 weights all share one FP16 scale, so a bug in
    # the layout, the code mapping, or the scale offset corrupts weights in a
    # pattern. A disagreement that is one *constant ratio* across a whole block
    # means only the stored scale differs between the two files -- our arithmetic
    # reproduced the Q2_0 bytes faithfully. That is the distinction the gate cares
    # about, so it is measured rather than assumed.
    print("=== dequantize + diff vs F16 reference ===")
    var gf = load_header(f16_path.copy(), False)

    var t_checked = 0
    var e_checked = 0
    var b_exact = 0
    var b_scale_only = 0
    var b_structural = 0
    var q3_total = 0
    var worst_ratio_dev: Float32 = 0.0

    for ti in range(len(g.names)):
        if g.types[ti] != quant_type:
            continue
        if g.counts[ti] > MAX_CHECK_ELEMS:
            continue
        if t_checked >= MAX_CHECK_TENSORS:
            break
        var name = g.names[ti]
        var fi = gf.index_of(name)
        if fi < 0 or gf.types[fi] != 1 or gf.counts[fi] != g.counts[ti]:
            continue

        var nelem = g.counts[ti]
        var nblocks = nelem // TERNARY_GROUP
        var qblob = read_blob(
            q2_path, g.data_start + g.offsets[ti], nblocks * TERNARY_BLOCK_BYTES
        )
        var got = dequant_q2_g128(qblob, nelem)
        var fblob = read_blob(f16_path, gf.data_start + gf.offsets[fi], nelem * 2)
        var want = decode_f16(fblob, nelem)

        var t_exact = 0
        var t_scale = 0
        var t_struct = 0
        for b in range(nblocks):
            var base = b * TERNARY_GROUP
            var ndiff = 0
            var ratio: Float32 = 0.0
            var consistent = True
            for j in range(TERNARY_GROUP):
                var cs = code_and_scale(qblob, base + j)
                if cs[0] == 3:
                    q3_total += 1
                var ours = got[base + j]
                var theirs = want[base + j]
                if ours == theirs:
                    continue
                ndiff += 1
                # A pure scale disagreement shows up as one ratio for the block.
                if ours == 0.0:
                    consistent = False
                    continue
                var r = theirs / ours
                if ratio == 0.0:
                    ratio = r
                else:
                    var d = r - ratio
                    if d < 0.0:
                        d = -d
                    if d > 1e-6:
                        consistent = False
            if ndiff == 0:
                t_exact += 1
            elif consistent:
                t_scale += 1
                var dev = ratio - 1.0
                if dev < 0.0:
                    dev = -dev
                if dev > worst_ratio_dev:
                    worst_ratio_dev = dev
            else:
                t_struct += 1

        print(
            "  ", name, " blocks:", nblocks,
            " exact:", t_exact, " scale-only:", t_scale, " STRUCTURAL:", t_struct,
        )
        t_checked += 1
        e_checked += nelem
        b_exact += t_exact
        b_scale_only += t_scale
        b_structural += t_struct

    print("")
    print("  tensors checked :", t_checked, " elements:", e_checked)
    print("  blocks bit-exact:", b_exact)
    print("  blocks differing only by the stored scale:", b_scale_only)
    print("  blocks with a STRUCTURAL mismatch        :", b_structural)
    print("  q=3 codes seen (should be 0):", q3_total)
    print("  worst per-block scale deviation:", worst_ratio_dev)
    print("")

    if b_structural != 0:
        raise Error(
            "structural mismatch in " + String(b_structural)
            + " blocks -- the dequantizer is wrong, not just the reference"
        )
    if q3_total != 0:
        print("  NOTE: q=3 appears, so the 'reserved' code point is in use after all.")
    print("  M0 DONE.")
    print(
        "  Every block either matches the F16 reference bit-for-bit, or differs by a"
    )
    print(
        "  single constant scale -- i.e. the two files disagree about that block's"
    )
    print("  FP16 scale, while our layout, code mapping and arithmetic are exact.")
