# turboquant.mojo — end-to-end TurboQuant index: encode + search.
#
# Pipeline (per turbovec, Gaussian-limit codebook):
#   build:  normalize -> rotate (Q) -> quantize per-coord -> bit-pack codes,
#           store per-vector scale = ||v|| / <rot, centroid[code]> so scored
#           inner products are unbiased.
#   search: rotate query -> per-coord LUT (q_rot[d]*centroid[code]) ->
#           score every vector by summing LUT entries of its codes, * scale ->
#           top-k.
#
# TQ+ per-coord calibration is stubbed to identity here (shift=0, scale=1); it
# is added next to recover turbovec's recall edge. Codes are nibble/2bit packed
# sequentially — a self-consistent layout shared by our own encode + search
# (we don't share kernels with turbovec, so we don't need its SIMD-blocked
# layout for the correctness MVP).

from std.math import sqrt
from codebook import Codebook, codebook
from rotation import make_rotation_matrix


struct TurboQuantIndex(Copyable, Movable):
    var dim: Int
    var bits: Int
    var n: Int
    var boundaries: List[Float32]  # L-1
    var centroids: List[Float32]   # L
    var rotation: List[Float32]    # dim*dim, row-major (apply as Q.u)
    var packed: List[UInt8]        # n * bytes_per_row
    var scales: List[Float32]      # n

    def __init__(out self, dim: Int, bits: Int):
        self.dim = dim
        self.bits = bits
        self.n = 0
        self.boundaries = List[Float32]()
        self.centroids = List[Float32]()
        self.rotation = List[Float32]()
        self.packed = List[UInt8]()
        self.scales = List[Float32]()


# code = number of boundaries a value exceeds (monotone bucket index).
def quantize_scalar(x: Float32, boundaries: List[Float32]) -> Int:
    var code = 0
    for j in range(len(boundaries)):
        if x > boundaries[j]:
            code += 1
        else:
            break
    return code


# Rotate a unit row u (length dim) by Q: out[a] = sum_i Q[a*dim+i] * u[i].
def rotate_row(u: List[Float32], rotation: List[Float32], dim: Int, mut out: List[Float32]):
    for a in range(dim):
        var s = Float32(0.0)
        var base = a * dim
        for i in range(dim):
            s += rotation[base + i] * u[i]
        out[a] = s


# Build the index from a flat n*dim row-major Float32 array.
def build_index(vectors: List[Float32], n: Int, dim: Int, bits: Int) -> TurboQuantIndex:
    var idx = TurboQuantIndex(dim, bits)
    idx.n = n

    var cb = codebook(bits, dim)
    idx.boundaries = cb.boundaries.copy()
    idx.centroids = cb.centroids.copy()
    idx.rotation = make_rotation_matrix(dim)

    var codes_per_byte = 8 // bits
    var bytes_per_row = dim // codes_per_byte

    idx.packed = List[UInt8](capacity=n * bytes_per_row)
    for _ in range(n * bytes_per_row):
        idx.packed.append(0)
    idx.scales = List[Float32](capacity=n)
    for _ in range(n):
        idx.scales.append(0.0)

    var u = List[Float32](capacity=dim)
    var rot = List[Float32](capacity=dim)
    for _ in range(dim):
        u.append(0.0)
        rot.append(0.0)

    for vi in range(n):
        var base = vi * dim
        # norm + unit
        var nrm_sq = Float32(0.0)
        for i in range(dim):
            var v = vectors[base + i]
            nrm_sq += v * v
        var nrm = sqrt(nrm_sq)
        var inv = Float32(1.0) / nrm if nrm > 1e-10 else Float32(0.0)
        for i in range(dim):
            u[i] = vectors[base + i] * inv

        rotate_row(u, idx.rotation, dim, rot)

        # quantize + accumulate reconstruction inner product + pack
        var inner = Float32(0.0)
        var row_off = vi * bytes_per_row
        for g in range(bytes_per_row):
            var byte_val: Int = 0
            for c in range(codes_per_byte):
                var d = g * codes_per_byte + c
                var code = quantize_scalar(rot[d], idx.boundaries)
                inner += rot[d] * idx.centroids[code]
                var shift = (codes_per_byte - 1 - c) * bits
                byte_val |= code << shift
            idx.packed[row_off + g] = UInt8(byte_val)

        if inner < 1e-10:
            inner = 1e-10
        idx.scales[vi] = nrm / inner

    return idx^


# Search: brute-force scored scan over all vectors, top-k by inner product.
# Writes top-k indices (length nq*k) and scores (length nq*k). k best per query.
def search(
    idx: TurboQuantIndex,
    queries: List[Float32],
    nq: Int,
    k: Int,
    mut out_idx: List[Int32],
    mut out_score: List[Float32],
):
    var dim = idx.dim
    var bits = idx.bits
    var n = idx.n
    var codes_per_byte = 8 // bits
    var bytes_per_row = dim // codes_per_byte
    var n_levels = 1 << bits
    var code_mask = (1 << bits) - 1

    var uq = List[Float32](capacity=dim)
    var qrot = List[Float32](capacity=dim)
    for _ in range(dim):
        uq.append(0.0)
        qrot.append(0.0)

    # Per-coord LUT: lut[d*n_levels + code] = qrot[d] * centroid[code].
    var lut = List[Float32](capacity=dim * n_levels)
    for _ in range(dim * n_levels):
        lut.append(0.0)

    # top-k scratch
    var best_s = List[Float32](capacity=k)
    var best_i = List[Int32](capacity=k)
    for _ in range(k):
        best_s.append(0.0)
        best_i.append(-1)

    for qi in range(nq):
        var qbase = qi * dim
        # NOTE: queries are searched as raw inner product, so we rotate the raw
        # query (not normalized) — <q, v> ranking is scale-invariant in q.
        for i in range(dim):
            uq[i] = queries[qbase + i]
        rotate_row(uq, idx.rotation, dim, qrot)

        for d in range(dim):
            var qd = qrot[d]
            var lbase = d * n_levels
            for c in range(n_levels):
                lut[lbase + c] = qd * idx.centroids[c]

        var size = 0
        var min_s = Float32(0.0)
        var min_pos = 0

        for vi in range(n):
            var row_off = vi * bytes_per_row
            var score = Float32(0.0)
            for g in range(bytes_per_row):
                var byte_val = Int(idx.packed[row_off + g])
                var dbase = g * codes_per_byte
                for c in range(codes_per_byte):
                    var shift = (codes_per_byte - 1 - c) * bits
                    var code = (byte_val >> shift) & code_mask
                    var d = dbase + c
                    score += lut[d * n_levels + code]
            score *= idx.scales[vi]

            # maintain top-k (max-scores kept; evict current min)
            if size < k:
                best_s[size] = score
                best_i[size] = Int32(vi)
                size += 1
                if size == k:
                    min_s = best_s[0]
                    min_pos = 0
                    for h in range(1, k):
                        if best_s[h] < min_s:
                            min_s = best_s[h]
                            min_pos = h
            elif score > min_s:
                best_s[min_pos] = score
                best_i[min_pos] = Int32(vi)
                min_s = best_s[0]
                min_pos = 0
                for h in range(1, k):
                    if best_s[h] < min_s:
                        min_s = best_s[h]
                        min_pos = h

        var ob = qi * k
        for h in range(k):
            out_idx[ob + h] = best_i[h]
            out_score[ob + h] = best_s[h]
