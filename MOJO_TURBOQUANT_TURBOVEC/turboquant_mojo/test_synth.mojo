# test_synth.mojo — end-to-end synthetic validation of the TurboQuant index.
# Generates random Gaussian vectors + queries, computes exact top-k by raw
# inner product (brute force), then measures recall@k of the quantized index.

from std.math import sqrt
from rotation import SplitMix64
from turboquant import build_index, search, TurboQuantIndex

comptime N = 4000
comptime DIM = 256
comptime NQ = 200
comptime K = 64
comptime BITS = 4


def gen_vectors(n: Int, dim: Int, seed: UInt64) -> List[Float32]:
    var rng = SplitMix64(seed)
    var out = List[Float32](capacity=n * dim)
    for _ in range(n * dim):
        out.append(Float32(rng.next_gaussian()))
    return out^


# Exact top-k indices per query by raw inner product <q, v>. Returns nq*k Int32.
def exact_topk(
    base: List[Float32], n: Int, queries: List[Float32], nq: Int, dim: Int, k: Int
) -> List[Int32]:
    var res = List[Int32](capacity=nq * k)
    for _ in range(nq * k):
        res.append(-1)
    var best_s = List[Float32](capacity=k)
    var best_i = List[Int32](capacity=k)
    for _ in range(k):
        best_s.append(0.0)
        best_i.append(-1)

    for qi in range(nq):
        var qb = qi * dim
        var size = 0
        var min_s = Float32(0.0)
        var min_pos = 0
        for vi in range(n):
            var vb = vi * dim
            var s = Float32(0.0)
            for d in range(dim):
                s += queries[qb + d] * base[vb + d]
            if size < k:
                best_s[size] = s
                best_i[size] = Int32(vi)
                size += 1
                if size == k:
                    min_s = best_s[0]
                    min_pos = 0
                    for h in range(1, k):
                        if best_s[h] < min_s:
                            min_s = best_s[h]
                            min_pos = h
            elif s > min_s:
                best_s[min_pos] = s
                best_i[min_pos] = Int32(vi)
                min_s = best_s[0]
                min_pos = 0
                for h in range(1, k):
                    if best_s[h] < min_s:
                        min_s = best_s[h]
                        min_pos = h
        var ob = qi * k
        for h in range(k):
            res[ob + h] = best_i[h]
    return res^


def main() raises:
    print("== TurboQuant synthetic validation ==")
    print("N =", N, " DIM =", DIM, " NQ =", NQ, " K =", K, " BITS =", BITS)

    var base = gen_vectors(N, DIM, 0x1234)
    var queries = gen_vectors(NQ, DIM, 0x9ABC)

    print("computing exact ground truth (brute force)...")
    var gt = exact_topk(base, N, queries, NQ, DIM, K)

    print("building index...")
    var idx = build_index(base, N, DIM, BITS)

    print("searching...")
    var out_idx = List[Int32](capacity=NQ * K)
    var out_score = List[Float32](capacity=NQ * K)
    for _ in range(NQ * K):
        out_idx.append(-1)
        out_score.append(0.0)
    search(idx, queries, NQ, K, out_idx, out_score)

    # recall@K: fraction of exact top-k found in approx top-k, averaged.
    var total = 0
    for qi in range(NQ):
        var ob = qi * K
        for a in range(K):
            var target = gt[ob + a]
            var found = False
            for b in range(K):
                if out_idx[ob + b] == target:
                    found = True
                    break
            if found:
                total += 1
    var recall = Float32(total) / Float32(NQ * K)
    print("recall@" + String(K) + " =", recall)

    # memory: bytes/vector at BITS bits + f32 scale
    var bytes_per_row = DIM // (8 // BITS)
    print("bytes/vector (codes):", bytes_per_row, " + 4 (scale) =", bytes_per_row + 4)
    print("fp32 original bytes/vector:", DIM * 4)
