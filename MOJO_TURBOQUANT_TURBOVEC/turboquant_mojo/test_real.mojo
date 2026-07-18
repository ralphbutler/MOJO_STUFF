# test_real.mojo — run the VALIDATED List-based pipeline on a real-data subset
# to isolate whether the recall bug is in shared logic or the pointer rewrite.

from std.math import sqrt
from turboquant import build_index, search

comptime DIM = 1536
comptime M = 8000       # database subset
comptime NQt = 200      # query subset
comptime K = 64
comptime BITS = 4
comptime DATA = "../data/"                 # relative to turboquant_mojo/ (run the binary from here)


def load_subset(path: String, count: Int, dim: Int) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var p = raw.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=count * dim)
    for i in range(count * dim):
        out.append(p[i])
    return out^


def main() raises:
    print("== real-data subset via validated List path ==")
    var base = load_subset(DATA + "base.f32", M, DIM)
    var queries = load_subset(DATA + "queries.f32", NQt, DIM)
    print("loaded base", M, "queries", NQt)

    # exact top-1 by raw dot (unit vectors -> cosine)
    var gt = List[Int32](capacity=NQt)
    for _ in range(NQt):
        gt.append(-1)
    for qi in range(NQt):
        var best = Float32(-1e30)
        var bi = -1
        for vi in range(M):
            var s = Float32(0.0)
            for d in range(DIM):
                s += queries[qi * DIM + d] * base[vi * DIM + d]
            if s > best:
                best = s
                bi = vi
        gt[qi] = Int32(bi)

    var idx = build_index(base, M, DIM, BITS)
    var out_idx = List[Int32](capacity=NQt * K)
    var out_score = List[Float32](capacity=NQt * K)
    for _ in range(NQt * K):
        out_idx.append(-1)
        out_score.append(0.0)
    search(idx, queries, NQt, K, out_idx, out_score)

    # recall@1-in-top-k for k in {1,8,64}
    for k in [1, 8, 64]:
        var hit = 0
        for qi in range(NQt):
            var t = gt[qi]
            var found = False
            for b in range(k):
                if out_idx[qi * K + b] == t:
                    found = True
                    break
            if found:
                hit += 1
        print("recall@1-in-top", k, "=", Float32(hit) / Float32(NQt))
