# run_dbpedia.mojo — real-data TurboQuant run, SIMD + multicore.
#
# Reads the raw float32 blobs written by compare/fetch_data.py, builds a 4-bit
# TurboQuant index, searches the queries, and writes the top-K indices (sorted
# by score descending) as int32 for Python to score with recall@1-in-top-k,
# matching the FAISS baseline.
#
# Two Mojo gotchas learned the hard way, both handled below:
#   1. rebind-ing a List's pointer to MutUntrackedOrigin severs lifetime
#      tracking, so the backing List can be freed early and its memory reused
#      (silent corruption). We keep the byte buffers live to end of main().
#   2. alloc/free INSIDE parallel workers races; all per-item scratch is
#      pre-allocated (one disjoint slice per work item) instead.

from std.math import sqrt
from std.sys import simd_width_of, num_physical_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize
from std.memory import alloc
from codebook import codebook
from rotation import make_rotation_matrix

comptime f32 = DType.float32
comptime W = simd_width_of[f32]()
comptime Ptr = UnsafePointer[Scalar[f32], MutUntrackedOrigin]

comptime DIM = 1536
comptime N = 100_000
comptime NQ = 1_000
comptime K = 64
comptime BITS = 4
comptime CODES_PER_BYTE = 8 // BITS              # 2
comptime BYTES_PER_ROW = DIM // CODES_PER_BYTE   # 768
comptime N_LEVELS = 1 << BITS                    # 16

comptime DATA = "/Users/rbutler/Desktop/TURBOQUANT/data/"


@always_inline
def dot(a: Ptr, b: Ptr, n: Int) -> Float32:
    var acc = SIMD[f32, W](0)
    var j = 0
    while j + W <= n:
        acc += a.load[width=W](j) * b.load[width=W](j)
        j += W
    var s = acc.reduce_add()
    while j < n:
        s += a[j] * b[j]
        j += 1
    return s


def read_f32(path: String, count: Int) raises -> List[UInt8]:
    var raw = open(path, "r").read_bytes()
    if len(raw) != count * 4:
        raise Error("size mismatch reading " + path)
    return raw^


def main() raises:
    var workers = num_physical_cores()
    print("== TurboQuant (Mojo) on DBpedia-1536 ==")
    print("DIM =", DIM, " N =", N, " NQ =", NQ, " K =", K, " BITS =", BITS,
          " | W =", W, " cores =", workers)

    var base_bytes = read_f32(DATA + "base.f32", N * DIM)
    var query_bytes = read_f32(DATA + "queries.f32", NQ * DIM)
    var base = rebind[Ptr](base_bytes.unsafe_ptr().bitcast[Scalar[f32]]())
    var queries = rebind[Ptr](query_bytes.unsafe_ptr().bitcast[Scalar[f32]]())

    var cb = codebook(BITS, DIM)
    var bnd = rebind[Ptr](cb.boundaries.unsafe_ptr())
    var ctr = rebind[Ptr](cb.centroids.unsafe_ptr())
    var n_bnd = len(cb.boundaries)

    var t0 = perf_counter_ns()
    var rot_list = make_rotation_matrix(DIM)
    var Q = rebind[Ptr](rot_list.unsafe_ptr())
    print("rotation matrix built:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    var packed = alloc[UInt8](N * BYTES_PER_ROW)
    var scales = alloc[Scalar[f32]](N)

    # ---- encode (parallel over vectors; no in-worker allocation) ----
    t0 = perf_counter_ns()

    @parameter
    def enc_worker(vi: Int):
        var vb = base + vi * DIM
        var nrm = sqrt(dot(vb, vb, DIM))
        var inv = Float32(1.0) / nrm if nrm > 1e-10 else Float32(0.0)
        var inner = Float32(0.0)
        var row = packed + vi * BYTES_PER_ROW
        for g in range(BYTES_PER_ROW):
            var byte_val: Int = 0
            for c in range(CODES_PER_BYTE):
                var d = g * CODES_PER_BYTE + c
                var x = dot(Q + d * DIM, vb, DIM) * inv   # rotated coord, on the fly
                var code = 0
                while code < n_bnd and x > bnd[code]:
                    code += 1
                inner += x * ctr[code]
                byte_val |= code << ((CODES_PER_BYTE - 1 - c) * BITS)
            row[g] = UInt8(byte_val)
        if inner < 1e-10:
            inner = 1e-10
        scales[vi] = nrm / inner

    parallelize[enc_worker](N, workers)
    print("encode:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    # ---- search (parallel over queries; pre-allocated per-query scratch) ----
    var out_idx = alloc[Int32](NQ * K)
    var lut_all = alloc[Scalar[f32]](NQ * DIM * N_LEVELS)
    var bestS_all = alloc[Scalar[f32]](NQ * K)
    var bestI_all = alloc[Int32](NQ * K)
    t0 = perf_counter_ns()

    @parameter
    def qry_worker(qi: Int):
        var qb = queries + qi * DIM
        var lut = lut_all + qi * DIM * N_LEVELS
        for d in range(DIM):
            var qd = dot(Q + d * DIM, qb, DIM)
            var lb = d * N_LEVELS
            for c in range(N_LEVELS):
                lut[lb + c] = qd * ctr[c]

        var best_s = bestS_all + qi * K
        var best_i = bestI_all + qi * K
        var size = 0
        var min_s = Float32(0.0)
        var min_pos = 0
        for vi in range(N):
            var row = packed + vi * BYTES_PER_ROW
            var score = Float32(0.0)
            for g in range(BYTES_PER_ROW):
                var bv = Int(row[g])
                var dbase = g * CODES_PER_BYTE
                for c in range(CODES_PER_BYTE):
                    var code = (bv >> ((CODES_PER_BYTE - 1 - c) * BITS)) & (N_LEVELS - 1)
                    score += lut[(dbase + c) * N_LEVELS + code]
            score *= scales[vi]
            if size < K:
                best_s[size] = score
                best_i[size] = Int32(vi)
                size += 1
                if size == K:
                    min_s = best_s[0]
                    min_pos = 0
                    for h in range(1, K):
                        if best_s[h] < min_s:
                            min_s = best_s[h]
                            min_pos = h
            elif score > min_s:
                best_s[min_pos] = score
                best_i[min_pos] = Int32(vi)
                min_s = best_s[0]
                min_pos = 0
                for h in range(1, K):
                    if best_s[h] < min_s:
                        min_s = best_s[h]
                        min_pos = h

        # selection-sort the K results by score descending, into out_idx
        var out = out_idx + qi * K
        for a in range(K):
            var bp = a
            for b in range(a + 1, K):
                if best_s[b] > best_s[bp]:
                    bp = b
            var ts = best_s[a]; best_s[a] = best_s[bp]; best_s[bp] = ts
            var ti = best_i[a]; best_i[a] = best_i[bp]; best_i[bp] = ti
            out[a] = best_i[a]

    parallelize[qry_worker](NQ, workers)
    var t_search = Float64(perf_counter_ns() - t0) / 1e9
    print("search:", t_search, "s  (", 1000.0 * t_search / Float64(NQ), "ms/query )")

    var fh = open(DATA + "results/mojo_topk.i32", "w")
    fh.write_bytes(Span[UInt8, MutUntrackedOrigin](ptr=out_idx.bitcast[UInt8](), length=NQ * K * 4))
    fh.close()
    print("bytes/vector:", BYTES_PER_ROW, "codes + 4 scale =", BYTES_PER_ROW + 4)
    print("wrote results/mojo_topk.i32")

    packed.free()
    scales.free()
    out_idx.free()
    lut_all.free()
    bestS_all.free()
    bestI_all.free()
    # Keep every List whose untracked pointer we used alive to here, so the
    # compiler can't free its backing store early and let another allocation
    # reuse the memory (that aliasing was the root cause of earlier garbage).
    print("(done; retained", base_bytes[0], query_bytes[0],
          cb.boundaries[0], cb.centroids[0], rot_list[0], ")")
