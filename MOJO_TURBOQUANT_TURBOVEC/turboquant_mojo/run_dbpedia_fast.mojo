# run_dbpedia_fast.mojo — TurboQuant with a SIMD FastScan scoring kernel.
#
# Same encode as run_dbpedia.mojo, but scoring uses NEON-style dynamic table
# lookup (SIMD._dynamic_shuffle) over a uint8 LUT to score LANES=16 vectors per
# instruction -- the FAISS FastScan trick. Codes are repacked into a blocked
# layout so 16 vectors' byte-g values are contiguous.
#
# Per-query float sub-LUTs (qrot[d]*centroid[c]) are quantized to uint8 with a
# per-coordinate min subtracted and one shared scale (closed-form: each
# sub-table is qrot[d]*centroids, so its min/span are exact), plus a per-query
# bias. Score = scale * (uint8 accumulation) + bias, then * per-vector scale.

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
comptime U8Ptr = UnsafePointer[UInt8, MutUntrackedOrigin]

comptime DIM = 1536
comptime N = 100_000
comptime NQ = 1_000
comptime K = 64
comptime BITS = 4
comptime CODES_PER_BYTE = 8 // BITS              # 2
comptime BYTES_PER_ROW = DIM // CODES_PER_BYTE   # 768
comptime N_LEVELS = 1 << BITS                    # 16
comptime LANES = 16                              # vectors scored per shuffle
comptime FLUSH_EVERY = 256                        # uint16 acc groups before uint32 flush (256*254<65535)
comptime MR = 4                                   # GEMM microkernel: vectors per register tile
comptime NR = 4                                   # GEMM microkernel: output coords per register tile
comptime N_BLOCKS = (N + LANES - 1) // LANES

comptime DATA = "../data/"                 # relative to turboquant_mojo/ (run the binary from here)


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
    print("== TurboQuant (Mojo, FastScan) on DBpedia-1536 ==")
    print("DIM =", DIM, " N =", N, " NQ =", NQ, " K =", K, " BITS =", BITS,
          " | W =", W, " LANES =", LANES, " cores =", workers)

    var base_bytes = read_f32(DATA + "base.f32", N * DIM)
    var query_bytes = read_f32(DATA + "queries.f32", NQ * DIM)
    var base = rebind[Ptr](base_bytes.unsafe_ptr().bitcast[Scalar[f32]]())
    var queries = rebind[Ptr](query_bytes.unsafe_ptr().bitcast[Scalar[f32]]())

    var cb = codebook(BITS, DIM)
    var bnd = rebind[Ptr](cb.boundaries.unsafe_ptr())
    var ctr = rebind[Ptr](cb.centroids.unsafe_ptr())
    var n_bnd = len(cb.boundaries)
    var c_lo = cb.centroids[0]
    var c_span = cb.centroids[N_LEVELS - 1] - cb.centroids[0]

    var rot_list = make_rotation_matrix(DIM)
    var Q = rebind[Ptr](rot_list.unsafe_ptr())

    var packed = alloc[UInt8](N * BYTES_PER_ROW)
    var scales = alloc[Scalar[f32]](N)

    # ---- encode ----
    var t0 = perf_counter_ns()

    # Pass 0: inverse L2 norms (so the rotation can produce unit-rotated coords).
    var invn = alloc[Scalar[f32]](N)
    var nrms = alloc[Scalar[f32]](N)

    @parameter
    def norm_worker(vi: Int):
        var nrm = sqrt(dot(base + vi * DIM, base + vi * DIM, DIM))
        nrms[vi] = nrm
        invn[vi] = Float32(1.0) / nrm if nrm > 1e-10 else Float32(0.0)

    parallelize[norm_worker](N, workers)

    # Pass 1: batched rotation X_rot = Base @ Q^T as a REGISTER-BLOCKED GEMM.
    #   x_rot[vi,d] = <Q[d,:], base[vi,:]> * invn[vi]
    # Microkernel holds an MR x NR tile of SIMD accumulators. The inner k-loop
    # loads MR base chunks + NR Q chunks (W lanes each) and issues MR*NR FMAs,
    # so every loaded value is reused MR (Q) / NR (base) times in registers.
    # This is what a dot-product loop cannot do (each dot output is independent),
    # and is why the naive per-vector matvec was memory-bound.
    var xrot = alloc[Scalar[f32]](N * DIM)
    var n_vtiles = (N + MR - 1) // MR

    # Requires N % MR == 0 and DIM % NR == 0 (true here: 100000%4, 1536%4) so the
    # microkernel fully unrolls with comptime indices -> accumulators live in
    # registers (a runtime index would force acc back to the stack every FMA).
    @parameter
    def gemm_worker(vtile: Int):
        var v0 = vtile * MR
        var d = 0
        while d < DIM:
            var acc = InlineArray[SIMD[f32, W], MR * NR](fill=SIMD[f32, W](0))
            var k = 0
            while k < DIM:
                var bch = InlineArray[SIMD[f32, W], MR](fill=SIMD[f32, W](0))
                var qch = InlineArray[SIMD[f32, W], NR](fill=SIMD[f32, W](0))
                comptime for i in range(MR):
                    bch[i] = (base + (v0 + i) * DIM).load[width=W](k)
                comptime for j in range(NR):
                    qch[j] = (Q + (d + j) * DIM).load[width=W](k)
                comptime for i in range(MR):
                    comptime for j in range(NR):
                        acc[i * NR + j] += bch[i] * qch[j]
                k += W
            comptime for i in range(MR):
                var iv = invn[v0 + i]
                comptime for j in range(NR):
                    xrot[(v0 + i) * DIM + (d + j)] = acc[i * NR + j].reduce_add() * iv
            d += NR

    parallelize[gemm_worker](n_vtiles, workers)

    # Pass 2: quantize the rotated coords -> codes, bit-pack, and record scale.
    @parameter
    def quant_worker(vi: Int):
        var xr = xrot + vi * DIM
        var inner = Float32(0.0)
        var row = packed + vi * BYTES_PER_ROW
        for g in range(BYTES_PER_ROW):
            var byte_val: Int = 0
            for c in range(CODES_PER_BYTE):
                var x = xr[g * CODES_PER_BYTE + c]
                var code = 0
                while code < n_bnd and x > bnd[code]:
                    code += 1
                inner += x * ctr[code]
                byte_val |= code << ((CODES_PER_BYTE - 1 - c) * BITS)
            row[g] = UInt8(byte_val)
        if inner < 1e-10:
            inner = 1e-10
        scales[vi] = nrms[vi] / inner

    parallelize[quant_worker](N, workers)
    print("encode:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    # ---- repack to blocked layout: blocked[(b*BYTES_PER_ROW + g)*LANES + lane] ----
    var blocked = alloc[UInt8](N_BLOCKS * BYTES_PER_ROW * LANES)
    t0 = perf_counter_ns()

    @parameter
    def repack_worker(b: Int):
        var base_vec = b * LANES
        for g in range(BYTES_PER_ROW):
            var outp = blocked + (b * BYTES_PER_ROW + g) * LANES
            for lane in range(LANES):
                var vi = base_vec + lane
                outp[lane] = packed[vi * BYTES_PER_ROW + g] if vi < N else UInt8(0)

    parallelize[repack_worker](N_BLOCKS, workers)
    print("repack:", Float64(perf_counter_ns() - t0) / 1e9, "s")

    # ---- FastScan search ----
    var out_idx = alloc[Int32](NQ * K)
    var qrot_all = alloc[Scalar[f32]](NQ * DIM)
    var lut8_all = alloc[UInt8](NQ * DIM * N_LEVELS)
    var bestS_all = alloc[Scalar[f32]](NQ * K)
    var bestI_all = alloc[Int32](NQ * K)
    t0 = perf_counter_ns()

    @parameter
    def qry_worker(qi: Int):
        var qb = queries + qi * DIM
        var qrot = qrot_all + qi * DIM
        var maxabs = Float32(0.0)
        for d in range(DIM):
            var qd = dot(Q + d * DIM, qb, DIM)
            qrot[d] = qd
            var a = qd if qd >= 0 else -qd
            if a > maxabs:
                maxabs = a

        # Cap LUT entries at 127 (not 255) so the two nibble sub-tables can be
        # combined in uint8 (127+127=254 < 256) before a single widening add.
        var scale = maxabs * c_span / 127.0
        if scale < 1e-20:
            scale = 1e-20
        var inv_scale = Float32(1.0) / scale

        # build uint8 LUT + bias
        var lut8 = lut8_all + qi * DIM * N_LEVELS
        var bias = Float32(0.0)
        for d in range(DIM):
            var qd = qrot[d]
            var submin = qd * c_lo if qd >= 0 else qd * (c_lo + c_span)  # min over centroids
            bias += submin
            var lb = d * N_LEVELS
            for c in range(N_LEVELS):
                var v = (qd * ctr[c] - submin) * inv_scale
                var iv = Int(v + 0.5)
                if iv < 0:
                    iv = 0
                elif iv > 127:
                    iv = 127
                lut8[lb + c] = UInt8(iv)

        var best_s = bestS_all + qi * K
        var best_i = bestI_all + qi * K
        var size = 0
        var min_s = Float32(0.0)
        var min_pos = 0

        for b in range(N_BLOCKS):
            var acc = SIMD[DType.uint32, LANES](0)
            var blk = blocked + b * BYTES_PER_ROW * LANES
            # FastScan hot loop: accumulate uint8 table lookups in a NARROW
            # uint16 accumulator (half the SIMD register traffic of uint32) and
            # flush to uint32 between fixed-size chunks. Each group adds <=2*255
            # =510; chunks of 128 keep acc16 <= 65280 < 65535. The flush lives
            # BETWEEN the inner runs so the hot inner loop stays branch-free.
            var g0 = 0
            while g0 < BYTES_PER_ROW:
                var g1 = g0 + FLUSH_EVERY
                if g1 > BYTES_PER_ROW:
                    g1 = BYTES_PER_ROW
                var acc16 = SIMD[DType.uint16, LANES](0)
                for g in range(g0, g1):
                    var bytes = blk.load[width=LANES](g * LANES)
                    var hi = bytes >> 4
                    var lo = bytes & 0xF
                    var hLUT = lut8.load[width=LANES]((g * CODES_PER_BYTE) * N_LEVELS)
                    var lLUT = lut8.load[width=LANES]((g * CODES_PER_BYTE + 1) * N_LEVELS)
                    # Combine both nibble sub-tables in uint8 (<=254), then a
                    # single widening add into the uint16 accumulator.
                    var s = hLUT._dynamic_shuffle(hi) + lLUT._dynamic_shuffle(lo)
                    acc16 += s.cast[DType.uint16]()
                acc += acc16.cast[DType.uint32]()
                g0 = g1

            var base_vec = b * LANES
            for lane in range(LANES):
                var vi = base_vec + lane
                if vi >= N:
                    break
                var raw = scale * Float32(acc[lane]) + bias
                var score = raw * scales[vi]
                if size < K:
                    best_s[size] = score
                    best_i[size] = Int32(vi)
                    size += 1
                    if size == K:
                        min_s = best_s[0]; min_pos = 0
                        for h in range(1, K):
                            if best_s[h] < min_s:
                                min_s = best_s[h]; min_pos = h
                elif score > min_s:
                    best_s[min_pos] = score
                    best_i[min_pos] = Int32(vi)
                    min_s = best_s[0]; min_pos = 0
                    for h in range(1, K):
                        if best_s[h] < min_s:
                            min_s = best_s[h]; min_pos = h

        var out = out_idx + qi * K
        for a in range(K):
            var bp = a
            for b2 in range(a + 1, K):
                if best_s[b2] > best_s[bp]:
                    bp = b2
            var ts = best_s[a]; best_s[a] = best_s[bp]; best_s[bp] = ts
            var ti = best_i[a]; best_i[a] = best_i[bp]; best_i[bp] = ti
            out[a] = best_i[a]

    parallelize[qry_worker](NQ, workers)
    var t_search = Float64(perf_counter_ns() - t0) / 1e9
    print("search:", t_search, "s  (", 1000.0 * t_search / Float64(NQ), "ms/query )")

    var fh = open(DATA + "results/mojo_topk.i32", "w")
    fh.write_bytes(Span[UInt8, MutUntrackedOrigin](ptr=out_idx.bitcast[UInt8](), length=NQ * K * 4))
    fh.close()
    print("wrote results/mojo_topk.i32")

    packed.free(); scales.free(); blocked.free()
    invn.free(); nrms.free(); xrot.free()
    out_idx.free(); qrot_all.free(); lut8_all.free(); bestS_all.free(); bestI_all.free()
    print("(done; retained", base_bytes[0], query_bytes[0],
          cb.boundaries[0], cb.centroids[0], rot_list[0], ")")
