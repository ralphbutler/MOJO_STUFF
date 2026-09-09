# gguf.mojo — shared GGUF container reader for this project.
#
# Parses headers, locates tensors, and dequantizes Q2_0 (g128) blocks. Imported by
# `gguf_meta.mojo` (the M0 report + verification gate) and `q2_gemv.mojo` (the M1
# kernel), so the parser exists once.
#
# Format facts verified against Ternary-Bonsai-1.7B-Q2_0.gguf on 2026-09-08:
#   ternary ggml type id = 42 (undocumented, absent from mainline ggml)
#   34 bytes per 128 weights: FP16 scale + 32 bytes of 2-bit codes
#   w = (q - 1) * scale, q in {0,1,2}; q = 3 is reserved and never observed

from std.memory import bitcast
from std.sys import argv

comptime GGUF_MAGIC: UInt32 = 0x46554747        # "GGUF" as little-endian u32
comptime HEADER_BUDGET = 64 * 1024 * 1024       # metadata is a few MB; 64MB is slack
comptime TERNARY_GROUP = 128                    # weights per Q2_0 block
comptime TERNARY_BLOCK_BYTES = 34               # 2-byte FP16 scale + 32 bytes of codes
comptime MAX_CHECK_TENSORS = 16                 # how many ternary tensors to diff
comptime MAX_CHECK_ELEMS = 16 * 1024 * 1024     # skip the giant embedding tensor

# GGUF metadata value type tags.
comptime GT_UINT8 = 0
comptime GT_INT8 = 1
comptime GT_UINT16 = 2
comptime GT_INT16 = 3
comptime GT_UINT32 = 4
comptime GT_INT32 = 5
comptime GT_FLOAT32 = 6
comptime GT_BOOL = 7
comptime GT_STRING = 8
comptime GT_ARRAY = 9
comptime GT_UINT64 = 10
comptime GT_INT64 = 11
comptime GT_FLOAT64 = 12


# Names for the ggml type ids we expect to meet. Anything unmapped prints as its
# raw number, which is exactly how we find Q2_0.
def ggml_type_name(t: Int) -> String:
    if t == 0:
        return "F32"
    if t == 1:
        return "F16"
    if t == 2:
        return "Q4_0"
    if t == 3:
        return "Q4_1"
    if t == 6:
        return "Q5_0"
    if t == 7:
        return "Q5_1"
    if t == 8:
        return "Q8_0"
    if t == 9:
        return "Q8_1"
    if t == 10:
        return "Q2_K"
    if t == 11:
        return "Q3_K"
    if t == 12:
        return "Q4_K"
    if t == 13:
        return "Q5_K"
    if t == 14:
        return "Q6_K"
    if t == 15:
        return "Q8_K"
    if t == 30:
        return "BF16"
    return "type#" + String(t) + " (unmapped)"


# ---------------------------------------------------------------------------
# Byte cursor. GGUF is little-endian; the shifts below are explicit about it
# rather than trusting host order.
# ---------------------------------------------------------------------------
struct Cursor(Movable):
    var buf: List[UInt8]
    var pos: Int

    def __init__(out self, var buf: List[UInt8]):
        self.buf = buf^
        self.pos = 0

    def need(self, n: Int) raises:
        if self.pos + n > len(self.buf):
            raise Error(
                "ran past the buffer at offset " + String(self.pos)
                + " wanting " + String(n) + " bytes; raise HEADER_BUDGET"
            )

    def u8(mut self) raises -> UInt8:
        self.need(1)
        var v = self.buf[self.pos]
        self.pos += 1
        return v

    def u16(mut self) raises -> UInt16:
        self.need(2)
        var v = UInt16(self.buf[self.pos]) | (UInt16(self.buf[self.pos + 1]) << 8)
        self.pos += 2
        return v

    def u32(mut self) raises -> UInt32:
        self.need(4)
        var v: UInt32 = 0
        for k in range(4):
            v |= UInt32(self.buf[self.pos + k]) << UInt32(8 * k)
        self.pos += 4
        return v

    def u64(mut self) raises -> UInt64:
        self.need(8)
        var v: UInt64 = 0
        for k in range(8):
            v |= UInt64(self.buf[self.pos + k]) << UInt64(8 * k)
        self.pos += 8
        return v

    def f32(mut self) raises -> Float32:
        return bitcast[DType.float32](self.u32())

    def f64(mut self) raises -> Float64:
        return bitcast[DType.float64](self.u64())

    # GGUF string: u64 byte length, then raw UTF-8. Not NUL-terminated.
    def gstring(mut self) raises -> String:
        var n = Int(self.u64())
        self.need(n)
        var out = String(
            StringSlice(unsafe_from_utf8=Span(self.buf)[self.pos : self.pos + n])
        )
        self.pos += n
        return out

    def skip(mut self, n: Int) raises:
        self.need(n)
        self.pos += n


# One metadata value, rendered for printing. Long arrays are summarised rather
# than dumped -- the token list alone is 248320 entries.
def read_value(mut c: Cursor, vtype: Int) raises -> String:
    if vtype == GT_UINT8:
        return String(c.u8())
    if vtype == GT_INT8:
        return String(Int8(bitcast[DType.int8](c.u8())))
    if vtype == GT_UINT16:
        return String(c.u16())
    if vtype == GT_INT16:
        return String(Int16(bitcast[DType.int16](c.u16())))
    if vtype == GT_UINT32:
        return String(c.u32())
    if vtype == GT_INT32:
        return String(Int32(bitcast[DType.int32](c.u32())))
    if vtype == GT_UINT64:
        return String(c.u64())
    if vtype == GT_INT64:
        return String(Int64(bitcast[DType.int64](c.u64())))
    if vtype == GT_FLOAT32:
        return String(c.f32())
    if vtype == GT_FLOAT64:
        return String(c.f64())
    if vtype == GT_BOOL:
        return "true" if c.u8() != 0 else "false"
    if vtype == GT_STRING:
        var s = c.gstring()
        if s.byte_length() > 72:
            return "\"" + String(StringSlice(unsafe_from_utf8=Span(s.as_bytes())[0:72])) + "...\""
        return "\"" + s + "\""
    if vtype == GT_ARRAY:
        var etype = Int(c.u32())
        var n = Int(c.u64())
        var shown = String("")
        var limit = 3 if n > 4 else n
        for i in range(limit):
            if i > 0:
                shown += ", "
            shown += read_value(c, etype)
        # Still have to walk the remainder so the cursor lands correctly.
        for _ in range(n - limit):
            _ = read_value(c, etype)
        var tail = ", ..." if n > limit else ""
        return "[" + shown + tail + "]  (" + String(n) + " x " + type_tag(etype) + ")"
    raise Error("unknown GGUF value type: " + String(vtype))


def type_tag(t: Int) -> String:
    if t == GT_UINT8:
        return "u8"
    if t == GT_INT8:
        return "i8"
    if t == GT_UINT16:
        return "u16"
    if t == GT_INT16:
        return "i16"
    if t == GT_UINT32:
        return "u32"
    if t == GT_INT32:
        return "i32"
    if t == GT_FLOAT32:
        return "f32"
    if t == GT_BOOL:
        return "bool"
    if t == GT_STRING:
        return "str"
    if t == GT_ARRAY:
        return "array"
    if t == GT_UINT64:
        return "u64"
    if t == GT_INT64:
        return "i64"
    if t == GT_FLOAT64:
        return "f64"
    return "?" + String(t)


# ---------------------------------------------------------------------------
# Parsed header. Kept so the report and the dequant check share one parser.
# ---------------------------------------------------------------------------
# Scalar metadata (ints, floats, strings). Arrays are walked past, not stored --
# the only array we ever want is the token list, and `load_tokens` handles that.
struct Meta(Movable):
    var keys: List[String]
    var kinds: List[Int]
    var ivals: List[Int]
    var fvals: List[Float64]
    var svals: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.kinds = List[Int]()
        self.ivals = List[Int]()
        self.fvals = List[Float64]()
        self.svals = List[String]()

    def find(self, key: String) -> Int:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return i
        return -1

    def get_int(self, key: String, fallback: Int) -> Int:
        var i = self.find(key)
        return self.ivals[i] if i >= 0 else fallback

    def get_f64(self, key: String, fallback: Float64) -> Float64:
        var i = self.find(key)
        if i < 0:
            return fallback
        # Integer-typed keys still answer sensibly when asked as a float.
        return self.fvals[i] if self.kinds[i] == 6 or self.kinds[i] == 12 else Float64(self.ivals[i])

    def get_str(self, key: String, fallback: String) -> String:
        var i = self.find(key)
        return self.svals[i] if i >= 0 else fallback

    def require_int(self, key: String) raises -> Int:
        var i = self.find(key)
        if i < 0:
            raise Error("required metadata key missing: " + key)
        return self.ivals[i]


struct GGUF(Movable):
    var path: String
    var version: Int
    var alignment: Int
    var data_start: Int
    var names: List[String]
    var types: List[Int]
    var offsets: List[Int]
    var counts: List[Int]
    var ne0: List[Int]          # fastest-varying axis: input features / columns
    var ne1: List[Int]          # next axis: output features / rows (1 if 1-D)
    var meta: Meta

    def __init__(out self, var path: String):
        self.path = path^
        self.version = 0
        self.alignment = 32
        self.data_start = 0
        self.names = List[String]()
        self.types = List[Int]()
        self.offsets = List[Int]()
        self.counts = List[Int]()
        self.ne0 = List[Int]()
        self.ne1 = List[Int]()
        self.meta = Meta()

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1


def load_header(var path: String, verbose: Bool) raises -> GGUF:
    var g = GGUF(path.copy())
    var f = open(path, "r")
    var head = f.read_bytes(HEADER_BUDGET)
    var c = Cursor(head^)

    var magic = c.u32()
    if magic != GGUF_MAGIC:
        raise Error("not a GGUF file: magic was " + String(magic))
    g.version = Int(c.u32())
    var n_tensors = Int(c.u64())
    var n_kv = Int(c.u64())
    if verbose:
        print("header bytes read:", HEADER_BUDGET)
        print(
            "magic: GGUF  version:", g.version, " tensors:", n_tensors,
            " kv pairs:", n_kv,
        )
        print("")
        print("=== metadata ===")

    for _ in range(n_kv):
        var key = c.gstring()
        var vtype = Int(c.u32())
        var pos_before = c.pos
        var rendered = read_value(c, vtype)
        if verbose:
            print("  ", key, "=", rendered)

        # Re-read scalars from their original position so they are available typed,
        # not just as the rendered string.
        if vtype != 9:
            var save = c.pos
            c.pos = pos_before
            var iv = 0
            var fv = 0.0
            var sv = String("")
            if vtype == 0 or vtype == 7:
                iv = Int(c.u8())
            elif vtype == 1:
                iv = Int(Int8(bitcast[DType.int8](c.u8())))
            elif vtype == 2:
                iv = Int(c.u16())
            elif vtype == 3:
                iv = Int(Int16(bitcast[DType.int16](c.u16())))
            elif vtype == 4:
                iv = Int(c.u32())
            elif vtype == 5:
                iv = Int(Int32(bitcast[DType.int32](c.u32())))
            elif vtype == 10:
                iv = Int(c.u64())
            elif vtype == 11:
                iv = Int(Int64(bitcast[DType.int64](c.u64())))
            elif vtype == 6:
                fv = Float64(c.f32())
            elif vtype == 12:
                fv = c.f64()
            elif vtype == 8:
                sv = c.gstring()
            g.meta.keys.append(key)
            g.meta.kinds.append(vtype)
            g.meta.ivals.append(iv)
            g.meta.fvals.append(fv)
            g.meta.svals.append(sv)
            c.pos = save

        if key == "general.alignment":
            var save2 = c.pos
            c.pos = pos_before
            g.alignment = Int(c.u32())
            c.pos = save2

    for _ in range(n_tensors):
        var name = c.gstring()
        var ndim = Int(c.u32())
        var nelem = 1
        var d0 = 1
        var d1 = 1
        for d in range(ndim):
            var dim = Int(c.u64())
            nelem *= dim
            if d == 0:
                d0 = dim
            elif d == 1:
                d1 = dim
            else:
                d1 *= dim          # fold any higher axes into the row count
        g.types.append(Int(c.u32()))
        g.offsets.append(Int(c.u64()))
        g.names.append(name)
        g.counts.append(nelem)
        g.ne0.append(d0)
        g.ne1.append(d1)

    var ds = c.pos
    if ds % g.alignment != 0:
        ds += g.alignment - (ds % g.alignment)
    g.data_start = ds
    return g^


# Pull the tokenizer vocabulary out of the metadata. The GGUF already carries all
# 151,669 token strings, so nothing needs downloading — but they must be walked with
# a parser that skips other values correctly rather than materialising them.
def load_tokens(path: String) raises -> List[String]:
    var f = open(path, "r")
    var head = f.read_bytes(HEADER_BUDGET)
    var c = Cursor(head^)
    if c.u32() != GGUF_MAGIC:
        raise Error("not a GGUF file")
    _ = c.u32()                     # version
    _ = c.u64()                     # tensor count
    var n_kv = Int(c.u64())

    var out = List[String]()
    for _ in range(n_kv):
        var key = c.gstring()
        var vtype = Int(c.u32())
        if key == "tokenizer.ggml.tokens":
            if vtype != 9:
                raise Error("tokenizer.ggml.tokens is not an array")
            var etype = Int(c.u32())
            if etype != 8:
                raise Error("token array is not strings")
            var n = Int(c.u64())
            for _t in range(n):
                out.append(c.gstring())
        else:
            _ = read_value(c, vtype)     # walk past it
    if len(out) == 0:
        raise Error("no tokenizer.ggml.tokens found in " + path)
    return out^


# Slurp an entire GGUF into memory. At 1.7B the file is 463MB, so reading it once
# and unpacking from the buffer beats 310 separate seeks.
def read_whole_file(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    return f.read_bytes()


# Read raw bytes for one tensor straight out of the file.
def read_blob(path: String, byte_offset: Int, nbytes: Int) raises -> List[UInt8]:
    var f = open(path, "r")
    _ = f.seek(byte_offset)
    var b = f.read_bytes(nbytes)
    if len(b) != nbytes:
        raise Error(
            "short read: wanted " + String(nbytes) + " got " + String(len(b))
        )
    return b^


# The whole point of M0: 2-bit codes + one FP16 scale per 128 -> FP32 weights.
# Plain scalar Mojo, deliberately obvious, no SIMD.
def dequant_q2_g128(blob: List[UInt8], nelem: Int) raises -> List[Float32]:
    if nelem % TERNARY_GROUP != 0:
        raise Error("element count is not a multiple of 128: " + String(nelem))
    var nblocks = nelem // TERNARY_GROUP
    if len(blob) != nblocks * TERNARY_BLOCK_BYTES:
        raise Error(
            "blob size " + String(len(blob)) + " != "
            + String(nblocks * TERNARY_BLOCK_BYTES)
        )

    var out = List[Float32](length=nelem, fill=0)
    for b in range(nblocks):
        var base = b * TERNARY_BLOCK_BYTES
        # 2-byte FP16 scale, little-endian, then 32 bytes of packed codes.
        var raw = UInt16(blob[base]) | (UInt16(blob[base + 1]) << 8)
        var scale = bitcast[DType.float16](raw).cast[DType.float32]()
        for j in range(32):
            var byte = blob[base + 2 + j]
            for k in range(4):
                # 4 codes per byte, low bits first.
                var q = Int((byte >> UInt8(2 * k)) & 0x3)
                out[b * TERNARY_GROUP + j * 4 + k] = Float32(q - 1) * scale
    return out^


# Recover the raw 2-bit code and the block's FP16 scale behind one element, so a
# mismatch can be explained rather than just counted.
def code_and_scale(blob: List[UInt8], i: Int) raises -> Tuple[Int, Float32]:
    var b = i // TERNARY_GROUP
    var within = i % TERNARY_GROUP
    var base = b * TERNARY_BLOCK_BYTES
    var raw = UInt16(blob[base]) | (UInt16(blob[base + 1]) << 8)
    var scale = bitcast[DType.float16](raw).cast[DType.float32]()
    var byte = blob[base + 2 + within // 4]
    var q = Int((byte >> UInt8(2 * (within % 4))) & 0x3)
    return (q, scale)


def decode_f16(blob: List[UInt8], nelem: Int) raises -> List[Float32]:
    if len(blob) != nelem * 2:
        raise Error("f16 blob size mismatch")
    var out = List[Float32](length=nelem, fill=0)
    for i in range(nelem):
        var raw = UInt16(blob[i * 2]) | (UInt16(blob[i * 2 + 1]) << 8)
        out[i] = bitcast[DType.float16](raw).cast[DType.float32]()
    return out^


