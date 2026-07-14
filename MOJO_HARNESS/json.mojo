"""A minimal, fully-native JSON parser + serializer.

No std.json exists in Mojo 1.0, so this is hand-rolled. It is the harness's
single best "look what Mojo can do natively" artifact: a recursive-descent
parser producing a recursive `JSONValue`, plus a serializer with correct string
escaping. Scope is deliberately small — enough for OpenAI-style chat bodies and
responses — but it is a real, general parser, not a shape-specific hack.

Objects preserve insertion order (parallel key/value lists) which keeps request
bodies stable and readable.
"""

from std.memory import Span


comptime KIND_NULL: Int = 0
comptime KIND_BOOL: Int = 1
comptime KIND_NUM: Int = 2
comptime KIND_STR: Int = 3
comptime KIND_ARR: Int = 4
comptime KIND_OBJ: Int = 5


struct JSONValue(Copyable, Movable):
    """A JSON value. Recursion goes through `List[JSONValue]`, whose heap
    pointer gives the type a fixed size, so self-reference is legal."""

    var kind: Int
    var b: Bool
    var num: Float64
    var s: String
    var arr: List[JSONValue]
    var keys: List[String]
    var vals: List[JSONValue]

    def __init__(out self):
        self.kind = KIND_NULL
        self.b = False
        self.num = 0.0
        self.s = String("")
        self.arr = List[JSONValue]()
        self.keys = List[String]()
        self.vals = List[JSONValue]()

    # ---- constructors for each kind ----
    @staticmethod
    def null() -> JSONValue:
        return JSONValue()

    @staticmethod
    def bool(value: Bool) -> JSONValue:
        var v = JSONValue()
        v.kind = KIND_BOOL
        v.b = value
        return v^

    @staticmethod
    def number(value: Float64) -> JSONValue:
        var v = JSONValue()
        v.kind = KIND_NUM
        v.num = value
        return v^

    @staticmethod
    def string(value: String) -> JSONValue:
        var v = JSONValue()
        v.kind = KIND_STR
        v.s = value
        return v^

    @staticmethod
    def array() -> JSONValue:
        var v = JSONValue()
        v.kind = KIND_ARR
        return v^

    @staticmethod
    def object() -> JSONValue:
        var v = JSONValue()
        v.kind = KIND_OBJ
        return v^

    # ---- builders ----
    def append(mut self, var item: JSONValue):
        self.arr.append(item^)

    def set(mut self, key: String, var item: JSONValue):
        self.keys.append(key)
        self.vals.append(item^)

    # ---- typed accessors ----
    def as_string(self) -> String:
        return self.s

    def as_number(self) -> Float64:
        return self.num

    def get(self, key: String) raises -> JSONValue:
        """Object field lookup by key (raises if absent)."""
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return self.vals[i].copy()
        raise Error("key not found: " + key)

    def has(self, key: String) -> Bool:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return True
        return False

    def at(self, i: Int) raises -> JSONValue:
        """Array element by index."""
        return self.arr[i].copy()

    def count(self) -> Int:
        """Number of array elements (0 for non-arrays, incl. null)."""
        return len(self.arr)

    def is_null(self) -> Bool:
        return self.kind == KIND_NULL


# =====================================================================
# Serializer
# =====================================================================

def _escape_string(s: String) -> String:
    """Escape a string for JSON output (quotes, backslash, control chars)."""
    var out = String('"')
    for cp in s.codepoint_slices():
        var c = String(cp)
        if c == '"':
            out += '\\"'
        elif c == "\\":
            out += "\\\\"
        elif c == "\n":
            out += "\\n"
        elif c == "\r":
            out += "\\r"
        elif c == "\t":
            out += "\\t"
        else:
            out += c
    out += '"'
    return out


def _num_to_string(x: Float64) -> String:
    """Emit integers without a trailing .0, else the float."""
    var as_int = Int(x)
    if Float64(as_int) == x:
        return String(as_int)
    return String(x)


def serialize(v: JSONValue) -> String:
    if v.kind == KIND_NULL:
        return String("null")
    elif v.kind == KIND_BOOL:
        return String("true") if v.b else String("false")
    elif v.kind == KIND_NUM:
        return _num_to_string(v.num)
    elif v.kind == KIND_STR:
        return _escape_string(v.s)
    elif v.kind == KIND_ARR:
        var out = String("[")
        for i in range(len(v.arr)):
            if i > 0:
                out += ","
            out += serialize(v.arr[i])
        out += "]"
        return out
    else:  # KIND_OBJ
        var out = String("{")
        for i in range(len(v.keys)):
            if i > 0:
                out += ","
            out += _escape_string(v.keys[i])
            out += ":"
            out += serialize(v.vals[i])
        out += "}"
        return out


# =====================================================================
# Parser (recursive descent over a byte cursor)
# =====================================================================

struct _Parser(Copyable, Movable):
    var src: String
    var pos: Int
    var n: Int

    def __init__(out self, var src: String):
        self.n = src.byte_length()
        self.src = src^
        self.pos = 0

    def _peek(self) -> Int:
        if self.pos >= self.n:
            return -1
        return Int(self.src.unsafe_ptr()[self.pos])

    def _next(mut self) -> Int:
        var c = self._peek()
        self.pos += 1
        return c

    def _skip_ws(mut self):
        while self.pos < self.n:
            var c = self._peek()
            if c == 32 or c == 9 or c == 10 or c == 13:  # space tab nl cr
                self.pos += 1
            else:
                break

    def parse_value(mut self) raises -> JSONValue:
        self._skip_ws()
        var c = self._peek()
        if c == Int(ord("{")):
            return self._parse_object()
        elif c == Int(ord("[")):
            return self._parse_array()
        elif c == Int(ord('"')):
            return JSONValue.string(self._parse_string())
        elif c == Int(ord("t")) or c == Int(ord("f")):
            return self._parse_bool()
        elif c == Int(ord("n")):
            return self._parse_null()
        else:
            return JSONValue.number(self._parse_number())

    def _parse_object(mut self) raises -> JSONValue:
        var obj = JSONValue.object()
        _ = self._next()  # consume {
        self._skip_ws()
        if self._peek() == Int(ord("}")):
            _ = self._next()
            return obj^
        while True:
            self._skip_ws()
            var key = self._parse_string()
            self._skip_ws()
            _ = self._next()  # consume :
            var val = self.parse_value()
            obj.set(key, val^)
            self._skip_ws()
            var d = self._next()  # , or }
            if d == Int(ord("}")):
                break
        return obj^

    def _parse_array(mut self) raises -> JSONValue:
        var arr = JSONValue.array()
        _ = self._next()  # consume [
        self._skip_ws()
        if self._peek() == Int(ord("]")):
            _ = self._next()
            return arr^
        while True:
            var val = self.parse_value()
            arr.append(val^)
            self._skip_ws()
            var d = self._next()  # , or ]
            if d == Int(ord("]")):
                break
        return arr^

    def _parse_string(mut self) raises -> String:
        # Accumulate RAW bytes so multibyte UTF-8 in the source is preserved
        # verbatim (chr() per byte would reinterpret each byte as a codepoint).
        _ = self._next()  # consume opening quote
        var bytes = List[UInt8]()
        while True:
            var c = self._next()
            if c == -1:
                raise Error("unterminated string")
            if c == Int(ord('"')):
                break
            if c == Int(ord("\\")):
                var e = self._next()
                if e == Int(ord("n")):
                    bytes.append(10)
                elif e == Int(ord("t")):
                    bytes.append(9)
                elif e == Int(ord("r")):
                    bytes.append(13)
                elif e == Int(ord("b")):
                    bytes.append(8)
                elif e == Int(ord("f")):
                    bytes.append(12)
                elif e == Int(ord('"')):
                    bytes.append(34)
                elif e == Int(ord("\\")):
                    bytes.append(92)
                elif e == Int(ord("/")):
                    bytes.append(47)
                elif e == Int(ord("u")):
                    # \uXXXX → codepoint → its UTF-8 bytes
                    var s = self._parse_unicode_escape()
                    for i in range(s.byte_length()):
                        bytes.append(s.unsafe_ptr()[i])
                else:
                    bytes.append(UInt8(e))
            else:
                bytes.append(UInt8(c))
        return String(StringSlice(unsafe_from_utf8=Span(bytes)))

    def _parse_unicode_escape(mut self) raises -> String:
        var code = 0
        for _ in range(4):
            var h = self._next()
            code = code * 16 + _hex_val(h)
        return String(chr(code))

    def _parse_bool(mut self) raises -> JSONValue:
        if self._peek() == Int(ord("t")):
            self.pos += 4  # true
            return JSONValue.bool(True)
        self.pos += 5      # false
        return JSONValue.bool(False)

    def _parse_null(mut self) raises -> JSONValue:
        self.pos += 4      # null
        return JSONValue.null()

    def _parse_number(mut self) raises -> Float64:
        var start = self.pos
        while self.pos < self.n:
            var c = self._peek()
            if (c >= Int(ord("0")) and c <= Int(ord("9"))) or c == Int(ord("-")) \
               or c == Int(ord("+")) or c == Int(ord(".")) or c == Int(ord("e")) \
               or c == Int(ord("E")):
                self.pos += 1
            else:
                break
        var tok = String("")
        for i in range(start, self.pos):
            tok += String(chr(Int(self.src.unsafe_ptr()[i])))
        return Float64(atof(tok))


def _hex_val(c: Int) raises -> Int:
    if c >= Int(ord("0")) and c <= Int(ord("9")):
        return c - Int(ord("0"))
    elif c >= Int(ord("a")) and c <= Int(ord("f")):
        return c - Int(ord("a")) + 10
    elif c >= Int(ord("A")) and c <= Int(ord("F")):
        return c - Int(ord("A")) + 10
    raise Error("bad hex digit")


def parse(src: String) raises -> JSONValue:
    var p = _Parser(src)
    return p.parse_value()
