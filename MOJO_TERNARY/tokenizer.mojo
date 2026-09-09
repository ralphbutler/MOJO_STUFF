# tokenizer.mojo — GPT-2 byte-level decoding, in pure Mojo.
#
# DECODE ONLY, and that is a deliberate split. Turning token ids back into text
# needs no regex: every token string is the UTF-8 encoding of a run of codepoints
# that each stand for one original byte. Reverse that map, concatenate, done.
#
# Encoding is the half that needs qwen2's pre-tokenizer regex, and Mojo's stdlib has
# no regex engine. That is deferred — prompt token ids come from the oracle for now.
#
# The vocabulary is not downloaded: all 151,669 token strings live in the GGUF's own
# metadata, read by gguf.load_tokens.
#
# The map itself is GPT-2's `bytes_to_unicode`: printable ASCII and two Latin-1 runs
# stand for themselves, and the remaining 68 bytes are displaced to U+0100 and up so
# that no token string ever contains a control character or a bare space.

comptime BYTE_MAP_SIZE = 512


struct ByteDecoder(Movable):
    var byte_of: List[Int]        # codepoint -> original byte, or -1

    def __init__(out self):
        self.byte_of = List[Int](length=BYTE_MAP_SIZE, fill=-1)

        # The bytes that represent themselves: '!'..'~', 0xA1..0xAC, 0xAE..0xFF.
        var direct = List[Int]()
        for b in range(33, 127):
            direct.append(b)
        for b in range(161, 173):
            direct.append(b)
        for b in range(174, 256):
            direct.append(b)
        for i in range(len(direct)):
            self.byte_of[direct[i]] = direct[i]

        # Everything else is displaced to 256, 257, ... in byte order.
        var n = 0
        for b in range(256):
            var is_direct = False
            for i in range(len(direct)):
                if direct[i] == b:
                    is_direct = True
                    break
            if not is_direct:
                self.byte_of[256 + n] = b
                n += 1

    # One token string -> the raw bytes it stands for.
    def to_bytes(self, tok: String, mut out: List[UInt8]) raises:
        for cp in tok.codepoints():
            var v = Int(cp)
            if v < 0 or v >= BYTE_MAP_SIZE or self.byte_of[v] < 0:
                # Control tokens like <|endoftext|> are plain ASCII and survive the
                # map; anything genuinely unmapped is a real problem, so say so.
                raise Error("codepoint outside the GPT-2 byte map: " + String(v))
            out.append(UInt8(self.byte_of[v]))


# Decode a run of token ids into text.
def decode(
    dec: ByteDecoder, vocab: List[String], ids: List[Int]
) raises -> String:
    var buf = List[UInt8]()
    for i in range(len(ids)):
        var t = ids[i]
        if t < 0 or t >= len(vocab):
            raise Error("token id out of range: " + String(t))
        dec.to_bytes(vocab[t], buf)
    return String(StringSlice(unsafe_from_utf8=Span(buf)))
