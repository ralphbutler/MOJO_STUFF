"""Run a shell command and capture its output — via libc popen/pclose.

We deliberately avoid `std.subprocess.run`: on this Mojo build it closes the
parent's stdin fd, which breaks an interactive REPL after the first tool call.
`popen` runs `/bin/sh -c <cmd>` over a pipe and leaves our stdin untouched — and
it keeps the tool half of the harness on the same native FFI seam as the socket.
"""

from std.ffi import external_call
from std.memory import Span


comptime READ_CHUNK: Int = 4096


def _cstr(s: String) -> List[UInt8]:
    """A guaranteed NUL-terminated byte buffer. `String.unsafe_ptr()` is NOT
    NUL-terminated, so passing it straight to a C string arg reads past the end
    — popen would see the command plus adjacent garbage."""
    var buf = List[UInt8]()
    for i in range(s.byte_length()):
        buf.append(s.unsafe_ptr()[i])
    buf.append(0)
    return buf^


def run_capture(command: String) raises -> String:
    """Run `command` via /bin/sh, return combined stdout+stderr.

    The `FILE*` from popen is carried as a pointer-sized `Int` handle — the C
    ABI passes it back verbatim to fread/pclose, sidestepping pointer-origin
    bookkeeping for an opaque handle we never dereference in Mojo."""
    # Wrap in a subshell so the redirects apply to the whole command as a
    # group: `</dev/null` neutralizes stray stdin reads without clobbering an
    # internal pipe (a bare trailing `</dev/null` would attach only to the last
    # stage of `find | wc -l` and make wc count /dev/null → 0).
    var cmd = _cstr("( " + command + " ) </dev/null 2>&1")
    var mode = _cstr("r")
    var fp = external_call["popen", Int](cmd.unsafe_ptr(), mode.unsafe_ptr())
    if fp == 0:
        raise Error("popen failed for: " + command)

    var buf = alloc[UInt8](READ_CHUNK)
    var out = String("")
    while True:
        var n = external_call["fread", Int](buf, Int(1), READ_CHUNK, fp)
        if n <= 0:
            break
        out += StringSlice(unsafe_from_utf8=Span(ptr=buf, length=n))
    buf.free()

    _ = external_call["pclose", Int32](fp)
    return out
