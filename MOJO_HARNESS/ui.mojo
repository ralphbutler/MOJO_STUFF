"""Tiny terminal rendering helpers + a native line reader.

`chr(27)` gives us ESC without relying on `\x1b` literal support. Kept trivial;
the harness is a systems demo, not a TUI framework.

`read_line` reads stdin one byte at a time via libc `read(2)` instead of the
builtin `input()`, which on this Mojo build over-reads a piped stdin buffer and
loses everything after the first line. On a real terminal stdin is in cooked
mode, so the kernel line discipline already provides in-line editing —
**Backspace**, word-erase (Ctrl-W), line-kill (Ctrl-U) — and delivers the edited
line on Enter; we just read the finished bytes. (Arrow-key cursor movement would
need raw mode via `termios`; that's a deferred stretch item.) Reading byte-wise
consumes exactly one line, preserves the rest, and accumulates raw bytes so
UTF-8 input survives.
"""

from std.ffi import external_call
from std.memory import Span


def _esc(code: String) -> String:
    return String(chr(27)) + "[" + code + "m"


comptime RESET_CODE: String = "0"

# reasoning display modes
comptime THINK_OFF: Int = 0       # hide thinking entirely
comptime THINK_COMPACT: Int = 1   # a single live "thinking… N" line
comptime THINK_FULL: Int = 2      # stream the whole reasoning, dimmed


def clear_line() -> String:
    """Erase the current terminal line and return the cursor to column 0."""
    return String(chr(27)) + "[2K\r"


def _wrap(code: String, s: String) -> String:
    return _esc(code) + s + _esc(RESET_CODE)


def bold(s: String) -> String:
    return _wrap("1", s)


def dim(s: String) -> String:
    return _wrap("2", s)


def cyan(s: String) -> String:
    return _wrap("36", s)


def green(s: String) -> String:
    return _wrap("32", s)


def yellow(s: String) -> String:
    return _wrap("33", s)


def magenta(s: String) -> String:
    return _wrap("35", s)


def read_line(prompt: String) raises -> Optional[String]:
    """Print `prompt`, read one line from stdin. On a terminal the kernel
    handles Backspace/Ctrl-W/Ctrl-U before the line is delivered. Returns None on
    EOF with no input (Ctrl-D / end of pipe)."""
    print(prompt, end="", flush=True)
    var one = alloc[UInt8](1)
    var bytes = List[UInt8]()
    var got_any = False
    while True:
        var n = external_call["read", Int](Int32(0), one, Int(1))
        if n <= 0:
            one.free()
            if got_any:
                return String(StringSlice(unsafe_from_utf8=Span(bytes)))
            return None
        got_any = True
        var c = Int(one[0])
        if c == 10:          # \n ends the line
            one.free()
            return String(StringSlice(unsafe_from_utf8=Span(bytes)))
        if c == 13:          # swallow \r (CRLF)
            continue
        bytes.append(UInt8(c))


def banner():
    print(cyan(bold("┌─ mojo-harness ──────────────────────────────────┐")))
    print(cyan("│") + " a tiny LLM coding agent, written natively in Mojo" + cyan("│"))
    print(cyan("│") + dim(" libc sockets · hand-rolled JSON · bash tool      ") + cyan("│"))
    print(cyan(bold("└─────────────────────────────────────────────────┘")))
    print(dim("type a task, or 'exit' / 'quit' to leave.\n"))
