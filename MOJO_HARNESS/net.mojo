"""Native TCP over libc FFI — the one narrow non-Mojo seam in the harness.

Everything here is `external_call` into libc: socket / connect / send / recv /
close. No std.socket, no std.net, no Python. Targets plain HTTP on localhost
(no TLS), matching the LM Studio endpoint at 127.0.0.1:1234.
"""

from std.ffi import external_call
from std.memory import Span


comptime AF_INET: Int32 = 2       # Darwin AF_INET
comptime SOCK_STREAM: Int32 = 1
comptime RECV_CHUNK: Int = 8192


def tcp_connect(ip0: UInt8, ip1: UInt8, ip2: UInt8, ip3: UInt8, port: UInt16) raises -> Int32:
    """Open a TCP connection, return a connected fd (raises on failure).

    The 16-byte Darwin sockaddr_in is hand-built with bytes written directly,
    so we never need htons/inet_pton."""
    var fd = external_call["socket", Int32](AF_INET, SOCK_STREAM, Int32(0))
    if fd < 0:
        raise Error("socket() failed")

    var addr = alloc[UInt8](16)
    for i in range(16):
        addr[i] = 0
    addr[0] = 16                       # sin_len (BSD/Darwin)
    addr[1] = UInt8(AF_INET)           # sin_family
    addr[2] = UInt8(port >> 8)         # sin_port hi (network/big-endian)
    addr[3] = UInt8(port & 0xFF)       # sin_port lo
    addr[4] = ip0                      # sin_addr, already network order
    addr[5] = ip1
    addr[6] = ip2
    addr[7] = ip3

    var rc = external_call["connect", Int32](fd, addr, UInt32(16))
    addr.free()
    if rc != 0:
        _ = external_call["close", Int32](fd)
        raise Error("connect() failed (is LM Studio listening?)")
    return fd


def tcp_send(fd: Int32, data: String) raises:
    """Send the full string over the socket."""
    var ptr = data.unsafe_ptr()
    var total = data.byte_length()
    var sent = 0
    while sent < total:
        var n = external_call["send", Int](fd, ptr + sent, total - sent, Int32(0))
        if n <= 0:
            raise Error("send() failed")
        sent += n


def tcp_recv_all(fd: Int32) raises -> String:
    """Read until the peer closes the connection, return the whole response."""
    var buf = alloc[UInt8](RECV_CHUNK)
    var out = String("")
    while True:
        var n = external_call["recv", Int](fd, buf, RECV_CHUNK, Int32(0))
        if n <= 0:
            break
        out += StringSlice(unsafe_from_utf8=Span(ptr=buf, length=n))
    buf.free()
    return out


def tcp_recv(fd: Int32) raises -> String:
    """Read a single chunk. Returns an empty string when the peer has closed —
    this is the streaming primitive the SSE loop drives."""
    var buf = alloc[UInt8](RECV_CHUNK)
    var n = external_call["recv", Int](fd, buf, RECV_CHUNK, Int32(0))
    var out = String("")
    if n > 0:
        out += StringSlice(unsafe_from_utf8=Span(ptr=buf, length=n))
    buf.free()
    return out


def tcp_close(fd: Int32):
    _ = external_call["close", Int32](fd)
