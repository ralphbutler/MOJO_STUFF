"""Tiny HTTP/1.1 client built on the native socket layer.

Only what the harness needs: a JSON POST that returns the response body. Uses
`Connection: close` so the server closes the stream when done and our recv loop
terminates naturally. Body extraction skips headers (and any chunked framing) by
seeking to the first `{` — good enough for OpenAI-style JSON responses.
"""

from std.memory import Span
from net import tcp_connect, tcp_send, tcp_recv_all, tcp_close


def _substr(s: String, start: Int) -> String:
    """Materialize s[start:] at the byte level (no String slice syntax exists)."""
    var length = s.byte_length() - start
    return String(StringSlice(unsafe_from_utf8=Span(ptr=s.unsafe_ptr() + start, length=length)))


def build_post_request(host: String, path: String, body: String) -> String:
    """Frame an HTTP/1.1 POST with a JSON body. Shared by the buffered and
    streaming clients."""
    var req = String("POST ")
    req += path
    req += " HTTP/1.1\r\nHost: "
    req += host
    req += "\r\nContent-Type: application/json\r\nContent-Length: "
    req += String(body.byte_length())
    req += "\r\nConnection: close\r\n\r\n"
    req += body
    return req


def http_post_json(host: String, ip0: UInt8, ip1: UInt8, ip2: UInt8, ip3: UInt8,
                   port: UInt16, path: String, body: String) raises -> String:
    """POST a JSON body, return the raw response body (headers stripped)."""
    var req = build_post_request(host, path, body)

    var fd = tcp_connect(ip0, ip1, ip2, ip3, port)
    tcp_send(fd, req)
    var raw = tcp_recv_all(fd)
    tcp_close(fd)

    var brace = raw.find("{")
    if brace < 0:
        raise Error("no JSON body in response:\n" + raw)
    return _substr(raw, brace)


def http_get(host: String, ip0: UInt8, ip1: UInt8, ip2: UInt8, ip3: UInt8,
             port: UInt16, path: String) raises -> String:
    """GET a path, return the JSON response body (headers stripped)."""
    var req = String("GET ")
    req += path
    req += " HTTP/1.1\r\nHost: "
    req += host
    req += "\r\nConnection: close\r\n\r\n"

    var fd = tcp_connect(ip0, ip1, ip2, ip3, port)
    tcp_send(fd, req)
    var raw = tcp_recv_all(fd)
    tcp_close(fd)

    var brace = raw.find("{")
    if brace < 0:
        raise Error("no JSON body in response:\n" + raw)
    return _substr(raw, brace)
