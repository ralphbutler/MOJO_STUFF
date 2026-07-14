"""Server-Sent-Events client for streaming chat completions.

Drives the native socket one recv-chunk at a time, buffers, strips the HTTP
headers once, then pulls complete `data:` lines out of the buffer and prints
each delta's text as it arrives. Tolerant by design: any line that isn't a
`data:` line (blank lines, chunked-transfer size markers) is ignored, so we do
not need a full chunked-encoding decoder. Terminates on `data: [DONE]`.
"""

from std.memory import Span
from net import tcp_connect, tcp_send, tcp_recv, tcp_close
from http import build_post_request
from json import parse, JSONValue
from ui import dim, green, bold, clear_line, THINK_OFF, THINK_COMPACT, THINK_FULL


def _slice(s: String, start: Int, end: Int) -> String:
    """Byte-level s[start:end] (String has no slice syntax)."""
    return String(StringSlice(unsafe_from_utf8=Span(ptr=s.unsafe_ptr() + start, length=end - start)))


def _delta_str(delta: JSONValue, key: String) raises -> String:
    """Return a string delta field, or "" if absent/null."""
    if not delta.has(key):
        return String("")
    var v = delta.get(key)
    if v.is_null():
        return String("")
    return v.as_string()


def stream_message(host: String, ip0: UInt8, ip1: UInt8, ip2: UInt8, ip3: UInt8,
                   port: UInt16, path: String, body: String,
                   reasoning_mode: Int) raises -> JSONValue:
    """Stream a chat completion, printing content live (and reasoning per
    `reasoning_mode`: off / compact live counter / full dimmed stream), and
    reconstruct the assistant `message` — including tool_calls, whose fragments
    arrive spread across chunks and must be concatenated by index."""
    var req = build_post_request(host, path, body)
    var fd = tcp_connect(ip0, ip1, ip2, ip3, port)
    tcp_send(fd, req)

    var buffer = String("")
    var headers_done = False
    var done = False

    var content = String("")
    var wrote_reasoning = False   # full-mode reasoning line is open
    var wrote_content = False
    var think_line = False        # compact-mode "thinking…" line is on screen
    var think_count = 0

    # tool_calls accumulate per index across deltas
    var tc_ids = List[String]()
    var tc_names = List[String]()
    var tc_args = List[String]()

    while not done:
        var chunk = tcp_recv(fd)
        if chunk.byte_length() == 0:
            break
        buffer += chunk

        if not headers_done:
            var hdr = buffer.find("\r\n\r\n")
            if hdr < 0:
                continue
            buffer = _slice(buffer, hdr + 4, buffer.byte_length())
            headers_done = True

        while True:
            var nl = buffer.find("\n")
            if nl < 0:
                break
            var line = _slice(buffer, 0, nl)
            buffer = _slice(buffer, nl + 1, buffer.byte_length())

            if line.endswith("\r"):
                line = _slice(line, 0, line.byte_length() - 1)
            if not line.startswith("data:"):
                continue
            var payload = _slice(line, 5, line.byte_length())
            if payload.startswith(" "):
                payload = _slice(payload, 1, payload.byte_length())
            if payload == "[DONE]":
                done = True
                break

            var obj = parse(payload)
            if not obj.has("choices"):
                continue
            var choice0 = obj.get("choices").at(0)
            if not choice0.has("delta"):
                continue
            var delta = choice0.get("delta")

            # -- reasoning (per mode) --
            if reasoning_mode != THINK_OFF:
                var r = _delta_str(delta, "reasoning_content")
                if r.byte_length() > 0:
                    if reasoning_mode == THINK_FULL:
                        if not wrote_reasoning:
                            print(dim("  💭 "), end="", flush=True)
                            wrote_reasoning = True
                        print(dim(r), end="", flush=True)
                    else:  # THINK_COMPACT — one self-updating line
                        think_count += 1
                        think_line = True
                        print(clear_line() + dim("  💭 thinking… " + String(think_count)),
                              end="", flush=True)

            # -- content (normal, live) --
            var c = _delta_str(delta, "content")
            if c.byte_length() > 0:
                if not wrote_content:
                    if wrote_reasoning:
                        print("")   # close the full reasoning line
                    if think_line:
                        print(clear_line(), end="", flush=True)  # drop the counter
                        think_line = False
                    print(green(bold("  🤖 ")), end="", flush=True)
                    wrote_content = True
                print(c, end="", flush=True)
                content += c

            # -- tool_calls (fragmented; concat by index) --
            if delta.has("tool_calls"):
                var tcs = delta.get("tool_calls")
                for i in range(tcs.count()):
                    var tc = tcs.at(i)
                    var idx = 0
                    if tc.has("index"):
                        idx = Int(tc.get("index").as_number())
                    while len(tc_ids) <= idx:
                        tc_ids.append(String(""))
                        tc_names.append(String(""))
                        tc_args.append(String(""))
                    var id_frag = _delta_str(tc, "id")
                    if id_frag.byte_length() > 0:
                        tc_ids[idx] = id_frag
                    if tc.has("function"):
                        var func = tc.get("function")
                        var nm = _delta_str(func, "name")
                        if nm.byte_length() > 0:
                            tc_names[idx] = nm
                        var ag = _delta_str(func, "arguments")
                        if ag.byte_length() > 0:
                            tc_args[idx] = tc_args[idx] + ag

    tcp_close(fd)
    if think_line:
        print(clear_line(), end="", flush=True)  # tool-only turn: drop counter
    if wrote_reasoning or wrote_content:
        print("")   # trailing newline after the live stream

    # -- assemble the message object --
    var msg = JSONValue.object()
    msg.set("role", JSONValue.string("assistant"))
    msg.set("content", JSONValue.string(content))
    if len(tc_ids) > 0:
        var arr = JSONValue.array()
        for i in range(len(tc_ids)):
            var func = JSONValue.object()
            func.set("name", JSONValue.string(tc_names[i]))
            func.set("arguments", JSONValue.string(tc_args[i]))
            var call = JSONValue.object()
            call.set("id", JSONValue.string(tc_ids[i]))
            call.set("type", JSONValue.string("function"))
            call.set("function", func^)
            arr.append(call^)
        msg.set("tool_calls", arr^)
    return msg^


def stream_completion(host: String, ip0: UInt8, ip1: UInt8, ip2: UInt8, ip3: UInt8,
                      port: UInt16, path: String, body: String) raises -> String:
    """POST a streaming request; print deltas live; return the full text."""
    var req = build_post_request(host, path, body)
    var fd = tcp_connect(ip0, ip1, ip2, ip3, port)
    tcp_send(fd, req)

    var buffer = String("")
    var full = String("")
    var headers_done = False
    var done = False

    while not done:
        var chunk = tcp_recv(fd)
        if chunk.byte_length() == 0:
            break
        buffer += chunk

        if not headers_done:
            var hdr = buffer.find("\r\n\r\n")
            if hdr < 0:
                continue
            buffer = _slice(buffer, hdr + 4, buffer.byte_length())
            headers_done = True

        while True:
            var nl = buffer.find("\n")
            if nl < 0:
                break
            var line = _slice(buffer, 0, nl)
            buffer = _slice(buffer, nl + 1, buffer.byte_length())

            if line.endswith("\r"):
                line = _slice(line, 0, line.byte_length() - 1)
            if not line.startswith("data:"):
                continue

            var payload = _slice(line, 5, line.byte_length())
            if payload.startswith(" "):
                payload = _slice(payload, 1, payload.byte_length())
            if payload == "[DONE]":
                done = True
                break

            var obj = parse(payload)
            if obj.has("choices"):
                var delta = obj.get("choices").at(0).get("delta")
                if delta.has("content"):
                    var piece = delta.get("content").as_string()
                    print(piece, end="", flush=True)
                    full += piece

    tcp_close(fd)
    print("")  # newline after the stream
    return full
