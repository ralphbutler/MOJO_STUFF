"""Build step 3: a real non-streaming chat completion, end to end, all native."""

from json import JSONValue, parse, serialize
from http import http_post_json


comptime MODEL: String = "qwen3.6-27b-mlx"


def build_body(system: String, user: String) -> String:
    var msgs = JSONValue.array()
    var m0 = JSONValue.object()
    m0.set("role", JSONValue.string("system"))
    m0.set("content", JSONValue.string(system))
    msgs.append(m0^)
    var m1 = JSONValue.object()
    m1.set("role", JSONValue.string("user"))
    m1.set("content", JSONValue.string(user))
    msgs.append(m1^)

    var body = JSONValue.object()
    body.set("model", JSONValue.string(MODEL))
    body.set("messages", msgs^)
    body.set("stream", JSONValue.bool(False))
    body.set("temperature", JSONValue.number(0.3))
    return serialize(body)


def main() raises:
    var body = build_body(
        "You are a terse assistant. Answer in one sentence.",
        "In one sentence, what is the Mojo programming language?",
    )
    print("→ POST /v1/chat/completions (", body.byte_length(), "bytes )")

    var resp_body = http_post_json(
        "localhost:1234", 127, 0, 0, 1, 1234, "/v1/chat/completions", body
    )
    var resp = parse(resp_body)
    var content = resp.get("choices").at(0).get("message").get("content").as_string()

    print("\n← assistant:")
    print(content)

    if resp.has("usage"):
        var usage = resp.get("usage")
        if usage.has("total_tokens"):
            print("\n[tokens:", Int(usage.get("total_tokens").as_number()), "]")
