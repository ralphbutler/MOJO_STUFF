"""Build step 4: streaming chat completion — text prints token-by-token."""

from json import JSONValue, serialize
from sse import stream_completion


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
    body.set("stream", JSONValue.bool(True))
    body.set("temperature", JSONValue.number(0.5))
    return serialize(body)


def main() raises:
    var body = build_body(
        "You are a concise assistant.",
        "List three reasons the Mojo language is interesting. One line each.",
    )
    print("← streaming:\n")
    var full = stream_completion(
        "localhost:1234", 127, 0, 0, 1, 1234, "/v1/chat/completions", body
    )
    print("\n[stream complete,", full.byte_length(), "chars]")
