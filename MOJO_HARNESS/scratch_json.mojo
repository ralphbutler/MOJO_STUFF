from json import JSONValue, parse, serialize


def main() raises:
    # ---- build a chat request body natively ----
    var msgs = JSONValue.array()
    var m0 = JSONValue.object()
    m0.set("role", JSONValue.string("system"))
    m0.set("content", JSONValue.string("You are terse."))
    msgs.append(m0^)
    var m1 = JSONValue.object()
    m1.set("role", JSONValue.string("user"))
    m1.set("content", JSONValue.string("Say \"hi\"\nthen stop."))
    msgs.append(m1^)

    var body = JSONValue.object()
    body.set("model", JSONValue.string("qwen3.6-27b-mlx"))
    body.set("messages", msgs^)
    body.set("stream", JSONValue.bool(False))
    body.set("temperature", JSONValue.number(0.7))
    body.set("max_tokens", JSONValue.number(256))

    var wire = serialize(body)
    print("REQUEST BODY:")
    print(wire)

    # ---- parse a mock chat completion response ----
    var resp_text = String(
        '{"id":"chatcmpl-1","object":"chat.completion",'
        '"choices":[{"index":0,"message":{"role":"assistant",'
        '"content":"Hello there!"},"finish_reason":"stop"}],'
        '"usage":{"total_tokens":42}}'
    )
    var resp = parse(resp_text)
    var content = resp.get("choices").at(0).get("message").get("content").as_string()
    var tokens = resp.get("usage").get("total_tokens").as_number()
    print("\nEXTRACTED content:", content)
    print("EXTRACTED total_tokens:", Int(tokens))
