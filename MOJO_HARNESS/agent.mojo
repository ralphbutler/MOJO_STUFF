"""The agent: a stateful conversation with a `bash` tool.

`Agent` holds the running conversation as a native `JSONValue` message array so
context persists across REPL turns. Each `send()` runs the tool loop: stream the
completion (printing reasoning + content live); if the reply carries
`tool_calls`, run each via the native `popen` runner, append a `tool` result,
and resend; when the model answers with plain content it has already streamed to
the terminal, so we just record it in history and return.

The completion is streamed (`sse.stream_message`), which also reassembles
tool_call fragments that arrive spread across SSE chunks. The model is chosen at
runtime by the caller, so different endpoints/models need no recompile.
"""

from json import JSONValue, parse, serialize
from proc import run_capture
from sse import stream_message
from ui import cyan, green, yellow, dim, bold, magenta


comptime MAX_TURNS: Int = 8


def make_message(role: String, content: String) -> JSONValue:
    var m = JSONValue.object()
    m.set("role", JSONValue.string(role))
    m.set("content", JSONValue.string(content))
    return m^


def build_bash_tool() -> JSONValue:
    """The OpenAI-style function schema for a single `bash` tool."""
    var cmd = JSONValue.object()
    cmd.set("type", JSONValue.string("string"))
    cmd.set("description", JSONValue.string("The bash command to execute."))

    var props = JSONValue.object()
    props.set("command", cmd^)

    var required = JSONValue.array()
    required.append(JSONValue.string("command"))

    var params = JSONValue.object()
    params.set("type", JSONValue.string("object"))
    params.set("properties", props^)
    params.set("required", required^)

    var func = JSONValue.object()
    func.set("name", JSONValue.string("bash"))
    func.set("description", JSONValue.string(
        "Run a bash command on the user's machine and return its combined"
        " stdout and stderr."
    ))
    func.set("parameters", params^)

    var tool = JSONValue.object()
    tool.set("type", JSONValue.string("function"))
    tool.set("function", func^)
    return tool^


def _run_bash(command: String) raises -> String:
    """Execute a command capturing stdout+stderr via the native popen runner."""
    return run_capture(command)


struct Agent(Movable):
    var messages: JSONValue
    var tools: JSONValue
    var model: String
    var reasoning_mode: Int

    def __init__(out self, system: String, model: String, reasoning_mode: Int):
        self.messages = JSONValue.array()
        self.messages.append(make_message("system", system))
        self.tools = JSONValue.array()
        self.tools.append(build_bash_tool())
        self.model = model
        self.reasoning_mode = reasoning_mode

    def _complete(self) raises -> JSONValue:
        """Stream one completion (printing live); return the assistant message."""
        var body = JSONValue.object()
        body.set("model", JSONValue.string(self.model))
        body.set("messages", self.messages.copy())
        body.set("tools", self.tools.copy())
        body.set("stream", JSONValue.bool(True))
        body.set("temperature", JSONValue.number(0.2))

        return stream_message(
            "localhost:1234", 127, 0, 0, 1, 1234,
            "/v1/chat/completions", serialize(body), self.reasoning_mode,
        )

    def send(mut self, user: String) raises:
        """Run the tool loop for one user turn; content streams live."""
        self.messages.append(make_message("user", user))

        for turn in range(MAX_TURNS):
            var message = self._complete()

            var has_calls = False
            if message.has("tool_calls"):
                var tcs = message.get("tool_calls")
                if tcs.count() > 0:
                    has_calls = True
                    self.messages.append(message.copy())  # echo tool-call msg
                    for i in range(tcs.count()):
                        var tc = tcs.at(i)
                        var call_id = tc.get("id").as_string()
                        var func = tc.get("function")
                        var name = func.get("name").as_string()
                        var args = parse(func.get("arguments").as_string())
                        var command = args.get("command").as_string()

                        print(yellow("  🔧 " + name) + dim(" · turn " + String(turn + 1)))
                        print(dim("     $ ") + command)
                        var output = _run_bash(command)
                        print(dim(_indent(output, "     ")))

                        var tool_msg = JSONValue.object()
                        tool_msg.set("role", JSONValue.string("tool"))
                        tool_msg.set("tool_call_id", JSONValue.string(call_id))
                        tool_msg.set("content", JSONValue.string(output))
                        self.messages.append(tool_msg^)

            if not has_calls:
                # content has already streamed to the terminal in _complete
                self.messages.append(message.copy())
                return

        print(magenta("  [reached MAX_TURNS without a final answer]"))


def _indent(text: String, prefix: String) -> String:
    """Prefix every line of tool output so it reads as a block."""
    var out = String("")
    var line = String("")
    for cp in text.strip().codepoint_slices():
        var c = String(cp)
        if c == "\n":
            out += prefix + line + "\n"
            line = String("")
        else:
            line += c
    out += prefix + line
    return out
