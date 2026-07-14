"""Entry point for mojo-harness — a multi-turn REPL over a native Agent.

Everything under the hood is native Mojo: libc sockets (`net.mojo`), a
hand-rolled JSON parser/serializer (`json.mojo`), an HTTP client (`http.mojo`),
streaming (`sse.mojo`), and the tool loop (`agent.mojo`). No Python, no TLS.
Talks to LM Studio at 127.0.0.1:1234.

Run:  uv run mojo run harness.mojo [model]
The model is picked as: CLI arg > $MOJO_HARNESS_MODEL > the default below.
"""

from std.sys import argv
from std.os import getenv
from agent import Agent
from http import http_get
from json import parse
from ui import banner, read_line, cyan, bold, dim, green, yellow
from ui import THINK_OFF, THINK_COMPACT, THINK_FULL


comptime DEFAULT_MODEL: String = "qwen-agentworld-35b-a3b"

comptime SYSTEM: String = (
    "You are mojo-harness, a terse coding assistant running on the user's macOS"
    " machine. You have a `bash` tool: when a task needs the filesystem or"
    " shell, call it rather than guessing. Prefer one command at a time. Keep"
    " final answers to a sentence or two."
)


def resolve_model() raises -> String:
    var args = argv()
    if len(args) > 1:
        return String(args[1])
    var env = getenv("MOJO_HARNESS_MODEL")
    if env.byte_length() > 0:
        return env
    return String(DEFAULT_MODEL)


def resolve_think() raises -> Int:
    """$MOJO_HARNESS_THINK = off | compact | full (default compact)."""
    var v = getenv("MOJO_HARNESS_THINK")
    if v == "off":
        return THINK_OFF
    if v == "full":
        return THINK_FULL
    return THINK_COMPACT


def _think_label(mode: Int) -> String:
    if mode == THINK_OFF:
        return String("off")
    if mode == THINK_FULL:
        return String("full")
    return String("compact")


def warn_if_model_missing(model: String):
    """`/v1/models` lists all *downloaded* models (not just loaded ones). If the
    requested id isn't among them LM Studio can't load it, and will silently
    answer from whatever model is currently loaded — so warn loudly. (A
    downloaded-but-not-loaded model is fine: LM Studio JIT-loads it on request.)
    Non-fatal: any failure is swallowed — the REPL surfaces connect errors."""
    try:
        var body = http_get("localhost:1234", 127, 0, 0, 1, 1234, "/v1/models")
        var doc = parse(body)
        if not doc.has("data"):
            return
        var data = doc.get("data")
        var found = False
        var ids = String("")
        for i in range(data.count()):
            var id = data.at(i).get("id").as_string()
            if ids.byte_length() > 0:
                ids += ", "
            ids += id
            if id == model:
                found = True
        if not found:
            print(yellow(bold("  ⚠ '" + model + "' is not a downloaded model in LM Studio.")))
            print(yellow("    It can't be loaded; LM Studio will answer from whatever model is loaded."))
            print(dim("    available: " + ids) + "\n")
    except e:
        print(dim("  (could not reach LM Studio to verify the model list)\n"))


def main() raises:
    banner()
    var model = resolve_model()
    var think = resolve_think()
    print(dim("model: " + model + "  ·  thinking: " + _think_label(think)) + "\n")
    warn_if_model_missing(model)
    var agent = Agent(SYSTEM, model, think)

    while True:
        var line = read_line(cyan(bold("you › ")))
        if not line:
            print(dim("\nbye."))    # EOF (Ctrl-D / end of piped input)
            break
        var task = String(line.value().strip())

        if task == "":
            continue
        if task == "exit" or task == "quit":
            print(dim("bye."))
            break

        agent.send(task)
        print("")
