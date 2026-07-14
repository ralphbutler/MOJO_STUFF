# 🔩 mojo-harness

A tiny LLM coding agent — a minimal `pi`/Claude-Code — written **natively in
Mojo**. It sends a chat request to a local LLM, streams the reply, and when the
model asks to run a tool, it shells out to `bash`, feeds the result back, and
loops.

The point is to *show off Mojo*: **all the code is Mojo** — no Python, no
third-party libraries. The only non-Mojo things it touches are about ten
operating-system calls through the C library
(`socket/connect/send/recv/close`, `popen/fread/pclose`, `read`, `getpid`) — the
same OS interface every native program uses to reach the network, run a process,
or read input. No Python interpreter is ever loaded; networking, JSON, HTTP, SSE
streaming, and the shell tool are all hand-written Mojo. No TLS (targets plain
HTTP on localhost by design).

## ✅ Requirements

- **macOS on Apple Silicon.** The socket layer hand-builds a BSD/Darwin
  `sockaddr_in` (with `sin_len`); Linux's layout differs slightly, so it would
  need a small tweak in `net.mojo`.
- **[uv](https://docs.astral.sh/uv/)** — the Python packaging tool. It fetches
  the Mojo toolchain for you; you do **not** need to install Mojo separately.
- **A local OpenAI-compatible LLM endpoint at `http://127.0.0.1:1234`** —
  e.g. **[LM Studio](https://lmstudio.ai/)** with its local server on. Use a
  model that supports **tool calling** (see below).

## 🚀 Getting started (with uv)

Install uv (if you don't have it):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this folder, just run it — the first run creates a virtualenv and
downloads the Mojo toolchain automatically (needs network once):

```bash
uv run mojo run harness.mojo
```

Start your LLM server first (LM Studio serving on `1234`, a tool-capable model
available). Type a task at the `you ›` prompt (the terminal's usual line editing —
Backspace, Ctrl-W, Ctrl-U — works); `exit` / `quit` / Ctrl-D to leave.

```
you › how many files here end in .mojo?
  🔧 bash · turn 1
     $ find . -name "*.mojo" | wc -l
     14
  🤖 There are 14 files ending in `.mojo`.
```

> ⚠️ The `bash` tool runs **real, unsandboxed** commands on your machine. This is
> a demo — point it at throwaway tasks, not anything destructive.

## 🧠 Choosing a model

The model is resolved as **CLI arg › `$MOJO_HARNESS_MODEL` › default** (default
is `qwen-agentworld-35b-a3b`; change `DEFAULT_MODEL` in `harness.mojo`):

```bash
uv run mojo run harness.mojo                              # default
uv run mojo run harness.mojo your-model-id                # arg
MOJO_HARNESS_MODEL=your-model-id uv run mojo run harness.mojo   # env var
```

The harness never *loads* models — it only names one. With LM Studio's
Just-In-Time loading (on by default) a downloaded model is auto-loaded on first
request. If the id isn't a downloaded model at all, LM Studio silently answers
from whatever is loaded — so at startup the harness checks `/v1/models` and warns
if your id isn't there. A mixture-of-experts model (few active params) tends to
be fast; a heavy "reasoning" model can be slow because it generates lots of
hidden thinking before every answer.

## 💭 Reasoning display

Control how much of the model's thinking you see with `$MOJO_HARNESS_THINK`:

| Value | Behavior |
|---|---|
| `compact` (default) | A single self-updating `💭 thinking… N` line that clears when the answer starts. |
| `off` | Silent until the tool call / answer. |
| `full` | Stream the whole reasoning, dimmed. |

```bash
MOJO_HARNESS_THINK=off uv run mojo run harness.mojo
```

## 🧱 Architecture

Core loop: assemble messages → POST to the LLM → parse the reply → on a tool
call, run the tool and loop. Everything native:

| File | Role |
|---|---|
| `net.mojo` | libc TCP via `external_call`: socket/connect/send/recv/close; hand-built Darwin `sockaddr_in`. |
| `json.mojo` | Hand-rolled JSON: recursive `JSONValue` + recursive-descent parser + serializer. |
| `http.mojo` | Minimal HTTP/1.1 client (`http_post_json`, `http_get`). |
| `sse.mojo` | Streaming (SSE) client; reassembles fragmented tool-call deltas. |
| `proc.mojo` | `bash` tool via libc `popen`/`pclose`. |
| `ui.mojo` | ANSI colors + a native `read_line` (terminal line editing via the kernel). |
| `agent.mojo` | `Agent`: stateful conversation + the tool-calling loop. |
| `harness.mojo` | Entry point: the multi-turn REPL. |
| `scratch_*.mojo` | Standalone per-step demos (not part of the harness). |

## 🧪 Per-step demos

Each is a self-contained program showing one capability (they import the real
modules; nothing imports them):

```bash
uv run mojo run scratch_get_models.mojo   # 1: raw GET over libc sockets
uv run mojo run scratch_json.mojo         # 2: JSON round-trip
uv run mojo run scratch_chat.mojo         # 3: non-streaming completion
uv run mojo run scratch_stream.mojo       # 4: streaming completion
```

## 📌 Notes

- Built and tested on **Mojo 1.0.0b2** (a fast-moving beta; newer versions may
  differ). The endpoint host/port (`127.0.0.1:1234`) is currently hardcoded.
- This is a feasibility experiment, not a production agent — small on purpose.
