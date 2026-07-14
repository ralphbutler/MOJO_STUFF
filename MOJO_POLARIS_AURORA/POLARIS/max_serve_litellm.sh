#!/usr/bin/env bash
# max_serve_litellm.sh
# Starts a MAX OpenAI-compatible server, sends ONE query through litellm,
# then shuts the server down. Mojo is not involved — this is pure serving/client plumbing.
# Compare with max_generate.sh (one-shot CLI generation, no server).
#
#   MAX (Metal GPU) ──serves──► http://localhost:8000/v1 ──called by──► litellm
#
# Run with:  ./max_serve_litellm.sh      (or:  bash max_serve_litellm.sh)

set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000
BASE="http://localhost:${PORT}/v1"
PROMPT="In one sentence, what is Mojo?"

echo "▶ Starting MAX server on :${PORT} (Metal, capped context)…  logs → max_serve.log"
uv run max serve \
  --model-path "$MODEL" \
  --max-length 4096 \
  --max-batch-size 1 \
  --port "$PORT" \
  > max_serve.log 2>&1 &
SERVER_PID=$!

# Always stop the server on exit (normal, error, or Ctrl-C).
cleanup() {
  echo "■ Stopping MAX server (pid $SERVER_PID)…"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "⏳ Waiting for the endpoint (first run compiles the model; ~20–60s)…"
for i in $(seq 1 180); do
  if curl -sf "${BASE}/models" >/dev/null 2>&1; then
    echo "✅ Server ready after ${i}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "❌ Server died during startup. Tail of max_serve.log:"; tail -25 max_serve.log; exit 1
  fi
  [ "$i" -eq 180 ] && { echo "❌ Timed out waiting for server."; tail -25 max_serve.log; exit 1; }
  sleep 1
done

echo "▶ Sending query through litellm: \"${PROMPT}\""
# --with litellm pulls litellm into an ephemeral overlay (doesn't touch pyproject).
uv run --with litellm python - "$PROMPT" <<'PY'
import sys, litellm
prompt = sys.argv[1]
resp = litellm.completion(
    model="openai/Qwen/Qwen2.5-0.5B-Instruct",  # "openai/" = talk OpenAI protocol; rest = model name MAX registered
    messages=[{"role": "user", "content": prompt}],
    api_base="http://localhost:8000/v1",
    api_key="not-needed",                        # MAX doesn't require a key by default
    max_tokens=128,
)
print("\n=== litellm → MAX response ===")
print(resp.choices[0].message.content)
print("\n=== usage:", resp.usage)
PY

echo "✔ Done (server will now be stopped by cleanup)."
