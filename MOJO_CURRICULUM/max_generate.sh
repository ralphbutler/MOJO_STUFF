#!/usr/bin/env bash
# max_generate.sh
# One-shot text generation with MAX — NO server, NO client protocol.
# `max generate` loads the model, runs the prompt, prints the completion, and exits.
# This is the "batch / offline inference" face of MAX; compare with max_serve_litellm.sh,
# which keeps a model resident behind an OpenAI-compatible HTTP endpoint.
# Mojo is not involved — MAX runs prebuilt kernels on the Apple GPU (Metal).
#
#   max serve    → long-lived endpoint, many requests   (production serving)
#   max generate → run once, print, exit                (quick check / batch)
#
# Run with:  ./max_generate.sh      (or:  bash max_generate.sh)

set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PROMPT="who was first US president?"

# --device-memory-utilization caps how much of the shared 128 GB MAX may reserve.
uv run max generate \
  --model-path "$MODEL" \
  --device-memory-utilization 0.5 \
  --prompt "$PROMPT"

# Alternative: bound context + batch size instead of a memory fraction.
# uv run max generate \
#   --model-path "$MODEL" \
#   --max-length 128 \
#   --max-batch-size 1 \
#   --prompt "$PROMPT"
