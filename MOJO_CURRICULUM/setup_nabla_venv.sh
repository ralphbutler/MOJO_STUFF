#!/usr/bin/env bash
# setup_nabla_venv.sh — build an ISOLATED venv for Nabla (Mojo-native training).
#
# Nabla is ALPHA and pins to Modular NIGHTLY, which would clash with the stable MAX
# in ./.venv. Giving it its own .venv-nabla means it can't destabilize the working
# MAX / Mojo setup. To undo everything: rm -rf .venv-nabla
#
# One-time macOS (Apple Silicon) prereq — the Metal toolchain:
#   xcode-select --install
#
# Run:  ./setup_nabla_venv.sh
# Then: .venv-nabla/bin/python train_nabla_mlp.py

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-nabla
echo "▶ Creating $VENV (isolated from ./.venv, which holds stable MAX)…"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip

# VERSION LOCK — do not "upgrade" these. Nabla's latest PyPI release (26.2251344, built
# 2026-02-25) declares only `modular>=26.2.0.dev2026021705` with NO upper bound. A bare
# install therefore pulls the newest modular nightly, which has drifted months past what
# Nabla was built against — backward() then dies with
#   AttributeError: 'ModuleOp' object has no attribute 'operation'
# So we pin BOTH to Nabla's Feb-2026 era. This is the alpha-ecosystem tax, in one line.
NABLA_VER="26.2251344"
MODULAR_PIN="26.2.0.dev2026021705"   # Nabla's own declared floor; matches its Feb-2026 build
echo "▶ Installing modular==${MODULAR_PIN} + nabla-ml==${NABLA_VER} (pinned; can take a while)…"
"$VENV/bin/pip" install --pre \
  --extra-index-url https://whl.modular.com/nightly/simple/ \
  "modular==${MODULAR_PIN}" "nabla-ml==${NABLA_VER}"
# If that exact nightly has been pruned from the index (pip: "no matching distribution"),
# try a nearby late-Feb-2026 build: modular==26.2.0.dev20260225XX (Nabla was built 02-25).

echo
echo "✅ venv built at $VENV"
"$VENV/bin/python" -c "import nabla as nb; print('nabla import OK:', getattr(nb, '__version__', '(no __version__)'))" \
  || echo "⚠️  import failed — read the install log above; nightly builds shift week to week."
echo "Next:  $VENV/bin/python train_nabla_mlp.py"
