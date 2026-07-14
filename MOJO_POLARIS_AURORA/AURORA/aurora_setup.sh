#!/bin/bash
# ── Aurora Mojo LOGIN-node prep — run on an Aurora LOGIN node (direct outbound) ──
# Installs uv and verifies live ModCon storage. Does NOT install Mojo: Aurora's glibc 2.31
# is too old for the manylinux_2_34 Mojo wheels, so Mojo is installed inside a container on a
# compute node (aurora_build.pbs). The host uv installed here is reused inside that container.
#
# STORAGE (fixed 2026-07-08 from login-node recon — see FROM_POLARIS.md):
#   Everything lives on the LIVE ModCon project FS (/lus/flare, 1000T quota).
#   The old ~/.cache symlink used to point into the DEAD candle_aesp_CNDA
#   allocation (over a zeroed quota: Used 60.67T* / Quota 1M), which would make
#   `uv sync` die with "Disk quota exceeded" mid-extract. That symlink has since
#   been repointed to a live ModCon dotcache dir. We still set UV_CACHE_DIR
#   explicitly so the multi-GB toolchain extract stays under mojo-aurora/.
set -e

# ── Live storage target (ModCon) ──────────────────────────────────────────────
# PROJROOT holds the kit sources, the uv project (pyproject.toml), and the .venv
# all in one place — mirrors the Mac's MOJO_POLARIS_AURORA/AURORA layout.
PROJ=ModCon
PROJROOT=/lus/flare/projects/$PROJ/$USER/MOJO_WORK/AURORA
export UV_CACHE_DIR="$PROJROOT/.uv_cache"        # multi-GB toolchain extract lands here

# 0. Preflight: record what we're on.
echo "== preflight =="
grep -E "^(NAME|VERSION)=" /etc/os-release || true
ldd --version | head -1 || true
echo "target : $PROJROOT"
echo

# 0a. Prove we can actually WRITE to the live project FS before doing anything.
mkdir -p "$PROJROOT" "$UV_CACHE_DIR"
if ! touch "$PROJROOT/.writetest" 2>/dev/null; then
  echo "FATAL: cannot write to $PROJROOT — check 'myquota' for ModCon." >&2
  exit 1
fi
rm -f "$PROJROOT/.writetest"
echo "write test OK on ModCon"
echo

# 1. Outbound proxy — test direct https first; fall back to the ALCF proxy if blocked.
if curl -sI --max-time 8 https://astral.sh >/dev/null 2>&1; then
  echo "outbound https: direct OK (no proxy needed)"
else
  echo "outbound https: direct blocked -> enabling ALCF proxy"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export ftp_proxy="http://proxy.alcf.anl.gov:3128"
fi
echo

# 2. Install uv (standalone, no root). Binary -> ~/.local/bin (on /home, has room).
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 3. DO NOT install Mojo on the bare host — it CANNOT work on Aurora.
#    Every Modular wheel (mojo/max/mojo-compiler, all versions) is manylinux_2_34 and needs
#    glibc >= 2.34; Aurora is glibc 2.31. `uv add mojo` here fails with:
#      "... only has wheels for: manylinux_2_34_x86_64 ..."
#    The install runs INSIDE an apptainer container (Ubuntu 24.04 / glibc 2.39) on a COMPUTE
#    node — see aurora_build.pbs. The host uv installed above IS reused inside that container
#    (auto-bound via $HOME), so this login prep is still needed.
cd "$PROJROOT"

echo
echo "Login prep done (uv installed; ModCon storage verified)."
echo "Next steps — on a COMPUTE node (batch; has both user-namespaces AND proxied internet):"
echo "  qsub aurora_build.pbs      # pull ubuntu2404.sif (first time) + install Mojo into .venv"
echo "  qsub aurora_l0.pbs         # L0 smoke test"
echo "  qsub -v MOJOFILE=<f>.mojo aurora_run.pbs   # generic runner for the CPU ladder"
echo "See AURORA_PORT.md / AURORA_RESULTS.md for the full story."
