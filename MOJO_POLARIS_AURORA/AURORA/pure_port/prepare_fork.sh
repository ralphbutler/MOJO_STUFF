#!/bin/bash
# prepare_fork.sh — aurora-uan-0007: set up the forked open-source Mojo compiler source for the
# Linux build (build_fork.pbs). Clones github.com/modular/modular at the commit the Mac fork is
# based on, applies fork_overlay.tar.gz (our spirv64 backend patches), and writes a local.bazelrc
# that keeps every Bazel cache on /lus/flare (home quotas are too small).
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash prepare_fork.sh
# Re-running re-applies the overlay (use this after each new overlay sync).

set -euo pipefail
W=/lus/flare/projects/ModCon/rbutler/MOJO_WORK
REPO=$W/modular
BASE_COMMIT=6417db28ceef430d067755e9826cec65e19f9333
OVERLAY="$(cd "$(dirname "$0")" && pwd)/fork_overlay.tar.gz"

echo "== host $(hostname), $(date)"
[ -f "$OVERLAY" ] || { echo "error: $OVERLAY missing (run sync_to_aurora.sh)" >&2; exit 1; }

if [ ! -d "$REPO/.git" ]; then
    echo "== cloning modular at $BASE_COMMIT into $REPO"
    mkdir -p "$REPO"
    cd "$REPO"
    git init -q
    git remote add origin https://github.com/modular/modular.git
    git fetch -q --depth 1 origin "$BASE_COMMIT"
    git checkout -q FETCH_HEAD
else
    cd "$REPO"
fi
[ "$(git rev-parse HEAD)" = "$BASE_COMMIT" ] || { echo "error: $REPO is not at $BASE_COMMIT" >&2; exit 1; }

echo "== applying overlay $(sha256sum "$OVERLAY" | cut -c1-16)…"
tar xzf "$OVERLAY" -C "$REPO"
tar tzf "$OVERLAY" | sed 's/^/     /'

cat > "$REPO/local.bazelrc" <<EOF
# Written by prepare_fork.sh (Aurora): build the compiler from source; keep caches on Lustre.
startup --output_user_root=$W/bazel/output_root
build --config=build-mojo
build --disk_cache=$W/bazel/disk_cache
common --repository_cache=$W/bazel/repo_cache
EOF
mkdir -p "$W/bazel"
echo "== local.bazelrc:"
sed 's/^/     /' "$REPO/local.bazelrc"
echo "== ready: qsub build_fork.pbs"
