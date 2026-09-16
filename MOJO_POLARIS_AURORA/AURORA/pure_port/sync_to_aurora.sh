#!/bin/bash
# sync_to_aurora.sh — push pure_port/ (and the shared docs) to the Aurora mirror.
# pre_process/ has its own copy of this script; neither touches the other's files.
#
# ONE ssh connection, so ONE MobilePASS+ code: the files go over as a single tar stream
# rather than one scp per file. (A flat `scp a b c dest/` also puts subdirectory files in
# the wrong place, which is the other reason this script exists.)
#
# ~/.ssh/config additionally keeps the authenticated connection alive for 8h
# (ControlMaster, added 2026-09-15), so repeated runs inside that window prompt for nothing.
# Drop it early with:  ssh -O exit rbutler@aurora.alcf.anl.gov
#
# Usage:  bash sync_to_aurora.sh            # code + docs
#         bash sync_to_aurora.sh --dry-run  # list what would be sent
#         bash sync_to_aurora.sh --code       # skip the docs

set -eu
cd "$(dirname "$0")/.."                       # the AURORA directory
HOST=${AURORA_HOST:-rbutler@aurora.alcf.anl.gov}
DEST=${AURORA_DEST:-/lus/flare/projects/ModCon/rbutler/MOJO_WORK/AURORA}
MODE=${1:-}

FILES=()
while IFS= read -r f; do FILES+=("$f"); done < <(
    ls pure_port/*.mojo pure_port/*.sh pure_port/*.pbs pure_port/*.cpp pure_port/*.py \
       pure_port/README.md pure_port/*.spv pure_port/*.spv.name pure_port/*.spv.args \
       pure_port/*.tar.gz 2>/dev/null
    find pure_port/mojo_intel_gpu -name '*.mojo' -o -name 'LICENSE' 2>/dev/null
    find pure_port/runtime pure_port/backend -type f 2>/dev/null
)
[ "$MODE" = "--code" ] || FILES+=(AURORA_PLAN.md AURORA_RESULTS.md)

if [ "$MODE" = "--dry-run" ]; then
    printf '%s\n' "${FILES[@]}"
    echo "--- ${#FILES[@]} files -> $HOST:$DEST"
    exit 0
fi

echo "== sending ${#FILES[@]} files to $HOST:$DEST (one connection)"
# --no-xattrs/--no-mac-metadata: macOS bsdtar otherwise embeds com.apple.provenance
# attributes that GNU tar on Aurora warns about once per file. COPYFILE_DISABLE stops
# the ._AppleDouble sidecars. All cosmetic, but the noise buries real errors.
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - "${FILES[@]}" | ssh "$HOST" "mkdir -p '$DEST' && tar xzf - -C '$DEST' && echo '== received:' && ls -lt '$DEST/pure_port' | head -8"
echo "== done"
