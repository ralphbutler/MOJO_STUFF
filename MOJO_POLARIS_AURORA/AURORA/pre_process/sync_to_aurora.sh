#!/bin/bash
# sync_to_aurora.sh — push pre_process/ to the Aurora mirror.
# A copy of pure_port/sync_to_aurora.sh, kept separate so either directory can be
# deleted without breaking the other.
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

set -eu
cd "$(dirname "$0")/.."                       # the AURORA directory
HOST=${AURORA_HOST:-rbutler@aurora.alcf.anl.gov}
DEST=${AURORA_DEST:-/lus/flare/projects/ModCon/rbutler/MOJO_WORK/AURORA}
MODE=${1:-}

FILES=()
while IFS= read -r f; do FILES+=("$f"); done < <(
    ls pre_process/*.mojo pre_process/*.sh pre_process/*.pbs pre_process/*.cpp \
       pre_process/*.py pre_process/*.cl pre_process/*.map pre_process/README.md \
       pre_process/*.air.ll 2>/dev/null
    find pre_process/mojo_intel_gpu -name '*.mojo' -o -name 'LICENSE' 2>/dev/null
)

if [ "$MODE" = "--dry-run" ]; then
    printf '%s\n' "${FILES[@]}"
    echo "--- ${#FILES[@]} files -> $HOST:$DEST"
    exit 0
fi

echo "== sending ${#FILES[@]} files to $HOST:$DEST (one connection)"
# --no-xattrs/--no-mac-metadata: macOS bsdtar otherwise embeds com.apple.provenance
# attributes that GNU tar on Aurora warns about once per file. COPYFILE_DISABLE stops
# the ._AppleDouble sidecars. All cosmetic, but the noise buries real errors.
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf - "${FILES[@]}" | ssh "$HOST" "mkdir -p '$DEST' && tar xzf - -C '$DEST' && echo '== received:' && ls -lt '$DEST/pre_process' | head -8"
echo "== done"
