#!/bin/bash
# ocloc_check.sh — G5: does Intel's GPU compiler (IGC, via ocloc) accept SPIR-V emitted by
# our spirv64 Mojo backend? Compile-only, no GPU and no queue: run on aurora-uan-0007.
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash ocloc_check.sh [file.spv ...]   (default: g5_*.spv)

set -u
cd "$(dirname "$0")"
FILES=("$@")
[ ${#FILES[@]} -gt 0 ] || FILES=(g5_*.spv)

echo "== host: $(hostname)   ocloc: $(command -v ocloc)"
echo "== OCL driver: $(ocloc query OCL_DRIVER_VERSION 2>/dev/null | head -1)"
echo "== files: ${FILES[*]}"

rc=0
for spv in "${FILES[@]}"; do
    out="${spv%.spv}_pvc"
    echo
    echo "== $spv ($(stat -c %s "$spv") bytes) -> ocloc -device pvc"
    rm -f "$out"*
    ocloc compile -spirv_input -file "$spv" -device pvc -output "$out" -out_dir . 2>&1 | tail -8
    if ls "$out"* >/dev/null 2>&1; then
        ls -l "$out"*
        echo "   PASS: IGC built a PVC binary"
    else
        echo "   FAIL: no ocloc output"
        rc=1
    fi
done
exit $rc
