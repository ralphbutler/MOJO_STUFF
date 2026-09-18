#!/bin/bash
# matmul_sweep_check.sh — Mac: compile every matmul_sweep_configs.txt geometry with our spirv64 backend and
# validate the SPIR-V, WITHOUT Aurora. Catches invalid geometry (comptime asserts) and invalid
# SPIR-V before any queue time is spent. Leaves matmul_tunable.mojo back at the baseline.
#
#   bash matmul_sweep_check.sh [output dir]
#
# Needs spirv-val on PATH (brew install spirv-tools) and the fork at ../../modular.

set -uo pipefail
cd "$(dirname "$0")"
GPUDIR=$PWD
REPO=$(cd ../../modular && pwd)
OUT=${1:-$PWD/matmul_sweep_spv}
mkdir -p "$OUT"

printf '%-12s %-28s %8s %9s %10s  %s\n' TAG GEOMETRY THREADS ACC SPV_BYTES STATUS
rc=0
while read -r tag bm bn bk tm tn _rest; do
    case "$tag" in ''|\#*) continue ;; esac
    bash set_matmul_geometry.sh "BM=$bm" "BN=$bn" "BK=$bk" "TM=$tm" "TN=$tn" >/dev/null
    threads=$(( bm * bn / (tm * tn) ))
    acc=$(( tm * tn ))
    log="$OUT/$tag.compile.log"
    ( cd "$REPO" && ./bazelw run //Mojo:mojo -- run \
        --ignore-deprecated Pointer.load --ignore-deprecated Pointer.store \
        --ignore-deprecated Pointer.bitcast --ignore-deprecated Pointer.address_space_cast \
        --ignore-deprecated Pointer.__add__ \
        -I "$REPO/max/mojo" -I "$REPO/max/kernels/src" -I "$GPUDIR" \
        "$GPUDIR/emit_matmul_tunable.mojo" "$OUT/$tag" ) >"$log" 2>&1
    if [ ! -s "$OUT/$tag.spv" ]; then
        printf '%-12s %-28s %8s %9s %10s  %s\n' "$tag" "BM$bm BN$bn BK$bk TM$tm TN$tn" \
            "$threads" "$acc" "-" "COMPILE FAIL (see $log)"
        rc=1; continue
    fi
    bytes=$(stat -f %z "$OUT/$tag.spv" 2>/dev/null || stat -c %s "$OUT/$tag.spv")
    if spirv-val --target-env spv1.4 "$OUT/$tag.spv" >>"$log" 2>&1; then
        status="ok"
    else
        status="SPIRV-VAL FAIL"; rc=1
    fi
    printf '%-12s %-28s %8s %9s %10s  %s\n' "$tag" "BM$bm BN$bn BK$bk TM$tm TN$tn" \
        "$threads" "$acc" "$bytes" "$status"
done < matmul_sweep_configs.txt

bash set_matmul_geometry.sh BM=128 BN=128 BK=8 TM=8 TN=8 >/dev/null
echo
echo "== matmul_tunable.mojo restored to the baseline geometry"
exit $rc
