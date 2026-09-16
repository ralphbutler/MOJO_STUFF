#!/bin/bash
# build_kernels.sh — one command: unchanged Mojo source -> PVC-ready SPIR-V kernels, on Aurora.
#
#   <source>.mojo
#     -> mojo build --emit asm --target-accelerator apple-m4   (Metal AIR, one .ll per kernel)
#     -> air_roles.py          (hash-named sidecars -> stable role names, verified against a map)
#     -> air2spir.py           (AIR -> SPIR, + <role>.spv.args argument manifest)
#     -> clang -target spir64  (LLVM 22 bitcode)
#     -> llvm-spirv            (SPIR-V)
#     -> ocloc -device pvc     (validation: does IGC accept it?)
#
# Replaces g2_build_kernel.sh / g3_build_kernels.sh / g4_build_kernels.sh, which all started from
# .air.ll files generated on a Mac and copied over. Since G5.0 (Aurora emits byte-identical IR)
# the whole pipeline runs here, so the build starts from the .mojo source instead.
#
# Usage (on aurora-uan-0007, from MOJO_WORK/AURORA/gpu):
#   bash build_kernels.sh curriculum/04b_train_mlp_gpu.mojo g4 g4_roles.map
#   bash build_kernels.sh curriculum/02_vecadd_gpu.mojo      g2
#
# With no map, roles are the kernels' base names; that only works when they are unique, and the
# script tells you to author a map (air_roles.py survey) when they are not. A map that no longer
# matches the source is a hard error, never a silent mismapping.

set -u
cd "$(dirname "$0")"
HERE=$(pwd)

SRC=${1:-}
PREFIX=${2:-}
ROLEMAP=${3:-}
if [ -z "$SRC" ] || [ -z "$PREFIX" ]; then
    sed -n '2,25p' "$0"; exit 1
fi
[ -f "$SRC" ] || { echo "FATAL: no such source: $SRC"; exit 1; }

MOJO=$HERE/.venv/bin/mojo
PY=$HERE/.venv/bin/python
ACCEL=${ACCEL:-apple-m4}
CLANG="$CMPLR_ROOT/bin/compiler/clang"
LLVM_SPIRV="$CMPLR_ROOT/bin/compiler/llvm-spirv"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

echo "== host: $(hostname)   source: $SRC   prefix: $PREFIX   accel: $ACCEL"
"$MOJO" --version

# ---- 1. Mojo -> Metal AIR (one .ll per kernel instantiation) ----
echo
echo "== 1. emit kernel IR"
( cd "$WORK" && "$MOJO" build --emit asm --target-accelerator "$ACCEL" "$HERE/$SRC" -o host.s ) \
    2>&1 | grep -v "Crashpad" | sed 's/^/   /'
N_LL=$(ls "$WORK"/*.ll 2>/dev/null | wc -l)
echo "   sidecars: $N_LL"
[ "$N_LL" -gt 0 ] || { echo "FATAL: no kernel IR emitted (is 'max' installed in .venv?)"; exit 1; }

# ---- 2. sidecars -> stable role names ----
echo
echo "== 2. resolve roles"
if [ -n "$ROLEMAP" ]; then
    [ -f "$ROLEMAP" ] || { echo "FATAL: no such role map: $ROLEMAP"; exit 1; }
    ROLES=$("$PY" air_roles.py resolve "$WORK" "$ROLEMAP") || {
        echo "FATAL: role map does not match this source (see errors above)."; exit 2; }
    echo "   from $ROLEMAP:"
else
    ROLES=$("$PY" air_roles.py names "$WORK") || {
        echo "FATAL: kernel base names are ambiguous, so a role map is required."
        echo "       $PY air_roles.py survey <dir>   # author one, then pass it as argument 3"
        exit 2; }
    echo "   derived from base names (no map given):"
fi
echo "$ROLES" | sed 's/^/     /' | sed "s#$WORK/#   #"

# ---- 3..6. per role: AIR -> SPIR -> bitcode -> SPIR-V -> ocloc ----
fail=0
while read -r role path; do
    [ -n "$role" ] || continue
    base=${PREFIX}_${role}
    echo
    echo "================ $role ================"
    rm -f "$base".spir.ll "$base".bc "$base".spv "$base".spv.args "$base".spvt ${base}_pvc*
    cp "$path" "$base.air.ll"
    "$PY" air2spir.py "$base.air.ll" "$base.spir.ll" --name "$base" --const-as-global \
        --args-out "$base.spv.args" || { echo "  $role: air2spir FAILED"; fail=1; continue; }
    "$CLANG" -target spir64-unknown-unknown -x ir -c -emit-llvm "$base.spir.ll" -o "$base.bc" \
        || { echo "  $role: clang FAILED"; fail=1; continue; }
    "$LLVM_SPIRV" "$base.bc" -o "$base.spv" || { echo "  $role: llvm-spirv FAILED"; fail=1; continue; }
    echo "  spv: $(stat -c%s "$base.spv") bytes, args: $(grep -c '^arg ' "$base.spv.args") ($(grep '^arg ' "$base.spv.args" | awk '{print $3}' | sort | uniq -c | tr -s ' \n' ' '))"
    ocloc compile -spirv_input -file "$base.spv" -device pvc -output ${base}_pvc -out_dir . 2>&1 \
        | grep -Ev "^$" | tail -2 | sed 's/^/  ocloc: /'
    ls ${base}_pvc* >/dev/null 2>&1 && echo "  $role: OK" || { echo "  $role: ocloc produced no binary"; fail=1; }
done <<< "$ROLES"

echo
echo "================ summary ================"
ls -l ${PREFIX}_*.spv ${PREFIX}_*.spv.args 2>/dev/null
echo
echo "built $(ls ${PREFIX}_*.spv 2>/dev/null | wc -l) kernels from $SRC"
echo "overall: $([ $fail -eq 0 ] && echo 'ALL OK' || echo 'SOME FAILED (see above)')"
exit $fail
