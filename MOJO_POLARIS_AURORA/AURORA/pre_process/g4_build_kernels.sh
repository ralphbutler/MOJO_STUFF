#!/bin/bash
# g4_build_kernels.sh — build the G4 MLP-training kernels (Mojo Metal AIR -> SPIR-V) on aurora-uan-0007.
#
# Inputs (generated on the Mac from the UNCHANGED MOJO_CURRICULUM/04b_train_mlp_gpu.mojo, see README.md):
#   g4_<role>.air.ll for the 12 distinct kernels behind 04b's 15 launches per epoch.
#   The 4 SGD launches share one byte-identical kernel; the other repeated generics differ
#   only in their layout strides (fwd1 vs fwd2, dW1 vs dW2, db1 vs db2).
# For each: air2spir.py -> clang -target spir64 (LLVM 22 bitcode) -> llvm-spirv -> ocloc PVC compile.
# Outputs: g4_<role>.spv and g4_<role>.spv.args (argument manifest read by g4_train_mlp_lz.mojo).
#
# Unlike G3 these kernels use no shared memory and no barriers, so no `local` args appear in
# the manifests — only `buffer` (TileTensor pointer holders) and `const` (Int32/Float32 holders).
# New in air2spir for this rung: @air.max.s.i64 -> inline icmp+select (the K-loop trip guard).
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash g4_build_kernels.sh 2>&1 | tee g4_build.txt

set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
CLANG="$CMPLR_ROOT/bin/compiler/clang"
LLVM_SPIRV="$CMPLR_ROOT/bin/compiler/llvm-spirv"

ROLES="fwd1 biasrelu fwd2 addb2 dz2 dW2 db2 da1 relugrad dW1 db1 sgd"

echo "== host"; hostname; grep VERSION= /etc/os-release

fail=0
build() {
    local v=$1
    local base=g4_$v
    echo
    echo "================ $v ================"
    rm -f $base.spir.ll $base.bc $base.spv $base.spv.args $base.spvt ${base}_pvc*
    $PY air2spir.py $base.air.ll $base.spir.ll --name g4_$v --const-as-global \
        --args-out $base.spv.args || { echo "  $v: air2spir FAILED"; fail=1; return 1; }
    "$CLANG" -target spir64-unknown-unknown -x ir -c -emit-llvm $base.spir.ll -o $base.bc \
        || { echo "  $v: clang FAILED"; fail=1; return 1; }
    "$LLVM_SPIRV" $base.bc -o $base.spv || { echo "  $v: llvm-spirv FAILED"; fail=1; return 1; }
    "$LLVM_SPIRV" -to-text $base.spv -o $base.spvt
    echo "  spv: $(stat -c%s $base.spv) bytes, magic: $(od -A n -t x1 -N 4 $base.spv | tr -s ' ')"
    echo "  capabilities / entry:"
    grep -E "Capability |EntryPoint" $base.spvt | sed 's/^/    /'
    echo "  args: $(grep -c '^arg ' $base.spv.args) ($(grep '^arg ' $base.spv.args | awk '{print $3}' | sort | uniq -c | tr -s ' \n' ' '))"
    ocloc compile -spirv_input -file $base.spv -device pvc -output ${base}_pvc -out_dir . 2>&1 | tail -3 | sed 's/^/  ocloc: /'
    ls ${base}_pvc* >/dev/null 2>&1 && echo "  $v: OK" || { echo "  $v: ocloc produced no binary"; fail=1; }
}

for v in $ROLES; do
    build $v
done

echo
echo "================ summary ================"
ls -l g4_*.spv g4_*.spv.args
echo
echo "expected 12 .spv and 12 .spv.args; got $(ls g4_*.spv 2>/dev/null | wc -l) and $(ls g4_*.spv.args 2>/dev/null | wc -l)"
echo "overall: $([ $fail -eq 0 ] && echo ALL OK || echo 'SOME FAILED (see above)')"
