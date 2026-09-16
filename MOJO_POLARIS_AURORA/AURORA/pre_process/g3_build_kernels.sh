#!/bin/bash
# g3_build_kernels.sh — build the G3 matmul kernels (Mojo Metal AIR -> SPIR-V) on aurora-uan-0007.
#
# Inputs (generated on the Mac from UNCHANGED MOJO_CURRICULUM sources, see README.md):
#   g3_naive.air.ll  (03a)   g3_tiled.air.ll (03b)   g3_coarse.air.ll (03c)   g3_check.air.ll (03d)
# For each: air2spir.py -> clang -target spir64 (LLVM 22 bitcode) -> llvm-spirv -> ocloc PVC compile.
# Outputs: g3_<v>.spv and g3_<v>.spv.args (argument manifest read by g3_matmul_lz.mojo).
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash g3_build_kernels.sh 2>&1 | tee g3_build.txt

set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
CLANG="$CMPLR_ROOT/bin/compiler/clang"
LLVM_SPIRV="$CMPLR_ROOT/bin/compiler/llvm-spirv"

echo "== host"; hostname; grep VERSION= /etc/os-release

build() {
    local v=$1
    local base=g3_$v
    echo
    echo "================ $v ================"
    rm -f $base.spir.ll $base.bc $base.spv $base.spv.args $base.spvt ${base}_pvc*
    $PY air2spir.py $base.air.ll $base.spir.ll --name matmul_$v --const-as-global \
        --args-out $base.spv.args || { echo "  $v: air2spir FAILED"; return 1; }
    "$CLANG" -target spir64-unknown-unknown -x ir -c -emit-llvm $base.spir.ll -o $base.bc \
        || { echo "  $v: clang FAILED"; return 1; }
    "$LLVM_SPIRV" $base.bc -o $base.spv || { echo "  $v: llvm-spirv FAILED"; return 1; }
    "$LLVM_SPIRV" -to-text $base.spv -o $base.spvt
    echo "  spv: $(stat -c%s $base.spv) bytes, magic: $(od -A n -t x1 -N 4 $base.spv | tr -s ' ')"
    echo "  capabilities / entry / storage:"
    grep -E "Capability |EntryPoint|ExecutionMode" $base.spvt | sed 's/^/    /'
    echo "  barrier calls: $(grep -c barrier $base.spvt)   Workgroup refs: $(grep -c Workgroup $base.spvt)"
    ocloc compile -spirv_input -file $base.spv -device pvc -output ${base}_pvc -out_dir . 2>&1 | tail -3 | sed 's/^/  ocloc: /'
    ls ${base}_pvc* >/dev/null 2>&1 && echo "  $v: OK" || echo "  $v: ocloc produced no binary"
}

for v in naive check tiled coarse; do
    build $v
done
echo
ls -l g3_*.spv g3_*.spv.args
