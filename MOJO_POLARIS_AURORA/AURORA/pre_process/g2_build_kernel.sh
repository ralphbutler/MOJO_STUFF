#!/bin/bash
# g2_build_kernel.sh — Mojo-written kernel -> SPIR-V, on aurora-uan-0007 (no queue).
#
# Input:  g2_vecadd.air.ll — Mojo's Metal-backend kernel IR for the UNCHANGED
#         MOJO_CURRICULUM/02_vecadd_gpu.mojo, generated on the Mac with:
#           mojo build --emit asm --target-accelerator apple-m4 02_vecadd_gpu.mojo -o host.s
#         (the kernel lands in a *_vecadd_kernel_*.ll sidecar next to host.s)
# Steps:  air2spir.py (AIR -> SPIR text IR)
#         -> oneAPI clang (text IR -> LLVM 22 bitcode; this is the version-skew check)
#         -> llvm-spirv (bitcode -> SPIR-V)
#         -> ocloc -spirv_input (IGC compiles it for PVC: acts as our validator)
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash g2_build_kernel.sh 2>&1 | tee g2_build.txt

set -u
cd "$(dirname "$0")"
PY=./.venv/bin/python
CLANG="$CMPLR_ROOT/bin/compiler/clang"
LLVM_SPIRV="$CMPLR_ROOT/bin/compiler/llvm-spirv"
rm -f g2_vecadd.spir.ll g2_vecadd.bc g2_vecadd.spv g2_vecadd.spvt g2_vecadd_pvc*

echo "== host"; hostname; grep VERSION= /etc/os-release
echo "== python: $($PY --version)"

echo
echo "== 1. air2spir"
$PY air2spir.py g2_vecadd.air.ll g2_vecadd.spir.ll --name vecadd --const-as-global || exit 1

echo
echo "== 2. clang: SPIR text IR -> bitcode (LLVM 22 parser)"
"$CLANG" -target spir64-unknown-unknown -x ir -c -emit-llvm g2_vecadd.spir.ll -o g2_vecadd.bc || { echo "  FAILED"; exit 1; }
echo "  OK: g2_vecadd.bc ($(stat -c%s g2_vecadd.bc) bytes)"

echo
echo "== 3. llvm-spirv: bitcode -> SPIR-V"
"$LLVM_SPIRV" g2_vecadd.bc -o g2_vecadd.spv || { echo "  FAILED"; exit 1; }
echo "  OK: g2_vecadd.spv ($(stat -c%s g2_vecadd.spv) bytes), magic: $(od -A n -t x1 -N 4 g2_vecadd.spv | tr -s ' ')"
"$LLVM_SPIRV" -to-text g2_vecadd.spv -o g2_vecadd.spvt
echo "  capabilities / entry point / builtins:"
grep -E "Capability|EntryPoint|ExecutionMode|BuiltIn" g2_vecadd.spvt | head -20

echo
echo "== 4. ocloc: compile SPIR-V for PVC (validation)"
ocloc compile -spirv_input -file g2_vecadd.spv -device pvc -output g2_vecadd_pvc -out_dir . 2>&1 | tail -5
ls -l g2_vecadd_pvc* 2>/dev/null || echo "  (no ocloc output file)"

echo
echo "== G2 kernel ready: g2_vecadd.spv"
