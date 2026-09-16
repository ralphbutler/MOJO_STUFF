#!/bin/bash
# g1_build_kernel.sh — compile g1_vecadd.cl -> g1_vecadd.spv on aurora-uan-0007.
#
# Tries two routes and reports both:
#   A) oneAPI clang -> LLVM bitcode -> llvm-spirv      (the route G2 will reuse)
#   B) ocloc -spv_only                                 (Intel's own OpenCL compiler)
# Route A's output becomes g1_vecadd.spv if it works; otherwise route B's.
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash g1_build_kernel.sh 2>&1 | tee g1_build.txt

set -u
cd "$(dirname "$0")"
CL=g1_vecadd.cl
rm -f g1_vecadd_A.bc g1_vecadd_A.spv g1_vecadd_B.spv g1_vecadd.spv g1_vecadd_A.spvt

echo "== host";  hostname; grep VERSION= /etc/os-release
echo "== CMPLR_ROOT=$CMPLR_ROOT"

LLVM_SPIRV="$CMPLR_ROOT/bin/compiler/llvm-spirv"
CLANG=""
for c in "$CMPLR_ROOT/bin/compiler/clang" "$CMPLR_ROOT/bin/clang"; do
    if [ -x "$c" ]; then CLANG="$c"; break; fi
done
echo "== clang      : ${CLANG:-<not found>}"
[ -n "$CLANG" ] && "$CLANG" --version | head -1
echo "== llvm-spirv : $LLVM_SPIRV"
"$LLVM_SPIRV" --version | head -2

magic() {  # print the first 4 bytes; SPIR-V starts 03 02 23 07
    od -A n -t x1 -N 4 "$1" | tr -s ' '
}

echo
echo "== Route A: clang -> bitcode -> llvm-spirv"
if [ -n "$CLANG" ] && \
   "$CLANG" -cl-std=CL3.0 -target spir64-unknown-unknown -O2 \
            -Xclang -finclude-default-header -Xclang -fdeclare-opencl-builtins \
            -emit-llvm -c "$CL" -o g1_vecadd_A.bc && \
   "$LLVM_SPIRV" g1_vecadd_A.bc -o g1_vecadd_A.spv; then
    echo "  OK: g1_vecadd_A.spv ($(stat -c%s g1_vecadd_A.spv) bytes), magic:$(magic g1_vecadd_A.spv)"
    "$LLVM_SPIRV" -to-text g1_vecadd_A.spv -o g1_vecadd_A.spvt && \
        echo "  entry points:" && grep -E "EntryPoint|ExecutionMode" g1_vecadd_A.spvt | head -5
else
    echo "  FAILED"
fi

echo
echo "== Route B: ocloc -spv_only"
ocloc compile -file "$CL" -device pvc -spv_only -output g1_vecadd_B -output_no_suffix -out_dir . 2>&1 | tail -5
ls -l g1_vecadd_B* 2>/dev/null
if [ -f g1_vecadd_B.spv ]; then
    echo "  OK: g1_vecadd_B.spv, magic:$(magic g1_vecadd_B.spv)"
elif [ -f g1_vecadd_B ]; then
    mv g1_vecadd_B g1_vecadd_B.spv
    echo "  OK (renamed): g1_vecadd_B.spv, magic:$(magic g1_vecadd_B.spv)"
else
    echo "  FAILED — ocloc options for reference:"
    ocloc compile --help 2>&1 | grep -i -E "spv|spirv|-output|-device|-out_dir" | head -10
fi

echo
if [ -f g1_vecadd_A.spv ]; then
    cp g1_vecadd_A.spv g1_vecadd.spv; echo "== USING route A -> g1_vecadd.spv"
elif [ -f g1_vecadd_B.spv ]; then
    cp g1_vecadd_B.spv g1_vecadd.spv; echo "== USING route B -> g1_vecadd.spv"
else
    echo "== NO SPIR-V PRODUCED"; exit 1
fi
