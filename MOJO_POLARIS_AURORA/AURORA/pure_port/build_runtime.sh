#!/bin/bash
# build_runtime.sh — aurora-uan-0007: build our Level Zero implementation of the AsyncRT_*
# entry points that Mojo's `max.gpu.host.DeviceContext` calls (G5 step 4d).
#
# Usage (from MOJO_WORK/AURORA/gpu):  bash build_runtime.sh
# Output: MOJO_WORK/runtime/lib/libmojo_level_zero_rt.so
#
# Programs pick it up through the JIT's linker passthrough:
#   mojo run --target-accelerator intel-pvc \
#       -Xlinker -L<...>/runtime/lib -Xlinker -lmojo_level_zero_rt prog.mojo

set -euo pipefail
W=/lus/flare/projects/ModCon/rbutler/MOJO_WORK
SRC="$(cd "$(dirname "$0")" && pwd)/runtime/mojo_level_zero_rt.cpp"
OUT=$W/runtime/lib
mkdir -p "$OUT"

echo "== host $(hostname), $(date)"
CXX=${CXX:-g++}
echo "== compiler: $($CXX --version | head -1)"
ls /usr/include/level_zero/ze_api.h
ldconfig -p | grep -m1 ze_loader || echo "   (libze_loader not in ldconfig cache — linking by name)"

set -x
$CXX -std=c++17 -O2 -fPIC -shared -Wall -Wextra -Wno-unused-parameter \
    "$SRC" -o "$OUT/libmojo_level_zero_rt.so" -lze_loader
set +x

ls -l "$OUT/libmojo_level_zero_rt.so"
echo "== exported AsyncRT symbols:"
nm -D --defined-only "$OUT/libmojo_level_zero_rt.so" | grep AsyncRT | awk '{print "    " $3}'
echo "== undefined (should be libc/libstdc++/libze_loader only):"
nm -D --undefined-only "$OUT/libmojo_level_zero_rt.so" | grep -c .
echo "== done"
