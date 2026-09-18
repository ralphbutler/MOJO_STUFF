#!/bin/bash
# syntax_check_runtime.sh — Mac: syntax-check mojo_level_zero_rt.cpp without Aurora.
#
# The real build (build_runtime.sh) needs Level Zero headers that only exist on
# Aurora, so a typo used to cost a full round trip: sync, ssh, build, read, fix.
# ze_stub/ declares just the handles, structs and entry points the runtime uses
# -- enough for the compiler to check every line. It is never linked or shipped.
#
#   bash syntax_check_runtime.sh

set -euo pipefail
cd "$(dirname "$0")"
CXX=${CXX:-c++}
$CXX -std=c++17 -fsyntax-only -Wall -Wextra -Wno-unused-parameter \
    -I ze_stub runtime/mojo_level_zero_rt.cpp
echo "syntax OK (stub headers; the real build is build_runtime.sh on uan-0007)"
