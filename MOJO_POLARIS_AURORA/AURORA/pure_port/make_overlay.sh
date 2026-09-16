#!/bin/bash
# make_overlay.sh — Mac: pack our spirv64-backend patches to the open-source Mojo repo into
# pure_port/fork_overlay.tar.gz, for prepare_fork.sh to apply on Aurora on top of the same commit.
#
# The file list is EXPLICIT, not taken from git: the repo's .gitignore has `target/`, which macOS's
# case-insensitive git also matches against Mojo/lib/**/Target/IntelGPU/, so `git status` and
# `git diff` do not show the backend's C++ sources.
#
# Usage:  bash pure_port/make_overlay.sh        (from anywhere)

set -euo pipefail
REPO=/Users/rbutler/Desktop/LLMs/MOJO_STUFF/MOJO_POLARIS_AURORA/modular
OUT="$(cd "$(dirname "$0")" && pwd)/fork_overlay.tar.gz"
BASE_COMMIT=6417db28ceef430d067755e9826cec65e19f9333

cd "$REPO"
[ "$(git rev-parse HEAD)" = "$BASE_COMMIT" ] || { echo "error: clone is not at $BASE_COMMIT" >&2; exit 1; }

FILES=(
    # build: LLVM SPIR-V target (and the macOS-only SDK pin, inert on Linux)
    MODULE.bazel
    bazel/internal/cc-toolchain/macos_sysroot_repository.bzl
    # compiler: the IntelGPU target (+ the --target-accelerator MAX-gate exemption for base targets)
    Mojo/lib/Target/TargetTraits.h
    Mojo/lib/Target/TargetTraits.cpp
    Mojo/lib/Target/IntelGPU/IntelGPUTraits.h
    Mojo/lib/Target/IntelGPU/IntelGPUTraits.cpp
    Mojo/lib/KGENToLLVM/Target/IntelGPU/IntelGPULowering.cpp
    Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUBackend.cpp
    Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUAddressSpaces.h
    Mojo/lib/Compiler/ObjectCompiler/Target/IntelGPU/IntelGPUAddressSpaces.cpp
    # stdlib
    Mojo/stdlib/std/sys/info.mojo
    Mojo/stdlib/std/sys/__init__.mojo
    Mojo/stdlib/std/_gpu/primitives/id.mojo
    Mojo/stdlib/std/_gpu/host/_builtin_targets.mojo
    Mojo/stdlib/std/gpu/__init__.mojo
    Mojo/stdlib/std/gpu/host/__init__.mojo
    Mojo/stdlib/std/builtin/debug_assert.mojo
    Mojo/stdlib/std/os/os.mojo
    Mojo/stdlib/std/memory/_poison.mojo
    Mojo/stdlib/std/collections/string/string.mojo
    # max
    max/mojo/max/gpu/sync/sync.mojo
)

# Warn about modified tracked files missing from the list (untracked/ignored ones can't be detected).
for f in $(git diff --name-only); do
    printf '%s\n' "${FILES[@]}" | grep -qx "$f" || echo "warning: modified but not in overlay: $f" >&2
done

COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$OUT" "${FILES[@]}"
echo "== wrote $OUT ($(stat -f %z "$OUT") bytes, ${#FILES[@]} files, base $BASE_COMMIT)"
shasum -a 256 "$OUT"
