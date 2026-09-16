#!/bin/bash
# g5_check_air_on_aurora.sh — G5-lite question 1: can Aurora emit the Metal AIR itself?
#
# Today the kernel IR is generated on a Mac (`mojo build --emit asm --target-accelerator apple-m4`)
# and copied over. That Mac step is the weakest part of the claim: it makes the pipeline
# two-machine and ties it to a macOS toolchain. Mojo's Linux build DOES list Apple Silicon under
# --print-supported-accelerators (seen in G0.2), so the sidecar may well be emittable here — the
# backend is a code generator, not a runtime, and nothing needs an Apple GPU to be present.
#
# This script answers that with evidence, and goes one step further: if Aurora emits a sidecar, it
# compares it with the Mac-generated g2_vecadd.air.ll that G2 actually ran. Byte-identical IR
# (modulo the mangled symbol name) means the Mac step can be deleted outright rather than merely
# duplicated.
#
# Needs the curriculum sources here (they are not in the repo mirror):
#   scp <MOJO_CURRICULUM>/02_vecadd_gpu.mojo <MOJO_CURRICULUM>/04b_train_mlp_gpu.mojo \
#       rbutler@aurora.alcf.anl.gov:/lus/flare/projects/ModCon/rbutler/MOJO_WORK/AURORA/gpu/curriculum/
#
# Usage (on aurora-uan-0007, from MOJO_WORK/AURORA/gpu):
#   bash g5_check_air_on_aurora.sh 2>&1 | tee g5_air_check.txt

set -u
cd "$(dirname "$0")"
HERE=$(pwd)
MOJO=$HERE/.venv/bin/mojo
SRC=${SRC:-./curriculum}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

verdict_accel="unknown"; verdict_max="unknown"; verdict_emit="unknown"; verdict_same="unknown"

echo "== host"; hostname; grep VERSION= /etc/os-release

echo
echo "== 1. Mojo version"
$MOJO --version || { echo "FATAL: no mojo in ./.venv"; exit 1; }

echo
echo "== 2. supported accelerators (is Apple Silicon / Metal listed on Linux?)"
$MOJO build --print-supported-accelerators 2>&1 | sed 's/^/   /'
if $MOJO build --print-supported-accelerators 2>&1 | grep -qi "apple"; then
    verdict_accel="YES - Apple target listed"
else
    verdict_accel="NO - no Apple target on this build"
fi
echo "   -> $verdict_accel"

echo
echo "== 3. is the 'max' package present? (the curriculum imports max.gpu.host and layout)"
if ls -d "$HERE"/.venv/lib/python3*/site-packages/max >/dev/null 2>&1; then
    verdict_max="YES"
    ls -d "$HERE"/.venv/lib/python3*/site-packages/max*.dist-info 2>/dev/null | sed 's/^/   /'
else
    verdict_max="NO - run: uv add max"
fi
echo "   -> max present: $verdict_max"

echo
echo "== 4. emit AIR for 02_vecadd_gpu.mojo on THIS machine"
if [ ! -f "$SRC/02_vecadd_gpu.mojo" ]; then
    echo "   SKIP: $SRC/02_vecadd_gpu.mojo not found (see the header for the scp command)"
    verdict_emit="skipped - source missing"
else
    ( cd "$WORK" && "$MOJO" build --emit asm --target-accelerator apple-m4 \
        "$HERE/$SRC/02_vecadd_gpu.mojo" -o host.s ) 2>&1 | sed 's/^/   /'
    SIDE=$(ls "$WORK"/*vecadd_kernel*.ll 2>/dev/null | head -1)
    if [ -n "$SIDE" ]; then
        verdict_emit="YES"
        echo "   sidecar: $(basename "$SIDE") ($(stat -c%s "$SIDE") bytes)"
        echo "   triple : $(grep -m1 'target triple' "$SIDE")"
    elif [ "$verdict_max" != "YES" ]; then
        verdict_emit="BLOCKED - max not installed, question unanswered"
        echo "   The build failed on the missing 'max' package, NOT on the Apple target."
        echo "   Install max (see the verdict) and re-run before drawing any conclusion."
    else
        verdict_emit="NO - no .ll sidecar produced"
        echo "   files in work dir:"; ls -l "$WORK" | sed 's/^/     /'
    fi
fi

echo
echo "== 5. compare with the Mac-generated IR that G2 ran (g2_vecadd.air.ll)"
if [ "$verdict_emit" = "YES" ] && [ -f "$HERE/g2_vecadd.air.ll" ]; then
    # Ignore the mangled symbol name and source_filename; compare everything else.
    norm() { sed -E 's/@_02_vecadd_gpu[A-Za-z0-9_]+/@K/g; /^source_filename/d' "$1"; }
    if diff <(norm "$SIDE") <(norm "$HERE/g2_vecadd.air.ll") > "$WORK/diff.txt"; then
        verdict_same="IDENTICAL"
    else
        verdict_same="DIFFERS ($(grep -c '^[<>]' "$WORK/diff.txt") lines)"
        head -40 "$WORK/diff.txt" | sed 's/^/   /'
    fi
else
    verdict_same="not compared"
fi
echo "   -> $verdict_same"

echo
echo "== 6. harder case: 04b_train_mlp_gpu.mojo (expect 14 sidecars)"
if [ "$verdict_emit" = "YES" ] && [ -f "$SRC/04b_train_mlp_gpu.mojo" ]; then
    W2=$(mktemp -d)
    ( cd "$W2" && "$MOJO" build --emit asm --target-accelerator apple-m4 \
        "$HERE/$SRC/04b_train_mlp_gpu.mojo" -o host.s ) 2>&1 | tail -5 | sed 's/^/   /'
    echo "   sidecars emitted: $(ls "$W2"/*.ll 2>/dev/null | wc -l) (expected 14)"
    rm -rf "$W2"
else
    echo "   SKIP"
fi

echo
echo "================ VERDICT ================"
echo "  Apple accelerator target on Linux : $verdict_accel"
echo "  max package installed             : $verdict_max"
echo "  AIR sidecar emitted on Aurora     : $verdict_emit"
echo "  matches the Mac IR G2 ran         : $verdict_same"
echo
if [ "${verdict_emit:0:7}" = "BLOCKED" ]; then
    echo "  => INCONCLUSIVE. The Apple target IS available on this Linux build; the emit"
    echo "     failed only because 'max' is missing from gpu/.venv (the curriculum imports"
    echo "     max.gpu.host and layout). Install it and re-run:"
    echo
    echo "       cd $HERE && uv add \"max==26.5.0\"      # the version the Mac uses with Mojo 1.0.0"
    echo "       ./.venv/bin/mojo --version                # MUST still be 1.0.0 (ed45d567)"
    echo
    echo "     Pin the version: an unpinned add may move Mojo and break the working G1-G4 setup."
elif [ "$verdict_emit" = "YES" ] && [ "$verdict_same" = "IDENTICAL" ]; then
    echo "  => The Mac step can be REMOVED. G5-lite can build end to end on Aurora."
elif [ "$verdict_emit" = "YES" ]; then
    echo "  => Aurora can emit IR, but it is not identical to what G2 ran."
    echo "     Rebuild and re-run G2 from the Aurora-generated IR before trusting it."
else
    echo "  => The Mac step stays for now. G5-lite's 'one-command build' will be"
    echo "     one command per machine, and the claim keeps its two-machine caveat."
fi
