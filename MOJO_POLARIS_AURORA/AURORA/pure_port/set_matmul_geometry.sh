#!/bin/bash
# set_matmul_geometry.sh — rewrite the SWEEP GEOMETRY block of matmul_tunable.mojo in place.
#
#   bash set_matmul_geometry.sh BM=64 BN=64 BK=8 TM=4 TN=4 [N=2048] [ITERS=50]
#
# The geometry lives as literal `comptime` constants rather than `-D` defines because a
# define stays an unfolded expression in the mangled kernel name and doubles its length
# (4,891 -> 9,747 characters); the emitted SPIR-V is identical either way, but names that
# long are untested against IGC and Level Zero. Rewriting literals keeps the baseline
# geometry byte-identical to the curriculum's 03c kernel, which is the sweep's control.

set -euo pipefail
cd "$(dirname "$0")"
SRC=matmul_tunable.mojo
[ -f "$SRC" ] || { echo "error: $SRC not found"; exit 1; }

for kv in "$@"; do
    key=${kv%%=*}
    val=${kv#*=}
    case "$key" in
        N|BM|BN|BK|TM|TN|ITERS) ;;
        *) echo "error: unknown geometry key '$key'"; exit 1 ;;
    esac
    case "$val" in
        ''|*[!0-9]*) echo "error: $key='$val' is not a positive integer"; exit 1 ;;
    esac
    # Only inside the marked block, and only the value column; comments are preserved.
    perl -i -pe "
        \$inblk = 1 if /--- SWEEP GEOMETRY/;
        \$inblk = 0 if /--- END SWEEP GEOMETRY/;
        s/^(comptime $key = )\d+/\${1}$val/ if \$inblk;
    " "$SRC"
done

# Read back and verify. A silently failed rewrite would run every sweep cell at the same
# geometry and produce a confident, wrong table -- worse than an error.
for kv in "$@"; do
    key=${kv%%=*}
    val=${kv#*=}
    got=$(sed -n "/--- SWEEP GEOMETRY/,/--- END SWEEP GEOMETRY/p" "$SRC" \
          | sed -n "s/^comptime $key = \([0-9]*\).*/\1/p")
    if [ "$got" != "$val" ]; then
        echo "ERROR: failed to set $key=$val in $SRC (reads back as '${got:-<missing>}')."
        echo "       Is perl available? The sweep must not run with a stale geometry."
        exit 1
    fi
done

echo "== geometry now:"
sed -n '/--- SWEEP GEOMETRY/,/--- END SWEEP GEOMETRY/p' "$SRC" | grep -E '^comptime' | sed 's/^/    /'
