# L0 smoke test — does Mojo install, compile, and run on this node, and what CPU
# capabilities does it see? Run on an Aurora compute node (via qsub). Expect
# has_accelerator() == False on Aurora (no CUDA/HIP/Metal backend for Intel GPUs).

from std.sys import has_accelerator
from std.sys.info import (
    simd_width_of,
    num_physical_cores,
    num_logical_cores,
)


def main():
    print("=== Mojo L0 smoke test ===")
    print("has_accelerator   :", has_accelerator(), "(expect False on Aurora)")
    print("simd width  f32   :", simd_width_of[DType.float32](), "(16 => AVX-512)")
    print("simd width  f64   :", simd_width_of[DType.float64](), "(8  => AVX-512)")
    print("physical cores    :", num_physical_cores())
    print("logical  cores    :", num_logical_cores())

    # Tiny compute to prove codegen + execution actually work on this node.
    var total: Int = 0
    for i in range(1_000_000):
        total += i
    print("sum 0..999999     :", total, "(expect 499999500000)")
    print("=== OK: Mojo runs on this node ===")
