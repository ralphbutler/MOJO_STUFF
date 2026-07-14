# 00_simd_type.mojo — the SIMD *type*, on its own (no buffers, no arrays)
#
# SIMD[dtype, W] is a fixed-width vector of W lanes that lives in a register.
# W is a compile-time power of 2 (1, 2, 4, 8, ...) — small, not a dynamic list.
# This file just pokes at the type so it feels concrete before we use it at scale.

from std.math import min

comptime dtype = DType.float32


def main() raises:
    # --- construct & index (this is the "sorta like a small list" part) ---
    var v = SIMD[dtype, 4](1.0, 2.0, 3.0, 4.0)   # 4 lanes
    print("v            :", v)
    print("v[2]         :", v[2])                 # read one lane -> Scalar[dtype]
    v[2] = 30.0                                   # write one lane
    print("after v[2]=30:", v)

    # Broadcast: one value fills every lane.
    var seven = SIMD[dtype, 4](7.0)
    print("splat 7      :", seven)

    # --- lane-wise arithmetic: one op hits all 4 lanes at once ---
    var a = SIMD[dtype, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[dtype, 4](10.0, 20.0, 30.0, 40.0)
    print("a + b        :", a + b)                # <11, 22, 33, 44>
    print("a * 2        :", a * 2.0)              # scalar broadcasts across lanes
    print("min(a, seven):", min(a, seven))        # element-wise; min is a free function

    # --- horizontal reductions: collapse the vector to one scalar ---
    print("a.reduce_add :", a.reduce_add())       # 1+2+3+4 = 10
    print("a.reduce_max :", a.reduce_max())       # 4

    # --- masks & select: per-lane branching without an if ---
    var x = SIMD[dtype, 4](-1.0, 2.0, -3.0, 4.0)
    var zeros = SIMD[dtype, 4](0.0)
    var mask = x.gt(zeros)                        # per-lane mask: <F, T, F, T>
    print("relu(x)      :", mask.select(x, zeros)) # keep x where >0, else 0

    # --- cast between element types (same lane count) ---
    print("a as int32   :", a.cast[DType.int32]()) # <1, 2, 3, 4> as integers

    # --- the tie-in to 01: a single element is just a 1-lane SIMD ---
    var one: Scalar[dtype] = 42.0                 # Scalar[dtype] == SIMD[dtype, 1]
    print("Scalar is SIMD[_,1]:", one)
