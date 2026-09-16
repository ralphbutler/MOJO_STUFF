# bringup_thread_ids.mojo — G5 step 2: vecadd-shaped kernels using Mojo's GPU thread indexing, compiled
# for our spirv64 target. Two lowerings are emitted for comparison on Aurora (ocloc / IGC):
#   A  (stdlib, as patched)  global_idx.x -> llvm.spv.* intrinsics -> vec3 BuiltIn input variables
#   B  (OpenCL calls)        get_group_id / get_local_size / get_local_id, as air2spir produced
#
# Needs the SOURCE-BUILT compiler (MOJO_POLARIS_AURORA/modular):
#   cd modular && ./bazelw run //Mojo:mojo -- run <abs>/bringup_thread_ids.mojo <abs out dir>
# Writes <out>/g5_ids_a.spv and <out>/g5_ids_b.spv.

from std.compile import compile_info
from std.ffi import external_call
from std.sys import argv
from std.sys.info import CompilationTarget
from std.gpu import global_idx  # the Mojo 1.0 spelling (curriculum), via the fork's compat re-export

comptime spirv_target = CompilationTarget[
    _mlir_value=__mlir_attr[
        `#kgen.target<triple = "spirv64-unknown-unknown", `,
        `arch = "intel-pvc", `,
        `features = "", `,
        `data_layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1", `,
        `index_bit_width = 64, `,
        `simd_bit_width = 128`,
        `> : !kgen.target`,
    ]
]()


# A: the stdlib's global_idx (block_idx * block_dim + thread_idx).
def vecadd_a(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    c: Pointer[Float32, MutAnyOrigin],
    size: Int32,
):
    var tid = global_idx.x
    if tid < Int(size):
        c[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


# B: the same index through OpenCL builtin calls (Itanium-mangled, size_t result).
def vecadd_b(
    a: Pointer[Float32, MutAnyOrigin],
    b: Pointer[Float32, MutAnyOrigin],
    c: Pointer[Float32, MutAnyOrigin],
    size: Int32,
):
    var g = external_call["_Z12get_group_idj", UInt64](UInt32(0))
    var d = external_call["_Z14get_local_sizej", UInt64](UInt32(0))
    var l = external_call["_Z12get_local_idj", UInt64](UInt32(0))
    var tid = Int(g * d + l)
    if tid < Int(size):
        c[unsafe_offset=tid] = a[unsafe_offset=tid] + b[unsafe_offset=tid]


def _write(path: String, bytes: Span[Byte, _]) raises:
    var f = open(path, "w")
    f.write_bytes(bytes)
    f.close()
    print("wrote", path, "(", len(bytes), "bytes )")


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bringup_thread_ids.mojo <absolute output dir>")
    var out = String(args[1])
    var ll = compile_info[vecadd_a, emission_kind="llvm-opt", target=spirv_target]()
    print(ll)
    var a = compile_info[vecadd_a, emission_kind="object", target=spirv_target]()
    _write(out + "/g5_ids_a.spv", a.asm.as_bytes())
    var b = compile_info[vecadd_b, emission_kind="object", target=spirv_target]()
    _write(out + "/g5_ids_b.spv", b.asm.as_bytes())
