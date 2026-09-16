# bringup_skeleton.mojo — G5 step 1: compile a plain function for our spirv64 target and write
# the SPIR-V binary our backend emits (LLVM's in-tree SPIR-V codegen, no llvm-spirv).
#
# Needs the SOURCE-BUILT compiler (MOJO_POLARIS_AURORA/modular), not the prebuilt 1.0.0:
#   cd modular && ./bazelw run //Mojo:mojo -- run <abs path>/bringup_skeleton.mojo <abs out.spv>
# (`bazel run` changes the working directory, so pass an absolute output path.)
# Check on the Mac:   spirv-val --target-env spv1.4 <out.spv>
# Check on Aurora:    bash ocloc_check.sh <out.spv>      (uan-0007, IGC for PVC)

from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget

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


def add_one(p: Pointer[Float32, MutAnyOrigin], n: Int32):
    for i in range(Int(n)):
        p[unsafe_offset=i] = p[unsafe_offset=i] + 1.0


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bringup_skeleton.mojo <absolute output .spv path>")
    var obj = compile_info[add_one, emission_kind="object", target=spirv_target]()
    var f = open(String(args[1]), "w")
    f.write_bytes(obj.asm.as_bytes())
    f.close()
    print("wrote", args[1], "(", obj.asm.byte_length(), "bytes )")
