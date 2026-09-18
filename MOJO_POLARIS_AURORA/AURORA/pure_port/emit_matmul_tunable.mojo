# emit_matmul_tunable.mojo — compile one geometry of matmul_tunable.mojo to SPIR-V with our
# spirv64 backend, on the Mac, without Aurora. Used by matmul_sweep_check.sh to validate every
# sweep configuration (and measure its SPIR-V size) before any queue time is spent.
#
#   bash set_matmul_geometry.sh BM=64 BN=64 BK=8 TM=4 TN=4
#   ./bazelw run //Mojo:mojo -- run -I $PWD/max/mojo -I $PWD/max/kernels/src -I <gpu dir> \
#       <abs>/emit_matmul_tunable.mojo <abs out dir>/<tag>

from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget
from matmul_tunable import matmul_pvc

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

comptime ARGS_MANIFEST = "arg 0 buffer\narg 1 buffer\narg 2 buffer\n"


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: emit_matmul_tunable.mojo <absolute output path prefix>")
    var path = String(args[1])
    var k = compile_info[
        matmul_pvc, emission_kind="object", target=spirv_target
    ]()
    var f = open(path + ".spv", "w")
    f.write_bytes(k.asm.as_bytes())
    f.close()
    var n = open(path + ".spv.name", "w")
    n.write(k.function_name)
    n.close()
    var a = open(path + ".spv.args", "w")
    a.write(ARGS_MANIFEST)
    a.close()
    print("wrote", path + ".spv", "(", k.asm.byte_length(), "bytes )")
