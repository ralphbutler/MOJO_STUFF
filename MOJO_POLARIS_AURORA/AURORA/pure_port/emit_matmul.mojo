# emit_matmul.mojo — G5 step 3: compile the UNCHANGED curriculum coarse-matmul kernels
# (shared memory + barrier) with our spirv64 backend, for run_matmul.mojo.
#
#   check  : 03d_matmul_check.mojo   N=256  — FULL CPU-reference check
#   coarse : 03c_matmul_coarse.mojo  N=2048 — the timed kernel
#
# Mac, from MOJO_POLARIS_AURORA/modular (curriculum file names start with a digit, so they are
# imported through symlinks in <dir>):
#   ln -sf <MOJO_CURRICULUM>/03c_matmul_coarse.mojo <dir>/curriculum_03c.mojo
#   ln -sf <MOJO_CURRICULUM>/03d_matmul_check.mojo  <dir>/curriculum_03d.mojo
#   ./bazelw run //Mojo:mojo -- run -I $PWD/max/mojo -I $PWD/max/kernels/src -I <dir> \
#       <abs>/emit_matmul.mojo <abs out dir>
# Writes g5_{check,coarse}.spv, .spv.name (entry point) and .spv.args (argument manifest in
# the g3_matmul_lz format: every argument is a buffer holder under the backend's ABI; shared
# memory is a module-scope Workgroup variable, so there are no `local` arguments).

from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget
from curriculum_03c import matmul_coarse as coarse_kernel
from curriculum_03d import matmul_coarse as check_kernel

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


def _write(path: String, content: StaticString, name: StaticString) raises:
    var f = open(path + ".spv", "w")
    f.write_bytes(content.as_bytes())
    f.close()
    var n = open(path + ".spv.name", "w")
    n.write(name)
    n.close()
    var a = open(path + ".spv.args", "w")
    a.write(ARGS_MANIFEST)
    a.close()
    print("wrote", path + ".spv", "(", content.byte_length(), "bytes )")


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: emit_matmul.mojo <absolute output dir>")
    var out = String(args[1])
    var check = compile_info[
        check_kernel, emission_kind="object", target=spirv_target
    ]()
    _write(out + "/g5_check", check.asm, check.function_name)
    var coarse = compile_info[
        coarse_kernel, emission_kind="object", target=spirv_target
    ]()
    _write(out + "/g5_coarse", coarse.asm, coarse.function_name)
