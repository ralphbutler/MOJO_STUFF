# emit_vecadd.mojo — G5 step 3: compile an UNCHANGED curriculum GPU kernel with our spirv64
# backend (source-built Mojo), and write the SPIR-V binary plus its entry-point name.
#
# No Metal backend, no air2spir, no llvm-spirv: Mojo -> LLVM's SPIR-V codegen -> .spv.
#
# Mac, from MOJO_POLARIS_AURORA/modular (the curriculum file name starts with a digit, so it is
# imported through a symlink named curriculum_02.mojo in <dir>):
#   ln -sf <MOJO_CURRICULUM>/02_vecadd_gpu.mojo <dir>/curriculum_02.mojo
#   ./bazelw run //Mojo:mojo -- run -I $PWD/max/mojo -I $PWD/max/kernels/src -I <dir> \
#       <abs>/emit_vecadd.mojo <abs out prefix>
# Writes <prefix>.spv and <prefix>.spv.name.
#
# Kernel ABI (IntelGPULowering): every argument is `ptr addrspace(1) byref(T)` — the same
# holder layout ../pre_process/g2_vecadd_lz.mojo already binds (8-byte TileTensor holders, 4-byte Int32 cell).

from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget
from curriculum_02 import vecadd_kernel

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


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: emit_vecadd.mojo <absolute output prefix>")
    var prefix = String(args[1])
    var obj = compile_info[
        vecadd_kernel, emission_kind="object", target=spirv_target
    ]()
    var f = open(prefix + ".spv", "w")
    f.write_bytes(obj.asm.as_bytes())
    f.close()
    var n = open(prefix + ".spv.name", "w")
    n.write(obj.function_name)
    n.close()
    print("wrote", prefix + ".spv", "(", obj.asm.byte_length(), "bytes )")
    print("entry point:", obj.function_name)
