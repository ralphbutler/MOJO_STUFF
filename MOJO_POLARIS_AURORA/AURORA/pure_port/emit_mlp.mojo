# emit_mlp.mojo — G5 step 3: compile the 12 distinct kernels of the UNCHANGED
# MOJO_CURRICULUM/04b_train_mlp_gpu.mojo with our spirv64 backend, for run_train_mlp.mojo.
#
# 04b names its kernels with comptime locals inside main() (K_fwd1 = matmul_kernel[...], ...),
# which cannot be imported, so this file instantiates the same generic kernels with 04b's own
# module-level layouts, mirroring 04b's main() line for line. The curriculum file is unchanged.
# As in G4, the four SGD launches share one kernel (1D row-major stride 1 in every case).
#
# Mac, from MOJO_POLARIS_AURORA/modular:
#   ln -sf <MOJO_CURRICULUM>/04b_train_mlp_gpu.mojo <dir>/curriculum_04b.mojo
#   ./bazelw run //Mojo:mojo -- run -I $PWD/max/mojo -I $PWD/max/kernels/src -I <dir> \
#       <abs>/emit_mlp.mojo <abs out dir>
# Writes g5_<role>.spv, .spv.name (entry point) and .spv.args (manifest in the g4 format:
# `buffer` = TileTensor holder, `const` = scalar cell; the backend passes both by pointer).

from std.compile import compile_info
from std.sys import argv
from std.sys.info import CompilationTarget
from curriculum_04b import (
    L_1H,
    L_11,
    L_DH,
    L_DHf,
    L_H1,
    L_N1,
    L_ND,
    L_NH,
    add_bias1_kernel,
    bias_relu_kernel,
    colsum_kernel,
    dz2_kernel,
    matmul_a_bt_kernel,
    matmul_at_b_kernel,
    matmul_kernel,
    relu_grad_kernel,
    sgd_kernel,
)

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

# Same instantiations as 04b's main().
comptime K_fwd1 = matmul_kernel[type_of(L_ND), type_of(L_DH), type_of(L_NH)]
comptime K_fwd2 = matmul_kernel[type_of(L_NH), type_of(L_H1), type_of(L_N1)]
comptime K_biasrelu = bias_relu_kernel[
    type_of(L_NH), type_of(L_1H), type_of(L_NH)
]
comptime K_addb2 = add_bias1_kernel[type_of(L_N1), type_of(L_11)]
comptime K_dz2 = dz2_kernel[type_of(L_N1), type_of(L_N1), type_of(L_N1)]
comptime K_dW2 = matmul_at_b_kernel[type_of(L_NH), type_of(L_N1), type_of(L_H1)]
comptime K_db2 = colsum_kernel[type_of(L_N1), type_of(L_11)]
comptime K_da1 = matmul_a_bt_kernel[type_of(L_N1), type_of(L_H1), type_of(L_NH)]
comptime K_relugrad = relu_grad_kernel[
    type_of(L_NH), type_of(L_NH), type_of(L_NH)
]
comptime K_dW1 = matmul_at_b_kernel[type_of(L_ND), type_of(L_NH), type_of(L_DH)]
comptime K_db1 = colsum_kernel[type_of(L_NH), type_of(L_1H)]
comptime K_sgd = sgd_kernel[type_of(L_DHf)]


def _manifest(n_buffers: Int, n_consts: Int) -> String:
    var s = String()
    for i in range(n_buffers):
        s += "arg " + String(i) + " buffer\n"
    for i in range(n_consts):
        s += "arg " + String(n_buffers + i) + " const\n"
    return s^


def _write(
    dir: String,
    role: String,
    content: StaticString,
    name: StaticString,
    n_buffers: Int,
    n_consts: Int,
) raises:
    var path = dir + "/g5_" + role
    var f = open(path + ".spv", "w")
    f.write_bytes(content.as_bytes())
    f.close()
    var n = open(path + ".spv.name", "w")
    n.write(name)
    n.close()
    var a = open(path + ".spv.args", "w")
    a.write(_manifest(n_buffers, n_consts))
    a.close()
    print("wrote", path + ".spv", "(", content.byte_length(), "bytes,", n_buffers, "buffers,", n_consts, "consts )")


def _emit[
    func_type: TrivialRegisterPassable, //, func: func_type
](dir: String, role: String, n_buffers: Int, n_consts: Int) raises:
    var k = compile_info[func, emission_kind="object", target=spirv_target]()
    _write(dir, role, k.asm, k.function_name, n_buffers, n_consts)


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: emit_mlp.mojo <absolute output dir>")
    var dir = String(args[1])
    _emit[K_fwd1](dir, "fwd1", 3, 3)
    _emit[K_fwd2](dir, "fwd2", 3, 3)
    _emit[K_biasrelu](dir, "biasrelu", 3, 2)
    _emit[K_addb2](dir, "addb2", 2, 1)
    _emit[K_dz2](dir, "dz2", 3, 2)
    _emit[K_dW2](dir, "dW2", 3, 3)
    _emit[K_db2](dir, "db2", 2, 2)
    _emit[K_da1](dir, "da1", 3, 3)
    _emit[K_relugrad](dir, "relugrad", 3, 2)
    _emit[K_dW1](dir, "dW1", 3, 3)
    _emit[K_db1](dir, "db1", 2, 2)
    _emit[K_sgd](dir, "sgd", 2, 2)
