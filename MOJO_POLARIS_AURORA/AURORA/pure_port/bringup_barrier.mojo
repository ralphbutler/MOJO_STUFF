# bringup_barrier.mojo — G5 step 2: a shared-memory + barrier kernel (the 03b/03c pattern) compiled
# for our spirv64 target. Two barrier lowerings, for comparison on Aurora (ocloc / IGC):
#   A  llvm.spv.group.memory.barrier.with.group.sync   (LLVM intrinsic -> OpControlBarrier)
#   B  OpenCL barrier(CLK_LOCAL_MEM_FENCE)            (as air2spir produced; proven on PVC in G3)
# Both use stack_allocation[SHARED] -> a pop.global_alloc<gpu_shared> in addrspace(3).
#
# Needs the SOURCE-BUILT compiler (MOJO_POLARIS_AURORA/modular):
#   cd modular && ./bazelw run //Mojo:mojo -- run <abs>/bringup_barrier.mojo <abs out dir>
# Writes <out>/g5_barrier_a.spv and <out>/g5_barrier_b.spv.

from std.compile import compile_info
from std.ffi import external_call
from std.memory import stack_allocation
from std.sys import argv, llvm_intrinsic
from std.sys.info import CompilationTarget
from std._gpu import thread_idx, block_idx, block_dim

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

comptime BLOCK = 16


# Each thread copies its input into block-shared memory; after the barrier every thread
# sees the whole block and writes the block sum. Wrong without barrier synchronization.
def block_sum_a(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    size: Int32,
):
    var shared = stack_allocation[BLOCK, DType.float32, address_space=.SHARED]()
    var t = thread_idx.x
    var g = block_idx.x * block_dim.x + t
    shared[unsafe_offset=t] = x[unsafe_offset=g] if g < Int(size) else Float32(0)
    llvm_intrinsic["llvm.spv.group.memory.barrier.with.group.sync", NoneType]()
    var s = Float32(0)
    for i in range(BLOCK):
        s += shared[unsafe_offset=i]
    if g < Int(size):
        y[unsafe_offset=g] = s


def block_sum_b(
    x: Pointer[Float32, MutAnyOrigin],
    y: Pointer[Float32, MutAnyOrigin],
    size: Int32,
):
    var shared = stack_allocation[BLOCK, DType.float32, address_space=.SHARED]()
    var t = thread_idx.x
    var g = block_idx.x * block_dim.x + t
    shared[unsafe_offset=t] = x[unsafe_offset=g] if g < Int(size) else Float32(0)
    external_call["_Z7barrierj", NoneType](UInt32(1))  # CLK_LOCAL_MEM_FENCE
    var s = Float32(0)
    for i in range(BLOCK):
        s += shared[unsafe_offset=i]
    if g < Int(size):
        y[unsafe_offset=g] = s


def _write(path: String, bytes: Span[Byte, _]) raises:
    var f = open(path, "w")
    f.write_bytes(bytes)
    f.close()
    print("wrote", path, "(", len(bytes), "bytes )")


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bringup_barrier.mojo <absolute output dir>")
    var out = String(args[1])
    print(compile_info[block_sum_a, emission_kind="llvm-opt", target=spirv_target]())
    var a = compile_info[block_sum_a, emission_kind="object", target=spirv_target]()
    _write(out + "/g5_barrier_a.spv", a.asm.as_bytes())
    var b = compile_info[block_sum_b, emission_kind="object", target=spirv_target]()
    _write(out + "/g5_barrier_b.spv", b.asm.as_bytes())
