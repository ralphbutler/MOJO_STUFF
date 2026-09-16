# run_vecadd.mojo — G5 step 3: the UNCHANGED curriculum vecadd kernel, compiled by OUR spirv64
# Mojo backend (source-built compiler, LLVM SPIR-V codegen), runs on an Aurora PVC tile.
#
# Kernel: `vecadd_kernel` from MOJO_CURRICULUM/02_vecadd_gpu.mojo, unchanged.
# Build (Mac, fork): emit_vecadd.mojo -> g5_vecadd.spv + g5_vecadd.spv.name (entry point).
# No Metal backend, no air2spir.py, no llvm-spirv, no clang.
#
# Host = ../pre_process/g2_vecadd_lz.mojo (frozen groundwork, used here only as a test harness). The backend's
# kernel ABI passes every argument as `ptr addrspace(1) byref(T)`, which is exactly the holder
# layout G2 bound: 8-byte USM holders with the TileTensor data pointer, a 4-byte Int32 cell.
# Differences from g2: the .spv path, and the entry-point name read from g5_vecadd.spv.name.
#
# Run:  qsub -v MOJOFILE=run_vecadd.mojo run_job.pbs

from std.math import ceildiv
from std.os import getenv
from std.time import perf_counter_ns
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount

comptime N = 1_000_000
comptime BLOCK = 256
comptime ITERS = 100
comptime SPV_PATH = "g5_vecadd.spv"
comptime NAME_PATH = "g5_vecadd.spv.name"


def a_val(i: Int) -> Float32:
    return Float32(i) * 0.5

def b_val(i: Int) -> Float32:
    return Float32(i) * -1.5 + 2.0


def make_ptr_holder(mut ctx: IntelGPUContext, device_ptr: Int) raises -> Int:
    """8-byte USM host buffer containing `device_ptr` (a TileTensor struct for the kernel)."""
    var h = ctx.allocate_host(8)
    Pointer[Int, MutUntrackedOrigin](unsafe_from_address=h)[] = device_ptr
    return h


def main() raises:
    print("=== G5: Mojo spirv64 backend kernel -> Level Zero -> Intel GPU ===")
    print("  ZE_FLAT_DEVICE_HIERARCHY :", getenv("ZE_FLAT_DEVICE_HIERARCHY", "<unset>"))
    print("  ZE_AFFINITY_MASK         :", getenv("ZE_AFFINITY_MASK", "<unset>"))

    var ctx = IntelGPUContext()
    print("  device                   :", ctx.device_info().name)

    var bytes = UInt64(N * 4)
    var h_a = ctx.allocate_host(bytes)
    var h_b = ctx.allocate_host(bytes)
    var h_c = ctx.allocate_host(bytes)
    var a_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_a)
    var b_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_b)
    var c_ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_c)
    for i in range(N):
        a_ptr.unsafe_offset(i)[] = a_val(i)
        b_ptr.unsafe_offset(i)[] = b_val(i)
        c_ptr.unsafe_offset(i)[] = 0.0

    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)

    # Argument holders (see header comment).
    var hold_a = make_ptr_holder(ctx, d_a)
    var hold_b = make_ptr_holder(ctx, d_b)
    var hold_c = make_ptr_holder(ctx, d_c)
    var hold_n = ctx.allocate_host(4)
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=hold_n)[] = Int32(N)

    var kernel_name = open(NAME_PATH, "r").read()
    print("  entry point              :", kernel_name.byte_length(), "chars")

    var t_load0 = perf_counter_ns()
    var kernel = Kernel(
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
        SPV_PATH, kernel_name,
    )
    var t_load1 = perf_counter_ns()
    kernel.set_arg_pointer(0, hold_a)
    kernel.set_arg_pointer(1, hold_b)
    kernel.set_arg_pointer(2, hold_c)
    kernel.set_arg_pointer(3, hold_n)
    # d_a/d_b/d_c are reached only through the holders, so Level Zero must be told
    # to make them resident (run 1 faulted "NotPresent" writing d_c without this).
    kernel.set_indirect_access(7)   # HOST | DEVICE | SHARED

    var groups = ceildiv(N, BLOCK)
    kernel.set_group_size(UInt32(BLOCK), 1, 1)

    kernel.launch(ZeGroupCount(UInt32(groups), 1, 1))   # warmup (JIT)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        kernel.launch(ZeGroupCount(UInt32(groups), 1, 1))
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var gpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    ctx.memcpy_dtoh(h_c, d_c, bytes)
    var mismatches = 0
    for j in range(N):
        if c_ptr.unsafe_offset(j)[] != a_val(j) + b_val(j):
            if mismatches < 5:
                print("  MISMATCH [", j, "]: expected", a_val(j) + b_val(j),
                      "got", c_ptr.unsafe_offset(j)[])
            mismatches += 1

    print("g5_vecadd_lz — spirv64-backend vector add kernel on an Intel Max 1550 tile")
    print("  kernel src :", "MOJO_CURRICULUM/02_vecadd_gpu.mojo (vecadd_kernel, unchanged)")
    print("  compiler   :", "source-built Mojo, spirv64 target (LLVM SPIR-V codegen)")
    print("  N          :", N)
    print("  block size :", BLOCK, "threads")
    print("  grid size  :", groups, "groups")
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  module load:", Float64(t_load1 - t_load0) / 1.0e6, "ms (zeModuleCreate + zeKernelCreate)")
    print("  GPU time   :", gpu_ms, "ms/pass (kernel only)")
    print("  c[1], c[999999] :", c_ptr.unsafe_offset(1)[], c_ptr.unsafe_offset(N - 1)[])

    _ = kernel^
    for h in [hold_a, hold_b, hold_c, hold_n, h_a, h_b, h_c]:
        ctx.free_host(h)
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.close()
