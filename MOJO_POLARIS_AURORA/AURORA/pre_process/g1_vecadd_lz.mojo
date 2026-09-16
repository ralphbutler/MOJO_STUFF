# g1_vecadd_lz.mojo — G1: Mojo drives an Aurora PVC GPU tile through Level Zero.
#
# Same experiment as POLARIS/02_vecadd_gpu.mojo (N, block size, input generators,
# warmup, kernel-only timing, exact-match check) so the numbers line up. The
# difference is the plumbing: Polaris used Mojo's built-in DeviceContext (CUDA);
# Aurora has no Mojo Intel backend, so this program calls Level Zero directly via
# the vendored mojo_intel_gpu package and launches a SPIR-V kernel (g1_vecadd.spv).
#
# Run on a next-eval compute node (see g1_run.pbs):
#   ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 .venv/bin/mojo run -I . g1_vecadd_lz.mojo

from std.math import ceildiv
from std.os import getenv
from std.time import perf_counter_ns
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount

comptime N = 1_000_000
comptime BLOCK = 256                 # local group size (threads per group)
comptime ITERS = 100                 # repeat launches for a stable timing average
comptime SPV_PATH = "g1_vecadd.spv"
comptime KERNEL_NAME = "vecadd"


# Same asymmetric generators as POLARIS/02 — used to fill inputs AND to check the result.
def a_val(i: Int) -> Float32:
    return Float32(i) * 0.5

def b_val(i: Int) -> Float32:
    return Float32(i) * -1.5 + 2.0


def main() raises:
    print("=== G1: Mojo -> Level Zero -> Intel GPU (vector add) ===")
    print("  ZE_FLAT_DEVICE_HIERARCHY :", getenv("ZE_FLAT_DEVICE_HIERARCHY", "<unset>"))
    print("  ZE_AFFINITY_MASK         :", getenv("ZE_AFFINITY_MASK", "<unset>"))

    var ctx = IntelGPUContext()
    var info = ctx.compute_info()
    print("  device                   :", ctx.device_info().name)
    print("  ", ctx.device_info())
    print("  ", info)
    if UInt32(BLOCK) > info.max_group_size:
        raise Error("BLOCK exceeds device max group size")

    var bytes = UInt64(N * 4)        # float32

    # Host (pinned USM) buffers: fill inputs, receive the result.
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

    # Device buffers + host -> device copies.
    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)

    # Load the SPIR-V module and bind arguments (pointers, then the int32 size).
    var t_load0 = perf_counter_ns()
    var kernel = Kernel(
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
        SPV_PATH, KERNEL_NAME,
    )
    var t_load1 = perf_counter_ns()
    kernel.set_arg_pointer(0, d_a)
    kernel.set_arg_pointer(1, d_b)
    kernel.set_arg_pointer(2, d_c)
    kernel.set_arg_int32(3, Int32(N))

    var groups = ceildiv(N, BLOCK)
    kernel.set_group_size(UInt32(BLOCK), 1, 1)

    # Warmup: the first launch pays one-time JIT (SPIR-V -> PVC code) costs — exclude it.
    kernel.launch(ZeGroupCount(UInt32(groups), 1, 1))
    ctx.synchronize()

    # Timed: kernel execution only (not the host<->device copies), same as POLARIS/02.
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        kernel.launch(ZeGroupCount(UInt32(groups), 1, 1))
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var gpu_ms = Float64(t1 - t0) / Float64(ITERS) / 1.0e6

    # Device -> host, then verify against the CPU formula (same float32 add → exact match).
    ctx.memcpy_dtoh(h_c, d_c, bytes)
    var mismatches = 0
    for j in range(N):
        if c_ptr.unsafe_offset(j)[] != a_val(j) + b_val(j):
            if mismatches < 5:
                print("  MISMATCH [", j, "]: expected", a_val(j) + b_val(j),
                      "got", c_ptr.unsafe_offset(j)[])
            mismatches += 1

    print("g1_vecadd_lz — vector add on an Intel Max 1550 tile (Level Zero)")
    print("  N          :", N)
    print("  block size :", BLOCK, "threads")
    print("  grid size  :", groups, "groups")
    print("  total thr  :", groups * BLOCK, "(overhang guarded)")
    print("  mismatches :", mismatches, "/", N)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  module load:", Float64(t_load1 - t_load0) / 1.0e6, "ms (zeModuleCreate + zeKernelCreate)")
    print("  GPU time   :", gpu_ms, "ms/pass (kernel only)")
    print("  c[1], c[999999] :", c_ptr.unsafe_offset(1)[], c_ptr.unsafe_offset(N - 1)[])

    # Destroy the kernel before the context it borrows from.
    _ = kernel^
    ctx.free_device(d_a)
    ctx.free_device(d_b)
    ctx.free_device(d_c)
    ctx.free_host(h_a)
    ctx.free_host(h_b)
    ctx.free_host(h_c)
    ctx.close()
