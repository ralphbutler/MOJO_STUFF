# run_matmul.mojo — G5 step 3: the UNCHANGED curriculum coarse-matmul kernels (shared memory +
# barrier), compiled by OUR spirv64 Mojo backend, on an Aurora PVC tile.
#
#   check  : 03d_matmul_check.mojo   N=256,  coarse kernel, FULL CPU-reference check
#   coarse : 03c_matmul_coarse.mojo  N=2048, 256-thread blocks, shared memory + barrier()
#
# Kernels: emit_matmul.mojo (source-built Mojo, spirv64 target) -> g5_{check,coarse}.spv, plus
# .spv.name (entry point) and .spv.args (manifest). No Metal backend / air2spir / llvm-spirv.
#
# Host = ../pre_process/g3_matmul_lz.mojo (frozen groundwork) used as a test harness. Differences: the variant
# table, and the kernel name read from <spv>.name. Under the backend's ABI all three arguments
# are 8-byte TileTensor holders; shared memory is a module-scope Workgroup variable that IGC
# allocates per work group, so there are no `local` arguments to bind.
#
# Run:  qsub -v MOJOFILE=run_matmul.mojo run_job.pbs     Subset: G5_VARIANTS="check" in the env.

from std.os import getenv
from std.pathlib import Path
from std.time import perf_counter_ns
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount


@fieldwise_init
struct Variant(Copyable, Movable):
    var name: String
    var src: String
    var spv: String
    var n: Int
    var gsize_x: Int     # threads per group (Metal block_dim)
    var gsize_y: Int
    var gcount_x: Int    # number of groups (Metal grid_dim)
    var gcount_y: Int
    var iters: Int
    var full_check: Bool


# Same deterministic, asymmetric generators as 03d_matmul_check.mojo.
def a_val(i: Int, j: Int) -> Float32:
    return Float32((i * 7 + j * 3) % 13) * 0.1 - 0.6

def b_val(i: Int, j: Int) -> Float32:
    return Float32((i * 5 + j * 11) % 17) * 0.1 - 0.8

def ref_elem(n: Int, i: Int, j: Int) -> Float64:
    var acc: Float32 = 0.0            # float32 accumulation, like the kernel
    for k in range(n):
        acc += a_val(i, k) * b_val(k, j)
    return Float64(acc)


def run_variant(mut ctx: IntelGPUContext, v: Variant) raises:
    print("\n=== G5 variant:", v.name, "(", v.src, ") ===")
    var n = v.n
    var bytes = UInt64(n * n * 4)

    # --- argument manifest from emit_matmul.mojo ---
    var manifest = Path(v.spv + ".args").read_text()
    var kernel_name = Path(v.spv + ".name").read_text()
    print("  entry point:", kernel_name.byte_length(), "chars")
    print("  manifest:", manifest.replace("\n", " | "))

    # --- inputs on host, copied to device ---
    var h_a = ctx.allocate_host(bytes)
    var h_b = ctx.allocate_host(bytes)
    var h_c = ctx.allocate_host(bytes)
    var pa = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_a)
    var pb = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_b)
    var pc = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_c)
    for i in range(n):
        for j in range(n):
            pa.unsafe_offset(i * n + j)[] = a_val(i, j)
            pb.unsafe_offset(i * n + j)[] = b_val(i, j)
            pc.unsafe_offset(i * n + j)[] = 0.0
    var d_a = ctx.allocate_device(bytes)
    var d_b = ctx.allocate_device(bytes)
    var d_c = ctx.allocate_device(bytes)
    ctx.memcpy_htod(d_a, h_a, bytes)
    ctx.memcpy_htod(d_b, h_b, bytes)
    ctx.memcpy_htod(d_c, h_c, bytes)

    var holders = List[Int]()
    for d in [d_a, d_b, d_c]:
        var h = ctx.allocate_host(8)
        Pointer[Int, MutUntrackedOrigin](unsafe_from_address=h)[] = d
        holders.append(h)

    # --- kernel + arguments ---
    var kernel = Kernel(
        ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
        v.spv, kernel_name,
    )
    var n_buffers = 0
    for line in manifest.split("\n"):
        var tok = List[String]()
        for t in line.split(" "):
            tok.append(String(t))
        if len(tok) < 3 or tok[0] != "arg":
            continue
        var idx = UInt32(Int(tok[1]))
        if tok[2] == "buffer":
            if n_buffers >= 3:
                raise Error("manifest has more than 3 buffer args")
            kernel.set_arg_pointer(idx, holders[n_buffers])
            n_buffers += 1
        elif tok[2] == "local":
            kernel.set_arg_local(idx, UInt64(Int(tok[3])))
        else:
            raise Error("unsupported manifest arg kind: " + tok[2])
    if n_buffers != 3:
        raise Error("expected 3 buffer args (A, B, C)")
    kernel.set_indirect_access(7)
    kernel.set_group_size(UInt32(v.gsize_x), UInt32(v.gsize_y), 1)
    var groups = ZeGroupCount(UInt32(v.gcount_x), UInt32(v.gcount_y), 1)

    # --- warmup (JIT), then timed launches ---
    var tw0 = perf_counter_ns()
    kernel.launch(groups)
    ctx.synchronize()
    var tw1 = perf_counter_ns()

    var t0 = perf_counter_ns()
    for _ in range(v.iters):
        kernel.launch(groups)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var avg_ms = Float64(t1 - t0) / Float64(v.iters) / 1.0e6
    var gflops = 2.0 * Float64(n) * Float64(n) * Float64(n) / (avg_ms / 1.0e3) / 1.0e9

    # --- correctness ---
    ctx.memcpy_dtoh(h_c, d_c, bytes)
    var checked = 0
    var mismatches = 0
    var max_rel: Float64 = 0.0
    var cells = List[Int]()
    if v.full_check:
        for i in range(n):
            for j in range(n):
                cells.append(i * n + j)
    else:
        for c in [0, n - 1, (n - 1) * n, n * n - 1]:
            cells.append(c)
        var x = 12345
        for _ in range(256):
            x = (x * 1103515245 + 12345) % 2147483648
            cells.append(x % (n * n))
    for cell in cells:
        var i = cell // n
        var j = cell % n
        var expected = ref_elem(n, i, j)
        var got = Float64(pc.unsafe_offset(cell)[])
        var ae = abs(got - expected)
        var re = ae / (abs(expected) + 1.0e-12)
        if re > max_rel and ae > 1.0e-3:
            max_rel = re
        if re > 1.0e-3 and ae > 1.0e-3:
            if mismatches < 5:
                print("  MISMATCH C[", i, ",", j, "]: expected", expected, "got", got)
            mismatches += 1
        checked += 1

    print("  N          :", n)
    print("  groups     :", v.gcount_x, "x", v.gcount_y, " group size:", v.gsize_x, "x", v.gsize_y)
    print("  check      :", "FULL" if v.full_check else "SAMPLED", "-", mismatches, "mismatches /", checked, "checked, max rel err", max_rel)
    print("  RESULT     :", "PASS" if mismatches == 0 else "FAIL")
    print("  warmup     :", Float64(tw1 - tw0) / 1.0e6, "ms (includes JIT)")
    print("  avg time   :", avg_ms, "ms over", v.iters, "iters")
    print("  perf       :", gflops, "GFLOP/s")

    _ = kernel^
    for h in holders:
        ctx.free_host(h)
    for h in [h_a, h_b, h_c]:
        ctx.free_host(h)
    for d in [d_a, d_b, d_c]:
        ctx.free_device(d)


def main() raises:
    print("=== G5: curriculum coarse matmul via our spirv64 backend on an Intel Max 1550 tile ===")
    print("  ZE_FLAT_DEVICE_HIERARCHY :", getenv("ZE_FLAT_DEVICE_HIERARCHY", "<unset>"))
    print("  ZE_AFFINITY_MASK         :", getenv("ZE_AFFINITY_MASK", "<unset>"))

    # Geometry must match the comptime constants in each curriculum file.
    var table = List[Variant]()
    table.append(Variant("check", "03d_matmul_check.mojo", "g5_check.spv", 256, 256, 1, 2, 2, 5, True))
    table.append(Variant("coarse", "03c_matmul_coarse.mojo", "g5_coarse.spv", 2048, 256, 1, 16, 16, 50, False))

    var wanted = getenv("G5_VARIANTS", "check coarse")
    print("  variants                 :", wanted)

    var ctx = IntelGPUContext()
    print("  device                   :", ctx.device_info().name)
    for ws in wanted.split(" "):
        var w = String(ws)
        if w == "":
            continue
        var found = False
        for v in table:
            if v.name == w:
                found = True
                try:
                    run_variant(ctx, v)
                except e:
                    print("  RESULT     : ERROR -", e)
        if not found:
            print("\n  unknown variant:", w)
    ctx.close()
