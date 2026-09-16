# g4_train_mlp_lz.mojo — G4: MOJO_CURRICULUM/04b_train_mlp_gpu.mojo training on an Aurora PVC tile.
#
# Kernels: the UNCHANGED Mojo source of 04b, via Mojo's Metal backend -> air2spir.py -> SPIR-V.
# 04b issues 15 kernel launches per epoch, but only 12 distinct kernels exist: the 4 SGD
# launches share one byte-identical instantiation, and the repeated generics (fwd1/fwd2,
# dW1/dW2, db1/db2) differ only in their comptime layout strides. Build them with
# g4_build_kernels.sh, which writes g4_<role>.spv plus a g4_<role>.spv.args manifest.
#
# Ordering: a Level Zero immediate command list is NOT in-order by default, and 04b's 15
# launches form a dependency chain (fwd1 -> biasrelu -> fwd2 -> ...). Without an explicit
# barrier the launches overlap and later kernels read buffers their producer has not written
# yet. ctx.barrier() (AURORA PATCH 6) orders them on the device, with no host round-trip.
#
# Argument binding (from the manifest):
#   buffer -> 8-byte USM holder containing the device data pointer   (TileTensor ABI, as in G2/G3)
#   const  -> 4-byte USM cell containing the scalar VALUE itself     (new in G4: 04b's kernels
#             take Int32 shapes and Float32 scalars, which Metal passes as constant-buffer
#             pointers; air2spir --const-as-global moves them to addrspace(1))
# Every argument is fixed for the whole run, so args are bound once and the epoch loop is
# nothing but launches. Buffers are reached indirectly through the holders, so each kernel
# still needs set_indirect_access(7) (G2 lesson 3).
#
# Correctness bar: reproduce 04b's loss curve (2.178 -> 0.000561) computed the same way —
# mean squared error of z2 against the host-side targets, printed at epoch 1 and every 50.
# The host-side LCG init is byte-identical to 04_train_mlp.mojo / 04b, so any divergence is
# the GPU path, not the data.
#
# Run: qsub -v MOJOFILE=g4_train_mlp_lz.mojo g1_run.pbs

from std.os import getenv
from std.pathlib import Path
from std.math import sqrt
from std.time import perf_counter_ns
from mojo_intel_gpu import IntelGPUContext, Kernel, ZeGroupCount

# --- 04b's comptime problem size (these must match the baked-in kernel constants) ---
comptime N = 256
comptime D = 2
comptime H = 16
comptime EPOCHS = 300
comptime LR = Float32(0.1)
comptime BLK = 16      # 2D block edge
comptime BLK1 = 64     # 1D block size

# --- buffer table indices ---
comptime B_X = 0
comptime B_Y = 1
comptime B_W1 = 2
comptime B_b1 = 3
comptime B_W2 = 4
comptime B_b2 = 5
comptime B_z1 = 6
comptime B_a1 = 7
comptime B_z2 = 8
comptime B_dz2 = 9
comptime B_dW2 = 10
comptime B_db2 = 11
comptime B_da1 = 12
comptime B_dz1 = 13
comptime B_dW1 = 14
comptime B_db1 = 15
comptime N_BUFS = 16

# --- const table indices ---
comptime C_N = 0       # Int32 256
comptime C_D = 1       # Int32 2
comptime C_H = 2       # Int32 16
comptime C_ONE = 3     # Int32 1
comptime C_DH = 4      # Int32 D*H = 32
comptime C_TWON = 5    # Float32 2/N
comptime C_LR = 6      # Float32 0.1
comptime N_CONSTS = 7


@fieldwise_init
struct Launch(Copyable, Movable):
    var name: String        # launch label (04b's K_* alias)
    var role: String        # which g4_<role>.spv provides the kernel
    var bufs: List[Int]     # buffer-table indices, in kernel-argument order
    var consts: List[Int]   # const-table indices, in kernel-argument order
    var gsize_x: Int        # threads per group (04b's block_dim)
    var gsize_y: Int
    var gcount_x: Int       # number of groups (04b's grid_dim)
    var gcount_y: Int


# host-side LCG, identical to 04_train_mlp.mojo / 04b_train_mlp_gpu.mojo
def next_rand(mut state: Int) -> Float64:
    state = (1103515245 * state + 12345) % (1 << 31)
    return Float64(state) / Float64(1 << 31)


def upload(mut ctx: IntelGPUContext, dst_device: Int, stage: Int, values: List[Float32]) raises:
    """Stage a host-side list into the USM scratch buffer, then copy it to the device.

    The command list is IMMEDIATE + ASYNCHRONOUS, so memcpy_htod only enqueues. The
    synchronize is mandatory: every upload reuses the one `stage` buffer, and without it the
    next upload overwrites bytes the previous copy has not read yet. This is init-time only.
    """
    var sp = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=stage)
    for k in range(len(values)):
        sp.unsafe_offset(k)[] = values[k]
    ctx.memcpy_htod(dst_device, stage, UInt64(len(values) * 4))
    ctx.synchronize()


def download(mut ctx: IntelGPUContext, dst_host: Int, src_device: Int, count: Int) raises:
    """Copy `count` floats device -> host and WAIT (same asynchronous-queue reason as upload)."""
    ctx.memcpy_dtoh(dst_host, src_device, UInt64(count * 4))
    ctx.synchronize()


def check_upload(mut ctx: IntelGPUContext, label: String, stage: Int, src_device: Int,
                 expected: List[Float32]) raises -> Int:
    """Read a buffer back and compare it with what we uploaded. Returns the mismatch count."""
    download(ctx, stage, src_device, len(expected))
    var sp = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=stage)
    var bad = 0
    for k in range(len(expected)):
        if sp.unsafe_offset(k)[] != expected[k]:
            if bad < 3:
                print("    upload mismatch:", label, "index", k,
                      "expected", expected[k], "got", sp.unsafe_offset(k)[])
            bad += 1
    return bad


def compare(label: String, stage: Int, expect: List[Float32], count: Int) -> Int:
    """Compare a downloaded buffer against a CPU reference. Prints a signature, returns bad count."""
    var sp = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=stage)
    var bad = 0
    var zeros = 0
    var max_rel = Float64(0)
    for k in range(count):
        var got = sp.unsafe_offset(k)[]
        if got == Float32(0):
            zeros += 1
        var ae = Float64(abs(got - expect[k]))
        var re = ae / (Float64(abs(expect[k])) + 1.0e-12)
        if re > 1.0e-4 and ae > 1.0e-5:
            if bad < 4:
                print("      ", label, "[", k, "] expected", expect[k], "got", got)
            bad += 1
            if re > max_rel:
                max_rel = re
    print("    ", label, ": ", bad, "/", count, "wrong,", zeros, "exact zeros, max rel err", max_rel)
    return bad


def verify_forward(mut ctx: IntelGPUContext, stage: Int, dev_z1: Int, dev_a1: Int, dev_z2: Int,
                   Xh: List[Float32], W1h: List[Float32], W2h: List[Float32]) raises:
    """After one forward pass, check z1, a1 and z2 against a CPU reference (b1 = b2 = 0).

    Tells us WHICH stage first diverges, and whether the wrong entries are exact zeros
    (a coverage/write problem) or wrong values (an arithmetic or routing problem).
    """
    var z1_ref = List[Float32](capacity=N * H)
    for i in range(N):
        for j in range(H):
            var acc = Float32(0)
            for k in range(D):
                acc += Xh[i * D + k] * W1h[k * H + j]
            z1_ref.append(acc)
    var a1_ref = List[Float32](capacity=N * H)
    for k in range(N * H):
        a1_ref.append(z1_ref[k] if z1_ref[k] > Float32(0) else Float32(0))
    var z2_ref = List[Float32](capacity=N)
    for i in range(N):
        var acc = Float32(0)
        for j in range(H):
            acc += a1_ref[i * H + j] * W2h[j]
        z2_ref.append(acc)

    print("\n  forward verification (one pass, b1 = b2 = 0):")
    download(ctx, stage, dev_z1, N * H)
    _ = compare("z1", stage, z1_ref, N * H)
    download(ctx, stage, dev_a1, N * H)
    _ = compare("a1", stage, a1_ref, N * H)
    download(ctx, stage, dev_z2, N)
    _ = compare("z2", stage, z2_ref, N)


def mse(mut ctx: IntelGPUContext, h_z2: Int, d_z2: Int, targets: List[Float32]) raises -> Float32:
    """04b's loss: mean squared error of the network output z2 against the targets."""
    download(ctx, h_z2, d_z2, len(targets))
    var pz = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=h_z2)
    var loss = Float32(0)
    for i in range(len(targets)):
        var d = pz.unsafe_offset(i)[] - targets[i]
        loss += d * d
    return loss / Float32(len(targets))


def build_table() -> List[Launch]:
    """04b's 15 launches per epoch, in order."""
    var t = List[Launch]()
    # forward
    t.append(Launch("K_fwd1", "fwd1", [B_X, B_W1, B_z1], [C_N, C_D, C_H], BLK, BLK, 1, 16))
    t.append(Launch("K_biasrelu", "biasrelu", [B_z1, B_b1, B_a1], [C_N, C_H], BLK, BLK, 1, 16))
    t.append(Launch("K_fwd2", "fwd2", [B_a1, B_W2, B_z2], [C_N, C_H, C_ONE], BLK, BLK, 1, 16))
    t.append(Launch("K_addb2", "addb2", [B_z2, B_b2], [C_N], BLK1, 1, 4, 1))
    # backward
    t.append(Launch("K_dz2", "dz2", [B_z2, B_Y, B_dz2], [C_N, C_TWON], BLK1, 1, 4, 1))
    t.append(Launch("K_dW2", "dW2", [B_a1, B_dz2, B_dW2], [C_N, C_H, C_ONE], BLK, BLK, 1, 1))
    t.append(Launch("K_db2", "db2", [B_dz2, B_db2], [C_N, C_ONE], BLK1, 1, 1, 1))
    t.append(Launch("K_da1", "da1", [B_dz2, B_W2, B_da1], [C_N, C_ONE, C_H], BLK, BLK, 1, 16))
    t.append(Launch("K_relugrad", "relugrad", [B_z1, B_da1, B_dz1], [C_N, C_H], BLK, BLK, 1, 16))
    t.append(Launch("K_dW1", "dW1", [B_X, B_dz1, B_dW1], [C_N, C_D, C_H], BLK, BLK, 1, 1))
    t.append(Launch("K_db1", "db1", [B_dz1, B_db1], [C_N, C_H], BLK1, 1, 1, 1))
    # SGD (all four share g4_sgd.spv)
    t.append(Launch("K_sgdW1", "sgd", [B_W1, B_dW1], [C_DH, C_LR], BLK1, 1, 1, 1))
    t.append(Launch("K_sgdb1", "sgd", [B_b1, B_db1], [C_H, C_LR], BLK1, 1, 1, 1))
    t.append(Launch("K_sgdW2", "sgd", [B_W2, B_dW2], [C_H, C_LR], BLK1, 1, 1, 1))
    t.append(Launch("K_sgdb2", "sgd", [B_b2, B_db2], [C_ONE, C_LR], BLK1, 1, 1, 1))
    return t^


def main() raises:
    print("=== G4: MOJO_CURRICULUM/04b MLP training on an Intel Max 1550 tile ===")
    print("  ZE_FLAT_DEVICE_HIERARCHY :", getenv("ZE_FLAT_DEVICE_HIERARCHY", "<unset>"))
    print("  ZE_AFFINITY_MASK         :", getenv("ZE_AFFINITY_MASK", "<unset>"))
    print("  N=", N, " D=", D, " H=", H, " epochs=", EPOCHS, " lr=", LR)

    var ctx = IntelGPUContext()
    print("  device                   :", ctx.device_info().name)

    # ---- host-side init via the shared LCG (order: X, then W1, then W2) ----
    var Xh = List[Float32](capacity=N * D)
    var Yh = List[Float32](capacity=N)
    var W1h = List[Float32](capacity=D * H)
    var W2h = List[Float32](capacity=H)
    var state = 1
    for _ in range(N * D):
        Xh.append(Float32(4.0 * next_rand(state) - 2.0))
    for i in range(N):
        var x0 = Xh[i * D + 0]
        var x1 = Xh[i * D + 1]
        Yh.append(sqrt(x0 * x0 + x1 * x1))
    for _ in range(D * H):
        W1h.append(Float32(next_rand(state) - 0.5))
    for _ in range(H):
        W2h.append(Float32(next_rand(state) - 0.5))

    # ---- device buffers (element counts mirror 04b's enqueue_create_buffer calls) ----
    var counts = List[Int]()
    for _ in range(N_BUFS):
        counts.append(0)
    counts[B_X] = N * D
    counts[B_Y] = N
    counts[B_W1] = D * H
    counts[B_b1] = H
    counts[B_W2] = H
    counts[B_b2] = 1
    counts[B_z1] = N * H
    counts[B_a1] = N * H
    counts[B_z2] = N
    counts[B_dz2] = N
    counts[B_dW2] = H
    counts[B_db2] = 1
    counts[B_da1] = N * H
    counts[B_dz1] = N * H
    counts[B_dW1] = D * H
    counts[B_db1] = H

    var dev = List[Int]()
    var holders = List[Int]()
    for i in range(N_BUFS):
        var nbytes = UInt64(counts[i] * 4)
        var d = ctx.allocate_device(nbytes)
        ctx.memset_device(d, 0, nbytes)          # b1/b2 start at zero; the rest are fully written
        dev.append(d)
        var h = ctx.allocate_host(8)             # TileTensor-ABI holder: 8 bytes of device pointer
        Pointer[Int, MutUntrackedOrigin](unsafe_from_address=h)[] = d
        holders.append(h)

    # ---- upload X, Y, W1, W2 ----
    var stage = ctx.allocate_host(UInt64(N * H * 4))   # big enough for any of them
    upload(ctx, dev[B_X], stage, Xh)
    upload(ctx, dev[B_Y], stage, Yh)
    upload(ctx, dev[B_W1], stage, W1h)
    upload(ctx, dev[B_W2], stage, W2h)

    # Read the uploads back and compare, so a data problem can never be mistaken for a
    # kernel problem. Reuses `stage` as the landing buffer (each check completes before the next).
    var bad = check_upload(ctx, "X", stage, dev[B_X], Xh)
    bad += check_upload(ctx, "Y", stage, dev[B_Y], Yh)
    bad += check_upload(ctx, "W1", stage, dev[B_W1], W1h)
    bad += check_upload(ctx, "W2", stage, dev[B_W2], W2h)
    print("  upload check             :", "OK" if bad == 0 else String(bad) + " MISMATCHES")

    # ---- scalar constant cells (the kernel loads the VALUE through this pointer) ----
    var consts = List[Int]()
    for _ in range(N_CONSTS):
        consts.append(ctx.allocate_host(4))

    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=consts[C_N])[] = Int32(N)
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=consts[C_D])[] = Int32(D)
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=consts[C_H])[] = Int32(H)
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=consts[C_ONE])[] = Int32(1)
    Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=consts[C_DH])[] = Int32(D * H)
    Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=consts[C_TWON])[] = Float32(2.0 / Float64(N))
    Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=consts[C_LR])[] = LR

    # ---- create and bind one Kernel per launch (args never change across epochs) ----
    var table = build_table()
    var kernels = List[Kernel]()
    print("\n  binding", len(table), "launches over 12 distinct kernels:")
    for entry in table:
        var spv = "g4_" + entry.role + ".spv"
        var manifest = Path(spv + ".args").read_text()
        var kernel = Kernel(
            ctx.library(), ctx.context(), ctx.device(), ctx.command_list(),
            spv, "g4_" + entry.role,
        )
        var n_buf = 0
        var n_const = 0
        for line in manifest.split("\n"):
            var tok = List[String]()
            for t in line.split(" "):
                tok.append(String(t))
            if len(tok) < 3 or tok[0] != "arg":
                continue
            var idx = UInt32(Int(tok[1]))
            if tok[2] == "buffer":
                if n_buf >= len(entry.bufs):
                    raise Error(entry.name + ": manifest has more buffer args than the table")
                kernel.set_arg_pointer(idx, holders[entry.bufs[n_buf]])
                n_buf += 1
            elif tok[2] == "const":
                if n_const >= len(entry.consts):
                    raise Error(entry.name + ": manifest has more const args than the table")
                kernel.set_arg_pointer(idx, consts[entry.consts[n_const]])
                n_const += 1
            else:
                raise Error(entry.name + ": unsupported manifest arg kind: " + tok[2])
        if n_buf != len(entry.bufs) or n_const != len(entry.consts):
            raise Error(
                entry.name + ": manifest wants " + String(n_buf) + " buffers / "
                + String(n_const) + " consts, table supplies " + String(len(entry.bufs))
                + " / " + String(len(entry.consts))
            )
        kernel.set_indirect_access(7)
        kernel.set_group_size(UInt32(entry.gsize_x), UInt32(entry.gsize_y), 1)
        print("   ", entry.name, "->", spv, " groups", entry.gcount_x, "x", entry.gcount_y,
              " group size", entry.gsize_x, "x", entry.gsize_y,
              " (", n_buf, "buffers,", n_const, "consts )")
        kernels.append(kernel^)

    var groups = List[ZeGroupCount]()
    for entry in table:
        groups.append(ZeGroupCount(UInt32(entry.gcount_x), UInt32(entry.gcount_y), 1))

    # ---- loss readback buffer ----
    var h_z2 = ctx.allocate_host(UInt64(N * 4))

    # ---- optional: one forward pass, checked stage by stage (G4_VERIFY=1) ----
    if getenv("G4_VERIFY", "1") == "1":
        for i in range(4):                      # fwd1, biasrelu, fwd2, addb2
            kernels[i].launch(groups[i])
            ctx.barrier()
        ctx.synchronize()
        verify_forward(ctx, stage, dev[B_z1], dev[B_a1], dev[B_z2], Xh, W1h, W2h)

    # ---- training loop: 04b's order, 15 launches per epoch ----
    print("")
    var t0 = perf_counter_ns()          # reset after epoch 1 (JIT warmup), like 04b
    for epoch in range(1, EPOCHS + 1):
        if epoch == 2:
            ctx.synchronize()
            t0 = perf_counter_ns()
        for i in range(len(kernels)):
            kernels[i].launch(groups[i])
            ctx.barrier()          # 04b's 15 launches are a dependency chain, not independent work
        if epoch % 50 == 0 or epoch == 1:
            ctx.synchronize()
            print("  epoch", epoch, "  loss", mse(ctx, h_z2, dev[B_z2], Yh))

    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("\n  train time:", Float64(t1 - t0) / Float64(EPOCHS - 1) / 1.0e6, "ms/epoch (excl. warmup)")
    print("\nfinal loss:", mse(ctx, h_z2, dev[B_z2], Yh))
    print("  reference (04b on Apple GPU / Polaris A100): 2.178 -> 0.000561")

    while len(kernels) > 0:
        _ = kernels.pop()
    ctx.free_host(h_z2)
    ctx.free_host(stage)
    for c in consts:
        ctx.free_host(c)
    for h in holders:
        ctx.free_host(h)
    for d in dev:
        ctx.free_device(d)
    ctx.close()
