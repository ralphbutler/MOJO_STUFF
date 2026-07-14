# 04b_train_mlp_gpu.mojo — 04's training loop, moved onto the Apple GPU.
#
# Same MLP, same task, same LCG init as 04_train_mlp.mojo — so this must reproduce the
# exact loss curve (2.178 → 0.000561), which is how we know the GPU kernels are correct.
# The difference is WHERE the work runs: every forward/backward step is a GPU kernel, and
# the matmuls reuse 03's design — one GPU thread per output element (global_idx), the naive
# kernel generalized from square N×N to the rectangular and transposed shapes backprop needs.
#
# Honest note (same lesson as 02): at THIS size (N=256, H=16) the GPU is NOT faster — the
# matrices are tiny and we pay ~15 kernel launches per epoch. The win comes at real model
# dimensions; bump N and H (comptime, below) and the GPU work starts to dominate the launch
# overhead. The point here is that training — forward AND backward — is "just" matmuls, and
# the 03 kernel is enough to run all of it on the accelerator.

from std.math import ceildiv, sqrt
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, TensorLayout, row_major

comptime dtype = DType.float32
comptime N = 256          # training points
comptime D = 2            # input dims
comptime H = 16           # hidden width
comptime EPOCHS = 300
comptime LR = Scalar[dtype](0.1)
comptime BLK = 16         # 2D block edge (16×16 threads)
comptime BLK1 = 64        # 1D block size

# 2D layouts (rows, cols) for every matrix in the net
comptime L_ND = row_major[N, D]()      # X
comptime L_N1 = row_major[N, 1]()      # Y, z2, dz2
comptime L_DH = row_major[D, H]()      # W1, dW1
comptime L_H1 = row_major[H, 1]()      # W2, dW2
comptime L_NH = row_major[N, H]()      # z1, a1, da1, dz1
comptime L_1H = row_major[1, H]()      # b1, db1
comptime L_11 = row_major[1, 1]()      # b2, db2
# flat (1D) layouts — a second view of the same buffers, for the SGD update
comptime L_DHf = row_major[D * H]()
comptime L_Hf = row_major[H]()
comptime L_1f = row_major[1]()


# ---- matmul family (03's one-thread-per-output design, generalized) ----

# C[M,P] = A[M,K] @ B[K,P]
def matmul_kernel[AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    A: TileTensor[dtype, AL, MutAnyOrigin], B: TileTensor[dtype, BL, MutAnyOrigin],
    C: TileTensor[dtype, CL, MutAnyOrigin], M: Int, K: Int, P: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var row = global_idx.y
    var col = global_idx.x
    if row < M and col < P:
        var acc = Scalar[dtype](0)
        for k in range(K):
            acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](B[k, col])
        C[row, col] = rebind[C.ElementType](acc)

# C[K,P] = A[M,K]^T @ B[M,P]   (contract over leading dim M)
def matmul_at_b_kernel[AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    A: TileTensor[dtype, AL, MutAnyOrigin], B: TileTensor[dtype, BL, MutAnyOrigin],
    C: TileTensor[dtype, CL, MutAnyOrigin], M: Int, K: Int, P: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var row = global_idx.y   # 0..K
    var col = global_idx.x   # 0..P
    if row < K and col < P:
        var acc = Scalar[dtype](0)
        for m in range(M):
            acc += rebind[Scalar[dtype]](A[m, row]) * rebind[Scalar[dtype]](B[m, col])
        C[row, col] = rebind[C.ElementType](acc)

# C[M,P] = A[M,K] @ B[P,K]^T
def matmul_a_bt_kernel[AL: TensorLayout, BL: TensorLayout, CL: TensorLayout](
    A: TileTensor[dtype, AL, MutAnyOrigin], B: TileTensor[dtype, BL, MutAnyOrigin],
    C: TileTensor[dtype, CL, MutAnyOrigin], M: Int, K: Int, P: Int,
):
    comptime assert A.flat_rank == 2 and B.flat_rank == 2 and C.flat_rank == 2
    var row = global_idx.y   # 0..M
    var col = global_idx.x   # 0..P
    if row < M and col < P:
        var acc = Scalar[dtype](0)
        for k in range(K):
            acc += rebind[Scalar[dtype]](A[row, k]) * rebind[Scalar[dtype]](B[col, k])
        C[row, col] = rebind[C.ElementType](acc)


# ---- elementwise + reduction kernels ----

# Z[r,c] += B[0,c]; A[r,c] = relu(Z[r,c])   (forward bias-add + activation)
def bias_relu_kernel[ZL: TensorLayout, BL: TensorLayout, AL: TensorLayout](
    Z: TileTensor[dtype, ZL, MutAnyOrigin], B: TileTensor[dtype, BL, MutAnyOrigin],
    A: TileTensor[dtype, AL, MutAnyOrigin], M: Int, P: Int,
):
    comptime assert Z.flat_rank == 2 and B.flat_rank == 2 and A.flat_rank == 2
    var row = global_idx.y
    var col = global_idx.x
    if row < M and col < P:
        var v = rebind[Scalar[dtype]](Z[row, col]) + rebind[Scalar[dtype]](B[0, col])
        Z[row, col] = rebind[Z.ElementType](v)
        A[row, col] = rebind[A.ElementType](v if v > 0 else Scalar[dtype](0))

# Z[r,0] += B[0,0]   (add the single output bias)
def add_bias1_kernel[ZL: TensorLayout, BL: TensorLayout](
    Z: TileTensor[dtype, ZL, MutAnyOrigin], B: TileTensor[dtype, BL, MutAnyOrigin], M: Int,
):
    comptime assert Z.flat_rank == 2 and B.flat_rank == 2
    var r = global_idx.x
    if r < M:
        var v = rebind[Scalar[dtype]](Z[r, 0]) + rebind[Scalar[dtype]](B[0, 0])
        Z[r, 0] = rebind[Z.ElementType](v)

# Dz[r,0] = scale * (Z[r,0] - Y[r,0])   (gradient of MSE wrt the output pre-activation)
def dz2_kernel[ZL: TensorLayout, YL: TensorLayout, DL: TensorLayout](
    Z: TileTensor[dtype, ZL, MutAnyOrigin], Y: TileTensor[dtype, YL, MutAnyOrigin],
    Dz: TileTensor[dtype, DL, MutAnyOrigin], M: Int, scale: Scalar[dtype],
):
    comptime assert Z.flat_rank == 2 and Y.flat_rank == 2 and Dz.flat_rank == 2
    var r = global_idx.x
    if r < M:
        var v = scale * (rebind[Scalar[dtype]](Z[r, 0]) - rebind[Scalar[dtype]](Y[r, 0]))
        Dz[r, 0] = rebind[Dz.ElementType](v)

# Dz[r,c] = Da[r,c] if Z[r,c] > 0 else 0   (relu gradient, using the saved pre-activation Z)
def relu_grad_kernel[ZL: TensorLayout, AL: TensorLayout, DL: TensorLayout](
    Z: TileTensor[dtype, ZL, MutAnyOrigin], Da: TileTensor[dtype, AL, MutAnyOrigin],
    Dz: TileTensor[dtype, DL, MutAnyOrigin], M: Int, P: Int,
):
    comptime assert Z.flat_rank == 2 and Da.flat_rank == 2 and Dz.flat_rank == 2
    var row = global_idx.y
    var col = global_idx.x
    if row < M and col < P:
        var g = rebind[Scalar[dtype]](Da[row, col]) if rebind[Scalar[dtype]](Z[row, col]) > 0 else Scalar[dtype](0)
        Dz[row, col] = rebind[Dz.ElementType](g)

# Out[0,j] = sum_i In[i,j]   (column sum -> bias gradient; one thread per column)
def colsum_kernel[IL: TensorLayout, OL: TensorLayout](
    In: TileTensor[dtype, IL, MutAnyOrigin], Out: TileTensor[dtype, OL, MutAnyOrigin],
    M: Int, P: Int,
):
    comptime assert In.flat_rank == 2 and Out.flat_rank == 2
    var j = global_idx.x
    if j < P:
        var s = Scalar[dtype](0)
        for i in range(M):
            s += rebind[Scalar[dtype]](In[i, j])
        Out[0, j] = rebind[Out.ElementType](s)

# p[k] -= lr * g[k]   (SGD update over a flat 1D view of a parameter buffer)
def sgd_kernel[L: TensorLayout](
    p: TileTensor[dtype, L, MutAnyOrigin], g: TileTensor[dtype, L, MutAnyOrigin],
    n: Int, lr: Scalar[dtype],
):
    comptime assert p.flat_rank == 1 and g.flat_rank == 1
    var k = global_idx.x
    if k < n:
        p[k] = rebind[p.ElementType](rebind[Scalar[dtype]](p[k]) - lr * rebind[Scalar[dtype]](g[k]))


# host-side LCG, identical to train_mlp_reference.py / 04_train_mlp.mojo
def next_rand(mut state: Int) -> Float64:
    state = (1103515245 * state + 12345) % (1 << 31)
    return Float64(state) / Float64(1 << 31)


def main() raises:
    comptime assert has_accelerator(), "Requires a GPU"
    var ctx = DeviceContext()

    # ---- host-side init via the shared LCG (order: X, then W1, then W2) ----
    var Xh = List[Scalar[dtype]](capacity=N * D)
    var Yh = List[Scalar[dtype]](capacity=N)
    var W1h = List[Scalar[dtype]](capacity=D * H)
    var W2h = List[Scalar[dtype]](capacity=H)
    var state = 1
    for _ in range(N * D):
        Xh.append(Scalar[dtype](4.0 * next_rand(state) - 2.0))
    for i in range(N):
        var x0 = Xh[i * D + 0]; var x1 = Xh[i * D + 1]
        Yh.append(sqrt(x0 * x0 + x1 * x1))
    for _ in range(D * H):
        W1h.append(Scalar[dtype](next_rand(state) - 0.5))
    for _ in range(H):
        W2h.append(Scalar[dtype](next_rand(state) - 0.5))

    # ---- device buffers ----
    var X_buf = ctx.enqueue_create_buffer[dtype](N * D)
    var Y_buf = ctx.enqueue_create_buffer[dtype](N)
    var W1_buf = ctx.enqueue_create_buffer[dtype](D * H)
    var b1_buf = ctx.enqueue_create_buffer[dtype](H)
    var W2_buf = ctx.enqueue_create_buffer[dtype](H)
    var b2_buf = ctx.enqueue_create_buffer[dtype](1)
    var z1_buf = ctx.enqueue_create_buffer[dtype](N * H)
    var a1_buf = ctx.enqueue_create_buffer[dtype](N * H)
    var z2_buf = ctx.enqueue_create_buffer[dtype](N)
    var dz2_buf = ctx.enqueue_create_buffer[dtype](N)
    var dW2_buf = ctx.enqueue_create_buffer[dtype](H)
    var db2_buf = ctx.enqueue_create_buffer[dtype](1)
    var da1_buf = ctx.enqueue_create_buffer[dtype](N * H)
    var dz1_buf = ctx.enqueue_create_buffer[dtype](N * H)
    var dW1_buf = ctx.enqueue_create_buffer[dtype](D * H)
    var db1_buf = ctx.enqueue_create_buffer[dtype](H)
    b1_buf.enqueue_fill(0.0)
    b2_buf.enqueue_fill(0.0)

    # upload host init -> device
    with X_buf.map_to_host() as h:
        for k in range(N * D): h[k] = Xh[k]
    with Y_buf.map_to_host() as h:
        for k in range(N): h[k] = Yh[k]
    with W1_buf.map_to_host() as h:
        for k in range(D * H): h[k] = W1h[k]
    with W2_buf.map_to_host() as h:
        for k in range(H): h[k] = W2h[k]

    # ---- TileTensor views (2D for compute, flat for SGD) ----
    var X = TileTensor(X_buf, L_ND); var Y = TileTensor(Y_buf, L_N1)
    var W1 = TileTensor(W1_buf, L_DH); var b1 = TileTensor(b1_buf, L_1H)
    var W2 = TileTensor(W2_buf, L_H1); var b2 = TileTensor(b2_buf, L_11)
    var z1 = TileTensor(z1_buf, L_NH); var a1 = TileTensor(a1_buf, L_NH)
    var z2 = TileTensor(z2_buf, L_N1)
    var dz2 = TileTensor(dz2_buf, L_N1); var dW2 = TileTensor(dW2_buf, L_H1)
    var db2 = TileTensor(db2_buf, L_11)
    var da1 = TileTensor(da1_buf, L_NH); var dz1 = TileTensor(dz1_buf, L_NH)
    var dW1 = TileTensor(dW1_buf, L_DH); var db1 = TileTensor(db1_buf, L_1H)
    # flat views onto the same buffers
    var W1f = TileTensor(W1_buf, L_DHf); var dW1f = TileTensor(dW1_buf, L_DHf)
    var b1f = TileTensor(b1_buf, L_Hf); var db1f = TileTensor(db1_buf, L_Hf)
    var W2f = TileTensor(W2_buf, L_Hf); var dW2f = TileTensor(dW2_buf, L_Hf)
    var b2f = TileTensor(b2_buf, L_1f); var db2f = TileTensor(db2_buf, L_1f)

    # ---- bind kernels to their concrete layouts ----
    comptime K_fwd1 = matmul_kernel[type_of(L_ND), type_of(L_DH), type_of(L_NH)]
    comptime K_fwd2 = matmul_kernel[type_of(L_NH), type_of(L_H1), type_of(L_N1)]
    comptime K_biasrelu = bias_relu_kernel[type_of(L_NH), type_of(L_1H), type_of(L_NH)]
    comptime K_addb2 = add_bias1_kernel[type_of(L_N1), type_of(L_11)]
    comptime K_dz2 = dz2_kernel[type_of(L_N1), type_of(L_N1), type_of(L_N1)]
    comptime K_dW2 = matmul_at_b_kernel[type_of(L_NH), type_of(L_N1), type_of(L_H1)]
    comptime K_db2 = colsum_kernel[type_of(L_N1), type_of(L_11)]
    comptime K_da1 = matmul_a_bt_kernel[type_of(L_N1), type_of(L_H1), type_of(L_NH)]
    comptime K_relugrad = relu_grad_kernel[type_of(L_NH), type_of(L_NH), type_of(L_NH)]
    comptime K_dW1 = matmul_at_b_kernel[type_of(L_ND), type_of(L_NH), type_of(L_DH)]
    comptime K_db1 = colsum_kernel[type_of(L_NH), type_of(L_1H)]
    comptime K_sgdW1 = sgd_kernel[type_of(L_DHf)]
    comptime K_sgdb1 = sgd_kernel[type_of(L_Hf)]
    comptime K_sgdW2 = sgd_kernel[type_of(L_Hf)]
    comptime K_sgdb2 = sgd_kernel[type_of(L_1f)]

    var two_over_N = Scalar[dtype](2.0 / Float64(N))
    print("N=", N, " hidden=", H, " epochs=", EPOCHS, " lr=", LR, " (Mojo, Apple GPU)")
    print("")

    comptime g2_NH = (ceildiv(H, BLK), ceildiv(N, BLK))   # grid for N×H outputs
    comptime g2_N1 = (ceildiv(1, BLK), ceildiv(N, BLK))   # grid for N×1 outputs
    comptime g2_H1 = (ceildiv(1, BLK), ceildiv(H, BLK))   # grid for H×1 outputs
    comptime g2_DH = (ceildiv(H, BLK), ceildiv(D, BLK))   # grid for D×H outputs
    comptime blk2 = (BLK, BLK)

    var t0 = perf_counter_ns()             # reset at epoch 2 below (this value unused)
    for epoch in range(1, EPOCHS + 1):
        if epoch == 2:                     # start timing after epoch 1 (kernel compile)
            ctx.synchronize()
            t0 = perf_counter_ns()
        # forward
        ctx.enqueue_function[K_fwd1](X, W1, z1, N, D, H, grid_dim=g2_NH, block_dim=blk2)
        ctx.enqueue_function[K_biasrelu](z1, b1, a1, N, H, grid_dim=g2_NH, block_dim=blk2)
        ctx.enqueue_function[K_fwd2](a1, W2, z2, N, H, 1, grid_dim=g2_N1, block_dim=blk2)
        ctx.enqueue_function[K_addb2](z2, b2, N, grid_dim=ceildiv(N, BLK1), block_dim=BLK1)
        # backward
        ctx.enqueue_function[K_dz2](z2, Y, dz2, N, two_over_N, grid_dim=ceildiv(N, BLK1), block_dim=BLK1)
        ctx.enqueue_function[K_dW2](a1, dz2, dW2, N, H, 1, grid_dim=g2_H1, block_dim=blk2)
        ctx.enqueue_function[K_db2](dz2, db2, N, 1, grid_dim=ceildiv(1, BLK1), block_dim=BLK1)
        ctx.enqueue_function[K_da1](dz2, W2, da1, N, 1, H, grid_dim=g2_NH, block_dim=blk2)
        ctx.enqueue_function[K_relugrad](z1, da1, dz1, N, H, grid_dim=g2_NH, block_dim=blk2)
        ctx.enqueue_function[K_dW1](X, dz1, dW1, N, D, H, grid_dim=g2_DH, block_dim=blk2)
        ctx.enqueue_function[K_db1](dz1, db1, N, H, grid_dim=ceildiv(H, BLK1), block_dim=BLK1)
        # SGD
        ctx.enqueue_function[K_sgdW1](W1f, dW1f, D * H, LR, grid_dim=ceildiv(D * H, BLK1), block_dim=BLK1)
        ctx.enqueue_function[K_sgdb1](b1f, db1f, H, LR, grid_dim=ceildiv(H, BLK1), block_dim=BLK1)
        ctx.enqueue_function[K_sgdW2](W2f, dW2f, H, LR, grid_dim=ceildiv(H, BLK1), block_dim=BLK1)
        ctx.enqueue_function[K_sgdb2](b2f, db2f, 1, LR, grid_dim=1, block_dim=BLK1)

        if epoch % 50 == 0 or epoch == 1:
            ctx.synchronize()
            var loss = Scalar[dtype](0)
            with z2_buf.map_to_host() as hz:
                for i in range(N):
                    var d = hz[i] - Yh[i]
                    loss += d * d
            print("  epoch", epoch, "  loss", loss / Scalar[dtype](N))

    ctx.synchronize()
    var t1 = perf_counter_ns()
    print("\n  train time:", Float64(t1 - t0) / Float64(EPOCHS - 1) / 1.0e6, "ms/epoch (excl. warmup)")

    var final = Scalar[dtype](0)
    with z2_buf.map_to_host() as hz:
        for i in range(N):
            var d = hz[i] - Yh[i]
            final += d * d
    print("\nfinal loss:", final / Scalar[dtype](N))
