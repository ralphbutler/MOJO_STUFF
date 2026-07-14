# 04_train_mlp.mojo — training a neural net in PURE MOJO, gradients by hand.
#
# The payoff of the whole curriculum. MAX has no autograd (03 built inference kernels;
# MAX serves models) — so when you train without a framework, YOU are the autograd:
# every gradient below is derived on paper and coded by hand. A 2-layer MLP
# (2 -> H -> 1) learns the radius r = sqrt(x0^2 + x1^2), the same nonlinear task as the
# PyTorch/Nabla demos, so a bare linear model can't do it — the hidden layer must.
#
# The forward pass is matmul + relu + matmul (the matmul you optimized in 03). The
# backward pass is *more of the same operation*: transposed matmuls plus an elementwise
# relu-gradient. Kept naive (triple loops, small N) — 03 already covered making matmul
# fast; here the point is the training math, not peak GFLOP/s.
#
# CORRECTNESS: train_mlp_reference.py runs this exact setup in NumPy from the same LCG,
# so both start from identical data + weights and their loss curves must track. Expect
# ~2.18 at epoch 1 falling to ~5e-4 by epoch 300 (small drift is float32 order-of-sum).

from std.math import sqrt
from std.time import perf_counter_ns

comptime dtype = DType.float32
comptime N = 256          # training points
comptime D = 2            # input dims (x0, x1)
comptime H = 16           # hidden width
comptime EPOCHS = 300
comptime LR = Scalar[dtype](0.1)


def zeros(n: Int) -> List[Scalar[dtype]]:
    var v = List[Scalar[dtype]](capacity=n)
    for _ in range(n):
        v.append(0)
    return v^


# 32-bit LCG -> float in [0,1). Identical constants/modulus to train_mlp_reference.py,
# so the two programs generate bit-comparable data and initial weights.
def next_rand(mut state: Int) -> Float64:
    state = (1103515245 * state + 12345) % (1 << 31)
    return Float64(state) / Float64(1 << 31)


# C[M,P] = A[M,K] @ B[K,P]
def matmul(A: List[Scalar[dtype]], B: List[Scalar[dtype]], mut C: List[Scalar[dtype]],
           M: Int, K: Int, P: Int):
    for i in range(M):
        for j in range(P):
            var acc = Scalar[dtype](0)
            for k in range(K):
                acc += A[i * K + k] * B[k * P + j]
            C[i * P + j] = acc


# C[K,P] = A[M,K]^T @ B[M,P]   (contract over the shared leading dim M)
def matmul_at_b(A: List[Scalar[dtype]], B: List[Scalar[dtype]], mut C: List[Scalar[dtype]],
                M: Int, K: Int, P: Int):
    for i in range(K):
        for j in range(P):
            var acc = Scalar[dtype](0)
            for m in range(M):
                acc += A[m * K + i] * B[m * P + j]
            C[i * P + j] = acc


# C[M,P] = A[M,K] @ B[P,K]^T   (B stored row-major as P×K)
def matmul_a_bt(A: List[Scalar[dtype]], B: List[Scalar[dtype]], mut C: List[Scalar[dtype]],
                M: Int, K: Int, P: Int):
    for i in range(M):
        for j in range(P):
            var acc = Scalar[dtype](0)
            for k in range(K):
                acc += A[i * K + k] * B[j * K + k]
            C[i * P + j] = acc


def main():
    # --- buffers ---
    var X = zeros(N * D)
    var Y = zeros(N)
    var W1 = zeros(D * H); var b1 = zeros(H)
    var W2 = zeros(H); var b2 = zeros(1)
    var z1 = zeros(N * H); var a1 = zeros(N * H); var z2 = zeros(N)
    var dz2 = zeros(N); var dW2 = zeros(H); var db2 = zeros(1)
    var da1 = zeros(N * H); var dz1 = zeros(N * H)
    var dW1 = zeros(D * H); var db1 = zeros(H)

    # --- deterministic init via the shared LCG (order: X, then W1, then W2) ---
    var state = 1
    for i in range(N):
        for j in range(D):
            X[i * D + j] = Scalar[dtype](4.0 * next_rand(state) - 2.0)   # x in [-2,2]
    for i in range(N):
        var x0 = X[i * D + 0]; var x1 = X[i * D + 1]
        Y[i] = sqrt(x0 * x0 + x1 * x1)                                   # target radius
    for i in range(D):
        for j in range(H):
            W1[i * H + j] = Scalar[dtype](next_rand(state) - 0.5)
    for i in range(H):
        W2[i] = Scalar[dtype](next_rand(state) - 0.5)

    var two_over_N = Scalar[dtype](2.0 / Float64(N))
    print("N=", N, " hidden=", H, " epochs=", EPOCHS, " lr=", LR, " (pure Mojo)")
    print("")

    var t0 = perf_counter_ns()             # reset at epoch 2 below (this value unused)
    for epoch in range(1, EPOCHS + 1):
        if epoch == 2:                     # start timing after the first (cold) epoch
            t0 = perf_counter_ns()
        # --- forward: z1 = X@W1 + b1 ; a1 = relu(z1) ; z2 = a1@W2 + b2 ---
        matmul(X, W1, z1, N, D, H)
        for i in range(N):
            for j in range(H):
                var v = z1[i * H + j] + b1[j]
                z1[i * H + j] = v                       # store pre-activation (needed in backward)
                a1[i * H + j] = v if v > 0 else Scalar[dtype](0)   # relu
        matmul(a1, W2, z2, N, H, 1)
        for i in range(N):
            z2[i] = z2[i] + b2[0]

        # --- loss = mean((z2 - Y)^2) ---
        var loss = Scalar[dtype](0)
        for i in range(N):
            var d = z2[i] - Y[i]
            loss += d * d
        loss = loss / Scalar[dtype](N)

        # --- backward (each gradient by hand) ---
        for i in range(N):
            dz2[i] = two_over_N * (z2[i] - Y[i])        # dLoss/dz2
        matmul_at_b(a1, dz2, dW2, N, H, 1)              # dW2 = a1^T @ dz2   (H,1)
        var s = Scalar[dtype](0)
        for i in range(N):
            s += dz2[i]
        db2[0] = s                                      # db2 = sum(dz2)
        matmul_a_bt(dz2, W2, da1, N, 1, H)              # da1 = dz2 @ W2^T   (N,H)
        for i in range(N):
            for j in range(H):
                # relu gradient: pass through only where the pre-activation was positive
                dz1[i * H + j] = da1[i * H + j] if z1[i * H + j] > 0 else Scalar[dtype](0)
        matmul_at_b(X, dz1, dW1, N, D, H)               # dW1 = X^T @ dz1    (D,H)
        for j in range(H):
            var c = Scalar[dtype](0)
            for i in range(N):
                c += dz1[i * H + j]
            db1[j] = c                                  # db1 = colsum(dz1)

        # --- SGD update ---
        for k in range(D * H): W1[k] -= LR * dW1[k]
        for k in range(H):     b1[k] -= LR * db1[k]
        for k in range(H):     W2[k] -= LR * dW2[k]
        b2[0] -= LR * db2[0]

        if epoch % 50 == 0 or epoch == 1:
            print("  epoch", epoch, "  loss", loss)

    var t1 = perf_counter_ns()
    print("\n  train time:", Float64(t1 - t0) / Float64(EPOCHS - 1) / 1.0e6, "ms/epoch (excl. warmup)")

    var final = Scalar[dtype](0)
    for i in range(N):
        var d = z2[i] - Y[i]
        final += d * d
    print("\nfinal loss:", final / Scalar[dtype](N))
