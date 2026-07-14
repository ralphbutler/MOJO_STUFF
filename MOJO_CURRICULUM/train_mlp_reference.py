"""train_mlp_reference.py — NumPy ground truth for 04_train_mlp.mojo.

A 2-layer MLP (2 -> H -> 1) trained by HAND-WRITTEN backprop + SGD to fit the radius
r = sqrt(x0^2 + x1^2). No autograd, no framework — every gradient is derived and coded
explicitly. This file exists so 04_train_mlp.mojo can reproduce the exact same numbers and
prove its by-hand Mojo backprop is correct.

Data and initial weights come from a tiny LCG (below) that 04_train_mlp.mojo implements
identically, so both programs start from bit-identical inputs. Their loss curves should
track closely; small drift late in training is expected (float32 summation order differs
between NumPy's BLAS and Mojo's naive loops).

Run: python train_mlp_reference.py
"""

import numpy as np

# ---- shared config — 04_train_mlp.mojo mirrors every value ----
N, D, H = 256, 2, 16
EPOCHS, LR = 300, np.float32(0.1)

# ---- shared 32-bit LCG -> float in [0,1). Same constants/modulus as the Mojo version.
# Call ORDER matters: X (row-major N×D), then W1 (D×H), then W2 (H×1). Mojo matches this.
_state = 1
def rand01() -> float:
    global _state
    _state = (1103515245 * _state + 12345) % (1 << 31)
    return _state / float(1 << 31)

def fill(n: int) -> np.ndarray:
    return np.array([rand01() for _ in range(n)], dtype=np.float32)

X = (4.0 * fill(N * D) - 2.0).reshape(N, D).astype(np.float32)          # points in [-2,2]
Y = np.sqrt((X ** 2).sum(1, keepdims=True)).astype(np.float32)         # radius, (N,1)
W1 = (fill(D * H) - 0.5).reshape(D, H).astype(np.float32)              # (2,H)
b1 = np.zeros((1, H), dtype=np.float32)
W2 = (fill(H * 1) - 0.5).reshape(H, 1).astype(np.float32)              # (H,1)
b2 = np.zeros((1, 1), dtype=np.float32)

two_over_N = np.float32(2.0 / N)
print(f"N={N}, hidden={H}, epochs={EPOCHS}, lr={float(LR)}  (NumPy reference)\n")

for epoch in range(1, EPOCHS + 1):
    # --- forward ---
    z1 = (X @ W1 + b1).astype(np.float32)        # (N,H)
    a1 = np.maximum(z1, np.float32(0.0))         # relu
    z2 = (a1 @ W2 + b2).astype(np.float32)       # (N,1)
    diff = (z2 - Y).astype(np.float32)           # (N,1)
    loss = float((diff ** 2).mean())

    # --- backward (every gradient by hand) ---
    dz2 = two_over_N * diff                      # dLoss/dz2   (N,1)
    dW2 = (a1.T @ dz2).astype(np.float32)        # (H,1)
    db2 = dz2.sum(0, keepdims=True)              # (1,1)
    da1 = (dz2 @ W2.T).astype(np.float32)        # (N,H)
    dz1 = da1 * (z1 > 0)                          # relu gradient (N,H)
    dW1 = (X.T @ dz1).astype(np.float32)         # (D,H)
    db1 = dz1.sum(0, keepdims=True)              # (1,H)

    # --- SGD update ---
    W1 -= LR * dW1; b1 -= LR * db1
    W2 -= LR * dW2; b2 -= LR * db2

    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:4d}   loss {loss:.6f}")

print(f"\nfinal loss: {loss:.6f}")
