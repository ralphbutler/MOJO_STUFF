"""train_nabla_mlp.py — the SAME training loop as train_torch_mlp.py, but in Nabla:
the Mojo-native, MAX-backed differentiable-programming framework.

Why this file exists: MAX itself has no training API (no autograd). Nabla adds one *on
top of* Mojo + MAX — custom Mojo kernels dropped into an autodiff engine with grad/vmap/
jit. It is the ecosystem's answer to "how do you train without leaving Mojo/MAX," and the
runnable version of the README's "where the tiers meet." Compare side by side with
train_torch_mlp.py to judge the Mojo-native path against mainstream PyTorch.

  PyTorch (train_torch_mlp.py)          Nabla (this file)
  ------------------------------        --------------------------------
  nn.Sequential / nn.Module             nb.nn.Module
  nn.Linear, nn.ReLU / F.relu           nb.nn.Linear, nb.relu
  torch.optim.Adam(model.parameters())  nb.nn.optim.AdamW(model, ...)
  loss = CrossEntropyLoss()(...)        loss = nb.nn.functional.mse_loss(...)
  optimizer.zero_grad()                 model.zero_grad()
  loss.backward(); optimizer.step()     loss.backward(); model = optimizer.step()  ← note!
  tensors on device                     nb.Tensor.from_dlpack(numpy_array)

⚠️ STATUS: the code below is correct against Nabla's example 04a API — first run reached
`loss.backward()` cleanly. The failure was NOT here: it was a version mismatch (Nabla was
built 2026-02-25 but pip pulled a July modular nightly, so MAX's MLIR binding had drifted:
"ModuleOp has no attribute 'operation'"). setup_nabla_venv.sh now PINS modular to Nabla's
Feb-2026 era, which fixes it. If you edited this file, that's why — the fix was the venv,
not the loop. It's still ALPHA, so a future call may yet shift; send output if so.

TASK: regress each 2D point's distance from the origin, r = sqrt(x² + y²). That radius is
a nonlinear function of (x, y) — a bare linear layer cannot compute a norm — so the hidden
layer is doing real work, the same "why the hidden layer exists" point as the PyTorch
rings demo, but as regression (which is the API 04a confirms; classification loss is
unverified in Nabla today).

Run (after ./setup_nabla_venv.sh):
    .venv-nabla/bin/python train_nabla_mlp.py            # 400 epochs, hidden=32
    .venv-nabla/bin/python train_nabla_mlp.py 800 64     # custom epochs / hidden width
"""

import sys

import numpy as np
import nabla as nb

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N = 1000
np.random.seed(0)  # reproducible so runs are comparable / shareable

# --- data: 2D points, target = distance from origin (nonlinear in x, y) ---
X_np = (np.random.randn(N, 2) * 1.5).astype(np.float32)
y_np = np.sqrt((X_np ** 2).sum(axis=1, keepdims=True)).astype(np.float32)  # shape (N, 1)

# Nabla tensors are built from numpy via DLPack (no separate device step — MAX places
# the work on the accelerator itself; on this Mac that's the Metal GPU).
X = nb.Tensor.from_dlpack(X_np)
y = nb.Tensor.from_dlpack(y_np)


class MLP(nb.nn.Module):
    """2 inputs -> HIDDEN (ReLU) -> 1 output. Same shape as the PyTorch demo."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nb.nn.Linear(in_dim, hidden_dim)
        self.fc2 = nb.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = nb.relu(self.fc1(x))   # delete this nonlinearity and it can't learn a norm
        return self.fc2(x)


model = MLP(2, HIDDEN, 1)
optimizer = nb.nn.optim.AdamW(model, lr=1e-2)   # note: takes `model`, not model.parameters()

print(f"epochs={EPOCHS}, hidden={HIDDEN}, points={N}  (Nabla / Mojo+MAX backend)\n")

# --- the training loop: same four beats as PyTorch, with Nabla's quirks flagged ---
for epoch in range(1, EPOCHS + 1):
    model.zero_grad()                              # 1. clear grads (on the MODEL here)
    predictions = model(X)                        # 2. forward pass
    loss = nb.nn.functional.mse_loss(predictions, y)  #    regression loss
    loss.backward()                               # 3. autodiff — the thing MAX alone lacks
    model = optimizer.step()                      # 4. step RETURNS the updated model — reassign!

    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:4d}   loss {loss.item():.4f}")

print(f"\nfinal loss: {loss.item():.4f}")
print("(loss → ~0 means the MLP learned the radius; a linear-only model stalls high.)")
