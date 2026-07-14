"""train_torch_mlp.py — the TRAINING tier, on the Apple GPU via PyTorch/MPS.

Where this sits in the three tools this curriculum covers:
  • Mojo   — you WRITE the GPU kernel          (00–03: vecadd, matmul).
  • MAX    — you RUN a prebuilt model          (max_serve_litellm.sh, max_generate.sh).
  • PyTorch— you TRAIN a model                 (this file, + bench_torch.py).

MAX deliberately has no training API — no autograd, no optimizers, no `.backward()`.
It is an inference/serving engine. When you need to *learn weights*, you reach for a
training framework. That is the point of this demo: the piece MAX does not do.

The task is deliberately tiny but genuinely nonlinear: two concentric rings of points
(inner ring = class 0, outer ring = class 1). No straight line separates them, so a
bare linear classifier is stuck near 50%. One hidden layer with a nonlinearity learns
the circular boundary — that's *why* the hidden layer exists.

Run from an env that has torch (your BASE env), NOT the mojo-learning venv
(MOJO_CURRICULUM/.venv holds MAX, not torch):

    python train_torch_mlp.py            # 400 epochs, hidden=32
    python train_torch_mlp.py 800 64     # custom epochs / hidden width
"""

import sys
import math

import torch
from torch import nn

# --- device: MPS on this Mac, else CUDA, else CPU (matches the house pattern) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N_PER_CLASS = 500
torch.manual_seed(0)  # reproducible so runs are comparable / shareable


def make_rings() -> tuple[torch.Tensor, torch.Tensor]:
    """Two concentric noisy rings. Returns X (2N, 2) points and y (2N,) labels."""
    def ring(radius: float, label: int) -> tuple[torch.Tensor, torch.Tensor]:
        angle = torch.rand(N_PER_CLASS) * (2 * math.pi)
        r = radius + torch.randn(N_PER_CLASS) * 0.15          # radial jitter
        pts = torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1)
        return pts, torch.full((N_PER_CLASS,), label, dtype=torch.long)

    x0, y0 = ring(1.0, 0)   # inner ring
    x1, y1 = ring(2.5, 1)   # outer ring
    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    return X, y


# Build the data once and move it to the GPU. This set is small, so we train on the
# whole thing every step (full-batch) — no DataLoader needed to keep the demo readable.
X, y = make_rings()
X, y = X.to(device), y.to(device)

# The model: 2 inputs -> HIDDEN (nonlinear) -> 2 class logits.
# Delete the ReLU (or the hidden layer) and this collapses to a linear classifier
# that cannot separate concentric rings — accuracy stalls near 50%. That collapse
# is the lesson; the nonlinearity is what buys the curved decision boundary.
model = nn.Sequential(
    nn.Linear(2, HIDDEN),
    nn.ReLU(),
    nn.Linear(HIDDEN, 2),
).to(device)

loss_fn = nn.CrossEntropyLoss()                       # expects int64 class labels
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"device={device}, epochs={EPOCHS}, hidden={HIDDEN}, "
      f"points={2 * N_PER_CLASS}, params={sum(p.numel() for p in model.parameters())}\n")

# --- the training loop: the four canonical steps, one per line ---
for epoch in range(1, EPOCHS + 1):
    optimizer.zero_grad()          # 1. clear gradients left from the last step
    logits = model(X)             # 2. forward pass  -> predictions
    loss = loss_fn(logits, y)     #    compare predictions to truth
    loss.backward()               # 3. autograd fills every .grad (this is what MAX lacks)
    optimizer.step()              # 4. nudge weights down the gradient

    if epoch % 50 == 0 or epoch == 1:
        print(f"  epoch {epoch:4d}   loss {loss.item():.4f}")

# --- evaluate: no gradients needed, so turn autograd off ---
with torch.no_grad():
    pred = model(X).argmax(dim=1)
    acc = (pred == y).float().mean().item()

print(f"\nfinal train accuracy: {acc * 100:.1f}%")
print("(≈50% means it failed to learn the rings; a working MLP reaches ~99%.)")
