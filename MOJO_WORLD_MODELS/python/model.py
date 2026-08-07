"""M4 — the dynamics model: (state, joint action) -> next state.

DESIGN CALL: this is multi-head **classification**, not regression. Every field
in the state vector is categorical — grid coordinates on a 12x12 board, facing
in {-1,0,1}, carry as an object index, locked as a flag. Regressing an (x, y)
would let the model emit 7.3 and force a rounding rule to paper over it, and
would put "off by one cell" and "off by six cells" on the same smooth loss
surface. One softmax per field instead, and exact-match accuracy means the
predicted next state IS the true next state, cell for cell.

turn and phase are NOT predicted: turn' = turn + 1 and phase' = turn' >=
prep_turns, both deterministic. Including them would inflate accuracy with
free wins. They stay as model INPUTS because they gate seeker freezing and LOS.

Ha used an MDN-RNN because a mixture-density head handles stochastic
environments. This gridworld is deterministic given the state (PLAN.md
decision 7 made states Markov), so an MLP should be sufficient — but that is
a claim to measure, not assume. See PLAN.md open question 1.
"""
from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 13


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Spec:
    """Field layout derived from the dataset meta — nothing hard-coded."""

    def __init__(self, meta: dict):
        g = meta["grid"]
        self.grid = g
        self.n_agents = meta["n_agents"]
        self.n_objs = meta["n_boxes"] + meta["n_ramps"]
        self.turns = meta["turns"]
        self.prep_turns = meta["prep_turns"]

        # (name, column, n_classes, shift) — shift makes values 0-based.
        f = []
        col = 2                                   # skip turn, phase
        for i in range(self.n_agents):
            f.append((f"a{i}.x", col + 0, g, 0))
            f.append((f"a{i}.y", col + 1, g, 0))
            f.append((f"a{i}.fx", col + 2, 3, 1))
            f.append((f"a{i}.fy", col + 3, 3, 1))
            f.append((f"a{i}.carry", col + 4, self.n_objs + 1, 1))
            f.append((f"a{i}.seen", col + 5, self.n_agents + 1, 1))
            col += 6
        for i in range(self.n_objs):
            f.append((f"o{i}.x", col + 0, g, 0))
            f.append((f"o{i}.y", col + 1, g, 0))
            f.append((f"o{i}.locked", col + 2, 2, 0))
            col += 3
        self.fields = f
        self.state_dim = col
        self.n_heads = len(f)
        self.out_dim = sum(n for _, _, n, _ in f)
        # 2 scalars (turn, phase) + one-hot per field + one-hot per agent action
        self.in_dim = 2 + self.out_dim + self.n_agents * N_ACTIONS

    def encode(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """int64 [B, state_dim] + [B, n_agents] -> float32 [B, in_dim]."""
        parts = [
            (s[:, 0:1].float() / self.turns),     # turn, normalised
            s[:, 1:2].float(),                    # phase, already 0/1
        ]
        for _, col, n, sh in self.fields:
            parts.append(F.one_hot(s[:, col] + sh, n).float())
        for i in range(self.n_agents):
            parts.append(F.one_hot(a[:, i], N_ACTIONS).float())
        return torch.cat(parts, dim=1)

    def targets(self, s_next: torch.Tensor) -> torch.Tensor:
        """int64 [B, state_dim] -> int64 [B, n_heads] of class indices."""
        return torch.stack(
            [s_next[:, col] + sh for _, col, _, sh in self.fields], dim=1)

    def decode(self, logits: torch.Tensor, s: torch.Tensor,
               turn_next: int) -> torch.Tensor:
        """Argmax each head back into a full state vector.

        `turn_next` is passed in, NOT derived as s[:,0]+1. GOTCHA: the sim's
        emit_frame runs BEFORE `self.turn += 1`, so frame k carries turn k-1
        and frames 0 and 1 BOTH have turn 0. Deriving turn by incrementing is
        therefore correct at every step except the first — which is enough to
        make a closed-loop dream diverge at step 0 and never recover. At dream
        step t (0-based) the produced frame is t+1, whose turn is exactly t.
        """
        out = torch.empty_like(s)
        out[:, 0] = turn_next
        out[:, 1] = 1 if turn_next >= self.prep_turns else 0
        off = 0
        for (_, col, n, sh) in self.fields:
            out[:, col] = logits[:, off:off + n].argmax(dim=1) - sh
            off += n
        return out


class DynamicsMLP(nn.Module):
    def __init__(self, spec: Spec, hidden: int = 512, layers: int = 2):
        super().__init__()
        self.spec = spec
        mods, d = [], spec.in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods += [nn.Linear(d, spec.out_dim)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)

    def loss(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Summed cross-entropy over heads. Sum, not mean: every field must be
        right for the dream to be right, so no head gets discounted."""
        total, off = 0.0, 0
        for h, (_, _, n, _) in enumerate(self.spec.fields):
            total = total + F.cross_entropy(logits[:, off:off + n], tgt[:, h])
            off += n
        return total

    def correct(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Bool [B, n_heads] — per-head correctness."""
        preds, off = [], 0
        for _, _, n, _ in self.spec.fields:
            preds.append(logits[:, off:off + n].argmax(dim=1))
            off += n
        return torch.stack(preds, dim=1) == tgt


def export_weights(model: DynamicsMLP, prefix: str, meta: dict) -> None:
    """Flat float32 dump + JSON sidecar, for mojo/dream.mojo (M5).

    Row-major, layer by layer: weight then bias. No parser needed on the Mojo
    side — same principle as the keyword-free .spec files in M1.
    """
    import struct
    blobs, shapes = [], []
    for m in model.net:
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu().numpy().astype(np.float32)
            b = m.bias.detach().cpu().numpy().astype(np.float32)
            blobs += [w.tobytes(), b.tobytes()]
            shapes.append({"in": int(w.shape[1]), "out": int(w.shape[0])})
    with open(prefix + ".weights.bin", "wb") as f:
        for blob in blobs:
            f.write(blob)
    side = {
        "layers": shapes,
        "activation": "relu",
        "in_dim": model.spec.in_dim,
        "out_dim": model.spec.out_dim,
        "state_dim": model.spec.state_dim,
        "n_agents": model.spec.n_agents,
        "n_objs": model.spec.n_objs,
        "grid": model.spec.grid,
        "turns": model.spec.turns,
        "prep_turns": model.spec.prep_turns,
        "n_actions": N_ACTIONS,
        "fields": [{"name": n, "col": c, "classes": k, "shift": s}
                   for n, c, k, s in model.spec.fields],
        "dtype": "<f4",
        "layout": "per Linear: weight [out,in] row-major, then bias [out]",
    }
    with open(prefix + ".weights.json", "w") as f:
        json.dump(side, f, indent=2)
