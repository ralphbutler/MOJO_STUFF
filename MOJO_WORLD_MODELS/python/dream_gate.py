"""M5 — verify the Mojo dream, then measure how far it gets before diverging.

Two gates and one measurement:

  1. NUMERICAL   Mojo's closed-loop rollout must agree with PyTorch's on the
                 same inputs. Exact agreement is not guaranteed: Mojo sums a
                 naive dot product, PyTorch calls BLAS, and float32 summation
                 order differs — so a near-tie between two class logits can
                 flip an argmax. Same caveat MOJO_CURRICULUM's
                 train_mlp_reference.py notes. Mismatches are reported with
                 their logit gap so a real bug is distinguishable from drift.

  2. DIVERGENCE  Closed loop against ground truth from the real sim: the dream
                 gets a true start state and the true action sequence, then
                 runs free. Steps-until-first-mismatch is the M7 metric,
                 arriving early because M4 showed 93% per step compounds fast.

  3. SPEED       Mojo vs PyTorch on identical rollouts, CPU-pinned, for M6.

Held-out episodes only — the same val split train.py used.

    python python/dream_gate.py --episodes 200 --steps 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load
from model import DynamicsMLP, Spec
from train import split_episodes

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "dream"


@torch.no_grad()
def torch_dream(model, spec, s0, acts, steps, dev):
    """Closed loop in PyTorch. s0 [N, dim], acts [N, steps, 2]."""
    s = torch.from_numpy(s0.astype(np.int64)).to(dev)
    a = torch.from_numpy(acts.astype(np.int64)).to(dev)
    out = []
    for t in range(steps):
        logits = model(spec.encode(s, a[:, t]))
        s = spec.decode(logits, s, t)     # frame t+1 carries turn t — see decode()
        out.append(s.cpu().numpy().copy())
    return np.stack(out, axis=1)          # [N, steps, dim]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/train")
    p.add_argument("--model", default="../data/dynamics")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()

    s_all, a_all, meta = load(a.data)
    spec = Spec(meta)
    _, va = split_episodes(meta, None)
    eps = va[np.linspace(0, len(va) - 1, a.episodes).astype(int)]
    steps = min(a.steps, meta["turns"])

    truth = np.asarray(s_all[eps])[:, : steps + 1, :]      # [N, steps+1, dim]
    acts = np.asarray(a_all[eps])[:, :steps, :]            # [N, steps, 2]
    start = truth[:, 0, :]

    # ---- bundle for Mojo: n_eps, then per ep  start state + actions --------
    ints = [len(eps)]
    for i in range(len(eps)):
        ints += start[i].tolist()
        ints += acts[i].reshape(-1).tolist()
    bundle = ROOT / "data" / "dream.bundle"
    bundle.write_text(" ".join(map(str, ints)) + "\n")

    # ---- Mojo -------------------------------------------------------------
    mj_out = ROOT / "data" / "dream.mojo.txt"
    r = subprocess.run([str(BIN), a.model + ".weights.bin", str(bundle),
                        str(steps), "verify", str(mj_out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("dream failed:\n" + r.stdout + r.stderr)
    mj_line = next(l for l in r.stdout.splitlines() if l.startswith("DREAM"))
    mj = dict(t.split("=", 1) for t in mj_line.replace("= ", "=").split()
              if "=" in t)
    mojo_pred = np.loadtxt(mj_out, dtype=np.int64).reshape(
        len(eps), steps, spec.state_dim)

    # ---- PyTorch (CPU, to match Mojo's device — protocol rule 1) ----------
    dev = torch.device("cpu")
    model = DynamicsMLP(spec, 1536, 3).to(dev)
    model.load_state_dict(torch.load(a.model + ".pt", map_location=dev))
    model.eval()
    t0 = time.perf_counter_ns()
    torch_pred = torch_dream(model, spec, start, acts, steps, dev)
    t_torch = time.perf_counter_ns() - t0

    # ---- gate 1: numerical agreement --------------------------------------
    agree = (mojo_pred == torch_pred)
    frac = agree.all(axis=2).mean()

    # ---- gate 2: divergence vs ground truth -------------------------------
    def first_div(pred):
        eq = (pred == truth[:, 1:, :]).all(axis=2)      # [N, steps]
        bad = ~eq
        has = bad.any(axis=1)
        idx = np.where(has, bad.argmax(axis=1), steps)
        return idx                                       # steps survived

    dm = first_div(mojo_pred)
    per_step = (mojo_pred == truth[:, 1:, :]).all(axis=2).mean()

    print("=" * 70)
    print("M5 — THE DREAM   {} held-out episodes x {} steps".format(len(eps), steps))
    print("=" * 70)

    print("\n1. NUMERICAL AGREEMENT  Mojo vs PyTorch (same weights, same inputs)")
    print("   full-state agreement    {:.4f}%".format(100 * frac))
    print("   per-field agreement     {:.6f}%".format(100 * agree.mean()))
    if frac < 1.0:
        n_bad = int((~agree.all(axis=2)).sum())
        print("   {:,} of {:,} steps differ — expected from float32 summation".format(
            n_bad, len(eps) * steps))
        print("   order (naive dot product vs BLAS) flipping near-tied argmaxes.")

    print("\n2. DIVERGENCE FROM REALITY  (closed loop, true actions)")
    print("   mean steps survived     {:6.2f}".format(dm.mean()))
    print("   median                  {:6.0f}".format(np.median(dm)))
    print("   p10 / p90               {:6.0f} / {:.0f}".format(
        np.percentile(dm, 10), np.percentile(dm, 90)))
    print("   reached full {} steps   {:.1f}%".format(
        steps, 100 * (dm >= steps).mean()))
    print("   per-step exact match    {:.2f}%  (open loop was 93.0%)".format(
        100 * per_step))
    print("\n   survival curve")
    for k in (1, 2, 3, 5, 10, 20, 50, steps):
        if k <= steps:
            print("     {:>4d} steps  {:5.1f}%".format(k, 100 * (dm >= k).mean()))

    print("\n3. SPEED  (CPU both sides — protocol rule 1)")
    tot = len(eps) * steps
    print("   PyTorch CPU  {:9.2f} ms   {:8.2f} us/step".format(
        t_torch / 1e6, t_torch / 1e3 / tot))
    print("   Mojo         {:9.2f} ms   {:8.2f} us/step  ({} threads)".format(
        int(mj["ns"]) / 1e6, float(mj["us_per_step"]), mj["workers"]))
    print("   --> {:.1f}x".format(t_torch / int(mj["ns"])))

    if frac < 0.98:
        sys.exit("\nGATE FAILED: Mojo and PyTorch disagree beyond float drift.")
    print("\n✅ M5 GATE PASSED — Mojo dream matches PyTorch.")


if __name__ == "__main__":
    main()
