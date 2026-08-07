"""M4 — train the dynamics model and measure held-out next-state accuracy.

The metric that matters is EXACT MATCH: every one of the 24 heads right, so the
predicted next state is the true next state cell-for-cell. Per-field accuracy
is reported too, because it says *where* a model fails, but a dream that gets
23 of 24 fields right every step still diverges.

Split is by EPISODE, and stratified across the three generation arms. A
transition-level split would leak: consecutive frames from one episode are
near-identical, so a random split would put a frame's neighbours in train and
score memorisation as generalisation. Arms are contiguous in the file, so a
plain tail split would hold out only build-biased episodes.

    python python/train.py --steps 8000
    python python/train.py --steps 4000 --episodes 2000 --quiet   # scaling point
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from dataset import load
from model import DynamicsMLP, Spec, device, export_weights

VAL_FRAC = 0.05


def split_episodes(meta: dict, use_episodes: int | None):
    """Stratified by arm: hold out the last VAL_FRAC of each arm separately."""
    train, val, off = [], [], 0
    for _, count in meta["arm_counts"].items():
        idx = np.arange(off, off + count)
        n_val = max(1, int(count * VAL_FRAC))
        val.append(idx[-n_val:])
        tr = idx[:-n_val]
        if use_episodes is not None:
            share = max(1, int(use_episodes * count / meta["episodes"]))
            tr = tr[:share]
        train.append(tr)
        off += count
    return np.concatenate(train), np.concatenate(val)


class Batcher:
    def __init__(self, s, a, eps, turns, dev, seed=0):
        self.s, self.a, self.eps, self.turns, self.dev = s, a, eps, turns, dev
        self.rng = np.random.default_rng(seed)

    def sample(self, n):
        e = self.eps[self.rng.integers(0, len(self.eps), n)]
        t = self.rng.integers(0, self.turns, n)
        return self._pull(e, t)

    def all_pairs(self):
        e = np.repeat(self.eps, self.turns)
        t = np.tile(np.arange(self.turns), len(self.eps))
        return e, t

    def _pull(self, e, t):
        s0 = torch.from_numpy(self.s[e, t].astype(np.int64)).to(self.dev)
        s1 = torch.from_numpy(self.s[e, t + 1].astype(np.int64)).to(self.dev)
        ac = torch.from_numpy(self.a[e, t].astype(np.int64)).to(self.dev)
        return s0, ac, s1


@torch.no_grad()
def evaluate(model, spec, b: Batcher, max_n=400_000, chunk=16384):
    model.eval()
    e, t = b.all_pairs()
    if len(e) > max_n:                       # deterministic subsample
        sel = np.linspace(0, len(e) - 1, max_n).astype(np.int64)
        e, t = e[sel], t[sel]
    heads = np.zeros(spec.n_heads, dtype=np.int64)
    exact = 0
    # Subsets worth separating: the thin ones from RESULTS_M3.md.
    sub = {"carrying": [0, 0], "sealed_arm": [0, 0], "seen": [0, 0]}
    for i in range(0, len(e), chunk):
        s0, ac, s1 = b._pull(e[i:i + chunk], t[i:i + chunk])
        c = model.correct(model(spec.encode(s0, ac)), spec.targets(s1))
        heads += c.sum(dim=0).cpu().numpy()
        ex = c.all(dim=1)
        exact += int(ex.sum())
        m = s0[:, 6] >= 0                     # hider carry column
        sub["carrying"][0] += int(ex[m].sum()); sub["carrying"][1] += int(m.sum())
        m = s0[:, 7] >= 0                     # hider seen_by column
        sub["seen"][0] += int(ex[m].sum()); sub["seen"][1] += int(m.sum())
        m = (s0[:, 16] + s0[:, 19] + s0[:, 22]) >= 2   # >=2 boxes locked
        sub["sealed_arm"][0] += int(ex[m].sum()); sub["sealed_arm"][1] += int(m.sum())
    model.train()
    n = len(e)
    return {
        "n": n,
        "exact": exact / n,
        "heads": {name: heads[i] / n for i, (name, _, _, _) in enumerate(spec.fields)},
        "subsets": {k: (v[0] / v[1] if v[1] else float("nan"), v[1])
                    for k, v in sub.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../data/train")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--episodes", type=int, default=None,
                   help="cap training episodes (for the scaling curve)")
    p.add_argument("--out", default="../data/dynamics")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    s, ac, meta = load(a.data)
    s = np.asarray(s)                        # 525 MB, fits comfortably
    ac = np.asarray(ac)
    spec = Spec(meta)
    dev = device()

    tr_eps, va_eps = split_episodes(meta, a.episodes)
    turns = meta["turns"]
    n_train = len(tr_eps) * turns

    model = DynamicsMLP(spec, a.hidden, a.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    tb = Batcher(s, ac, tr_eps, turns, dev, seed=0)
    vb = Batcher(s, ac, va_eps, turns, dev, seed=1)

    if not a.quiet:
        print("device {}  in_dim {}  heads {}  out_dim {}  params {:,}".format(
            dev, spec.in_dim, spec.n_heads, spec.out_dim,
            sum(p.numel() for p in model.parameters())))
        print("train {:,} transitions ({:,} eps)   val {:,} ({:,} eps)".format(
            n_train, len(tr_eps), len(va_eps) * turns, len(va_eps)))

    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        s0, act, s1 = tb.sample(a.batch)
        loss = model.loss(model(spec.encode(s0, act)), spec.targets(s1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if not a.quiet and (step % 1000 == 0 or step == 1):
            print("  step {:>6d}  loss {:8.4f}  {:5.1f}s".format(
                step, float(loss), time.perf_counter() - t0))

    r = evaluate(model, spec, vb)
    train_time = time.perf_counter() - t0

    if a.quiet:
        print("SCALE episodes={} transitions={} exact={:.6f}".format(
            len(tr_eps), n_train, r["exact"]))
        return

    print("\n" + "=" * 68)
    print("M4 — HELD-OUT NEXT-STATE ACCURACY   ({:,} val transitions)".format(r["n"]))
    print("=" * 68)
    print("  EXACT MATCH (all {} fields correct)   {:.4f}%".format(
        spec.n_heads, 100 * r["exact"]))
    print("\n  per-field accuracy (worst first)")
    for name, acc in sorted(r["heads"].items(), key=lambda kv: kv[1])[:8]:
        print("    {:<14s} {:.4f}%".format(name, 100 * acc))
    print("\n  exact match on hard subsets")
    for k, (acc, n) in r["subsets"].items():
        print("    {:<14s} {:.4f}%   ({:,} transitions)".format(k, 100 * acc, n))
    print("\n  trained {:,} steps x batch {} in {:.1f}s on {}".format(
        a.steps, a.batch, train_time, dev))

    export_weights(model, a.out, meta)
    torch.save(model.state_dict(), a.out + ".pt")
    print("  exported -> {}.weights.bin / .weights.json / .pt".format(a.out))


if __name__ == "__main__":
    main()
