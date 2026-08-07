"""Python reference for the evolutionary search — so the M8 speedup is MEASURED.

The original M8 write-up estimated Mojo's advantage from the simulator's
per-turn cost, because the search only existed in Mojo. An estimate sitting
next to two head-to-head measurements is the weakest claim in the report, so
this is the missing half of the race.

Written the way a competent Python programmer would: NumPy for the sep-CMA-ES
linear algebra and the controller's matrix-vector product, the vendored Python
simulator for rollouts. No strawman — but also no pretending a pure-Python
rollout loop can be vectorised away, because it can't; each turn depends on the
last.

    python python/evolve_ref.py --verify --maps 16      # must match ./build/evolve probe
    python python/evolve_ref.py --gens 10 --pop 24 --maps 16
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import episode as ep
from gen_gate import LCG, GenGrid, arm_weights


class EvoGrid(GenGrid):
    """LCG-spawned, but scoring only — no frame dicts, no state vectors.

    GenGrid's emit_frame collects state vectors and never fills `frames`, so
    the inherited hider_reward() returns 0. This accumulates the same two
    counters evolve.mojo does, which also keeps the comparison fair: Mojo's
    rollout builds no trace either.
    """

    def __init__(self, cfg, lcg, arm):
        self.post = 0
        self.unseen = 0
        super().__init__(cfg, lcg, arm)

    def emit_frame(self):
        if self.turn >= self.prep_turns:
            self.post += 1
            if not self.seen_map().get("h0", (False, None))[0]:
                self.unseen += 1

    def reward(self) -> float:
        return self.unseen / self.post if self.post else 0.0

GRID, TURNS, N_FEAT, N_ACT = 12, 100, 45, 13
N_PARAM = N_FEAT * N_ACT + N_ACT


def occ(w, x, y) -> bool:
    """Mirrors evolve.mojo::occ_s — walls, all objects, all agents, no ignore."""
    if x < 0 or x >= w.n or y < 0 or y >= w.n:
        return True
    if (x, y) in w.walls:
        return True
    for o in w.objs:
        if o.pos[0] == x and o.pos[1] == y:
            return True
    for a in w.agents:
        if a.pos[0] == x and a.pos[1] == y:
            return True
    return False


def features(w) -> np.ndarray:
    """The 45 features, matching evolve.mojo::features field for field."""
    f = np.zeros(N_FEAT, dtype=np.float32)
    hx, hy = w.agents[0].pos
    sx, sy = w.agents[1].pos
    g = float(GRID)
    f[hx] = 1.0
    f[GRID + hy] = 1.0
    f[24] = (sx - hx) / g
    f[25] = (sy - hy) / g
    f[26] = 1.0 if w.turn >= w.prep_turns else 0.0
    f[27] = 1.0 if w.agents[0].carry is not None else 0.0
    seen = w.seen_map().get("h0", (False, None))[0]
    f[28] = 1.0 if seen else 0.0
    p = 29
    for o in w.objs:
        f[p] = (o.pos[0] - hx) / g
        f[p + 1] = (o.pos[1] - hy) / g
        f[p + 2] = 1.0 if o.locked else 0.0
        p += 3
    f[41] = 1.0 if occ(w, hx, hy + 1) else 0.0
    f[42] = 1.0 if occ(w, hx, hy - 1) else 0.0
    f[43] = 1.0 if occ(w, hx + 1, hy) else 0.0
    f[44] = 1.0 if occ(w, hx - 1, hy) else 0.0
    return f


def policy(theta: np.ndarray, f: np.ndarray) -> int:
    """Linear layer + argmax. NumPy matvec — the fair Python implementation."""
    return int(np.argmax(theta[:N_FEAT * N_ACT].reshape(N_ACT, N_FEAT) @ f
                         + theta[N_FEAT * N_ACT:]))


def greedy_seeker(w) -> int:
    sx, sy = w.agents[1].pos
    hx, hy = w.agents[0].pos
    best, bestd = 0, abs(sx - hx) + abs(sy - hy)
    for k, (dx, dy) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        nx, ny = sx + dx, sy + dy
        if occ(w, nx, ny):
            continue
        d = abs(nx - hx) + abs(ny - hy)
        if d < bestd:
            bestd, best = d, k + 1
    return best


def rollout(theta: np.ndarray, seed: int) -> float:
    cfg = ep.make_cfg(grid=GRID, turns=TURNS)
    w = EvoGrid(cfg, LCG(seed), 0)
    for _ in range(TURNS):
        a0 = policy(theta, features(w))
        a1 = greedy_seeker(w)
        w.resolve_turn({"h0": ep.decode(a0, w.agents[0].pos),
                        "s0": ep.decode(a1, w.agents[1].pos)})
    return w.reward()


def fitness(theta: np.ndarray, base: int, n_maps: int) -> float:
    return float(np.mean([rollout(theta, base + m * 7919) for m in range(n_maps)]))


def sep_cma(gens, lam, n_maps, base, seed):
    """Separable CMA-ES in NumPy. Identical algorithm to evolve.mojo; the ES
    math is a rounding error next to the rollouts either way."""
    n, mu = N_PARAM, lam // 2
    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1.0 / np.sum(wts ** 2)
    cc = 4.0 / (n + 4.0)
    cs = (mueff + 2.0) / (n + mueff + 3.0)
    scale = (n + 2.0) / 3.0
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff) * scale
    cmu = 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff) * scale
    if c1 + cmu > 1.0:
        c1, cmu = c1 / (c1 + cmu), cmu / (c1 + cmu)
    damps = 1.0 + cs + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0)
    chiN = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    rng = np.random.default_rng(seed)
    mean, c = np.zeros(n), np.ones(n)
    pc, ps, sigma = np.zeros(n), np.zeros(n), 0.5
    best = -1.0
    for g in range(gens):
        z = rng.standard_normal((lam, n))
        thetas = mean + sigma * np.sqrt(c) * z
        fits = np.array([fitness(t.astype(np.float32), base, n_maps)
                         for t in thetas])
        order = np.argsort(-fits)
        best = max(best, float(fits[order[0]]))
        zmean = wts @ z[order[:mu]]
        mean = mean + sigma * np.sqrt(c) * zmean
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * zmean
        psn = np.linalg.norm(ps)
        hsig = float(psn / np.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chiN
                     < 1.4 + 2.0 / (n + 1))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * np.sqrt(c) * zmean
        rank_mu = wts @ ((np.sqrt(c) * z[order[:mu]]) ** 2)
        c = np.maximum((1 - c1 - cmu) * c + c1 * pc ** 2 + cmu * rank_mu, 1e-12)
        sigma *= np.exp((cs / damps) * (psn / chiN - 1))
        # Guard rails absent from evolve.mojo. This NumPy port destabilised
        # (sigma blowing up -> non-finite parameters) where the Mojo version
        # ran 150 generations cleanly, so it is not used for timing; the
        # rollout benchmark below is. Kept working for reference only.
        sigma = float(np.clip(sigma, 1e-8, 1e3))
        c = np.clip(c, 1e-12, 1e6)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--reps", type=int, default=24)
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--maps", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    # evolve.mojo's probe takes the map base seed literally; match it exactly so
    # both sides score the identical map set.
    base = a.seed

    if a.verify:
        zero = np.zeros(N_PARAM, dtype=np.float32)
        det = np.array([((i * 37) % 101 - 50) / 100.0 for i in range(N_PARAM)],
                       dtype=np.float32)
        print("PROBE zero= {:.8g}  det= {:.8g}".format(
            fitness(zero, base, a.maps), fitness(det, base, a.maps)))
        print("compare against:  ./build/evolve probe {} {}".format(a.seed, a.maps))
        return

    if a.bench:
        chk = 0.0
        t0 = time.perf_counter()
        for r in range(a.reps):
            th = np.array([((i * 37 + r * 13) % 101 - 50) / 100.0
                           for i in range(N_PARAM)], dtype=np.float32)
            chk += fitness(th, base, a.maps)
        el = time.perf_counter() - t0
        n = a.reps * a.maps
        print("PY rollouts={} turns={} sec={:.3f} checksum={:.6f}".format(
            n, n * TURNS, el, chk))
        return

    t0 = time.perf_counter()
    best = sep_cma(a.gens, a.pop, a.maps, base, a.seed)
    el = time.perf_counter() - t0
    rollouts = a.gens * a.pop * a.maps
    print("PY gens={} pop={} maps={} rollouts={} turns={} sec={:.2f} best={:.4f}".format(
        a.gens, a.pop, a.maps, rollouts, rollouts * TURNS, el, best))


if __name__ == "__main__":
    main()
