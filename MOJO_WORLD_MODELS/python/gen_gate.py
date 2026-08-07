"""M3 gate — Mojo's self-generated dataset must match a Python reference.

M1 kept the RNG in Python and shipped specs across the boundary; that does not
scale to 10^5 episodes, so mojo/gen.mojo now draws maps and actions itself.
This file is what keeps that trustworthy: it reimplements every draw in the
same call order, runs the VENDORED sim (still the parity oracle), and compares
the resulting state vectors against Mojo's binary output element by element.

Same idea as MOJO_CURRICULUM's train_mlp_reference.py — a shared LCG so both
languages start from bit-identical inputs.

    python python/gen_gate.py --episodes 60 --turns 100 --grid 12 --seed 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import episode as ep
from gridworld import GridWorld

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "gen"
MOD = 1 << 31


class LCG:
    """Must match mojo/gen.mojo::LCG exactly, including the seed reduction."""

    def __init__(self, seed: int):
        s = seed % MOD
        self.s = s if s != 0 else 1

    def next(self) -> int:
        self.s = (1103515245 * self.s + 12345) % MOD
        return self.s

    def below(self, n: int) -> int:
        return self.next() % n

    def pick(self, w: list[int], total: int) -> int:
        r = self.below(total)
        c = 0
        for i, wi in enumerate(w):
            c += wi
            if r < c:
                return i
        return len(w) - 1


def arm_weights(arm: int) -> list[int]:
    if arm == 3:
        return [1, 8, 8, 8, 8, 10, 1, 1, 1, 2, 2, 2, 2]
    if arm == 2:
        return [1, 4, 4, 4, 4, 8, 1, 4, 1, 6, 6, 6, 6]
    return [1, 6, 6, 6, 6, 4, 1, 2, 1, 3, 3, 3, 3]


class GenGrid(GridWorld):
    """Vendored sim, but spawned from the LCG and emitting state vectors."""

    def __init__(self, cfg, lcg: LCG, arm: int):
        self.lcg = lcg
        self.arm = arm
        self.states: list[int] = []
        self._idx = None
        super().__init__(cfg, None)

    def _spawn(self, s: dict):
        n = self.n
        for x in range(n):
            self.walls.add((x, 0)); self.walls.add((x, n - 1))
            self.walls.add((0, x)); self.walls.add((n - 1, x))
        mid = n // 2
        for dy in range(-1, 2):
            self.walls.add((mid, mid + dy))

        occ = set(self.walls)
        n_ag = s["hiders"] + s["seekers"]
        n_ob = s["num_boxes"] + s["num_ramps"]
        cells = []
        for _ in range(n_ag + n_ob):
            while True:
                x = 1 + self.lcg.below(n - 2)
                y = 1 + self.lcg.below(n - 2)
                if (x, y) not in occ:
                    occ.add((x, y))
                    cells.append([x, y])
                    break

        from gridworld import Agent, Obj
        for i in range(s["hiders"]):
            self.agents.append(Agent(f"h{i}", "hider", cells[i]))
        for i in range(s["seekers"]):
            self.agents.append(
                Agent(f"s{i}", "seeker", cells[s["hiders"] + i]))
        for i in range(s["num_boxes"]):
            self.objs.append(Obj(f"box{i}", cells[n_ag + i], "box"))
        for i in range(s["num_ramps"]):
            self.objs.append(
                Obj(f"ramp{i}", cells[n_ag + s["num_boxes"] + i], "ramp"))

        # arm B: pre-locked corner seal (draws happen AFTER spawn, as in Mojo)
        if self.arm == 1 and s["num_boxes"] >= 3:
            corner = self.lcg.below(4)
            cx = 1 if (corner & 1) == 0 else n - 2
            cy = 1 if (corner & 2) == 0 else n - 2
            ix = 1 if cx == 1 else -1
            iy = 1 if cy == 1 else -1
            sx = [cx + ix, cx, cx + ix]
            sy = [cy, cy + iy, cy + iy]
            k = 1 + self.lcg.below(3)
            for b in range(k):
                self.objs[b].pos = [sx[b], sy[b]]
                self.objs[b].locked = True
                self.objs[b].locked_by = "hider"
            self.agents[0].pos = [cx, cy]
            if k < s["num_boxes"] and self.lcg.below(2) == 0:
                self.agents[0].carry = self.objs[k].id

        # arm D: park box 0 beside the hider so `grab` can succeed. Fixed
        # direction order, no LCG draw — mirrors mojo/gen.mojo exactly.
        if self.arm == 3:
            hx, hy = self.agents[0].pos
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if self._placeable([hx + dx, hy + dy], ignore_id="box0"):
                    self.objs[0].pos = [hx + dx, hy + dy]
                    break

    def emit_frame(self):
        if self._idx is None:
            self._idx = {a.id: i for i, a in enumerate(self.agents)}
            self._oidx = {o.id: i for i, o in enumerate(self.objs)}
        seen = self.seen_map()
        st = self.states
        st.append(self.turn)
        st.append(0 if self.turn < self.prep_turns else 1)
        for a in self.agents:
            sb = -1
            if a.team == "hider":
                s, by = seen.get(a.id, (False, None))
                if s:
                    sb = self._idx[by]
            st.append(a.pos[0]); st.append(a.pos[1])
            st.append(a.face[0]); st.append(a.face[1])
            st.append(self._oidx[a.carry] if a.carry else -1)
            st.append(sb)
        for o in self.objs:
            st.append(o.pos[0]); st.append(o.pos[1]); st.append(int(o.locked))


def gen_episode(grid: int, turns: int, seed: int, arm: int):
    cfg = ep.make_cfg(grid=grid, turns=turns)
    lcg = LCG(seed)
    w = GenGrid(cfg, lcg, arm)
    wt = arm_weights(arm)
    total = sum(wt)
    codes = [[lcg.pick(wt, total) for _ in range(len(w.agents))]
             for _ in range(turns)]
    for tc in codes:
        w.resolve_turn({a.id: ep.decode(c, a.pos)
                        for a, c in zip(w.agents, tc)})
    return w.states, [c for tc in codes for c in tc]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    # Equal split across the three arms, matching how gen.mojo assigns them.
    per = a.episodes // 4
    counts = [per, per, per, a.episodes - 3 * per]

    r = subprocess.run(["uv", "run", "mojo", "build", "mojo/gen.mojo",
                        "-o", str(BIN)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("mojo build failed:\n" + r.stdout + r.stderr)

    out = ROOT / "data" / "_gate"
    r = subprocess.run([str(BIN), str(a.grid), str(a.turns), str(a.seed),
                        str(out), str(counts[0]), str(counts[1]),
                        str(counts[2]), str(counts[3])],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("gen failed:\n" + r.stdout + r.stderr)

    n_agents, n_objs = 2, 4
    state_dim = 2 + n_agents * 6 + n_objs * 3
    got_s = np.fromfile(str(out) + ".states.bin", dtype="<i2")
    got_a = np.fromfile(str(out) + ".actions.bin", dtype="<i2")
    got_s = got_s.reshape(a.episodes, a.turns + 1, state_dim)
    got_a = got_a.reshape(a.episodes, a.turns, n_agents)

    bad = []
    for idx in range(a.episodes):
        cum = [counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2]]
        arm = 0 if idx < cum[0] else (1 if idx < cum[1] else (2 if idx < cum[2] else 3))
        st, cd = gen_episode(a.grid, a.turns, a.seed + idx * 7919, arm)
        want_s = np.array(st, dtype="<i2").reshape(a.turns + 1, state_dim)
        want_a = np.array(cd, dtype="<i2").reshape(a.turns, n_agents)
        if not np.array_equal(want_s, got_s[idx]):
            f = int(np.argmax((want_s != got_s[idx]).any(axis=1)))
            bad.append("ep {} (arm {}) frame {}\n  py  {}\n  mjo {}".format(
                idx, arm, f, want_s[f].tolist(), got_s[idx][f].tolist()))
        elif not np.array_equal(want_a, got_a[idx]):
            bad.append("ep {} (arm {}) actions differ".format(idx, arm))

    print("=" * 62)
    print("M3 GATE  {} episodes x {} turns, grid {}, arms {}".format(
        a.episodes, a.turns, a.grid, counts))
    print("=" * 62)
    for m in bad[:5]:
        print("FAIL " + m)
    print("{}/{} episodes identical  ({:,} state vectors)".format(
        a.episodes - len(bad), a.episodes, got_s.shape[0] * got_s.shape[1]))
    if bad:
        sys.exit("\nGATE FAILED")
    print("\n✅ M3 GATE PASSED — Mojo's self-generated data matches Python.")


if __name__ == "__main__":
    main()
