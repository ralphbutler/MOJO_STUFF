"""M1 parity gate — Mojo must reproduce the Python sim byte-for-byte.

Sweeps seeds, runs both sims on identical maps and action sequences, and diffs
the frame digests. Also reports COVERAGE: parity on code paths that never fired
proves nothing, so the gate fails if a required path went unexercised.

    python python/parity.py --seeds 200 --turns 40

PLAN.md M2 (benchmarking) is blocked on this being green.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import episode as ep
from gridworld import GridWorld

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "gridworld"


def build_mojo() -> None:
    BIN.parent.mkdir(exist_ok=True)
    r = subprocess.run(
        ["uv", "run", "mojo", "build", "mojo/gridworld.mojo", "-o", str(BIN)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit("mojo build failed:\n" + r.stdout + r.stderr)


class Coverage:
    """Did the sweep actually exercise the interesting branches?"""

    def __init__(self):
        self.hit = {k: 0 for k in (
            "move_ok", "move_blocked", "grab_ok", "grab_locked_fail",
            "drop", "lock_in_place", "place_and_lock", "unlock_ok",
            "carry_trail", "seen", "clamped_noop",
        )}

    def note(self, k: str, n: int = 1):
        self.hit[k] += n

    def missing(self) -> list[str]:
        return [k for k, v in self.hit.items() if v == 0]


def observe(w: GridWorld, before: dict, after: dict, code: int, cov: Coverage):
    """Classify what a single turn actually did, for coverage accounting."""
    if code in (ep.MOVE_N, ep.MOVE_S, ep.MOVE_E, ep.MOVE_W):
        if before["pos"] != after["pos"]:
            cov.note("move_ok")
            if before["carry"] is not None:
                cov.note("carry_trail")
        elif before["clamped"]:
            cov.note("clamped_noop")
        else:
            cov.note("move_blocked")
    elif code == ep.GRAB:
        if before["carry"] is None and after["carry"] is not None:
            cov.note("grab_ok")
        elif before["carry"] is None and before["adj_locked"]:
            cov.note("grab_locked_fail")
    elif code == ep.DROP:
        if before["carry"] is not None and after["carry"] is None:
            cov.note("drop")
    elif code == ep.LOCK:
        if after["n_locked"] > before["n_locked"]:
            cov.note("lock_in_place")
    elif code in (ep.LOCK_N, ep.LOCK_S, ep.LOCK_E, ep.LOCK_W):
        if before["carry"] is not None and after["carry"] is None:
            cov.note("place_and_lock")
        elif after["n_locked"] > before["n_locked"]:
            cov.note("lock_in_place")
    elif code == ep.UNLOCK:
        if after["n_locked"] < before["n_locked"]:
            cov.note("unlock_ok")


def snapshot(w: GridWorld, ag, code: int) -> dict:
    adj = w._adjacent_obj(ag)
    clamped = False
    if code in (ep.MOVE_N, ep.MOVE_S, ep.MOVE_E, ep.MOVE_W):
        d = {ep.MOVE_N: (0, 1), ep.MOVE_S: (0, -1),
             ep.MOVE_E: (1, 0), ep.MOVE_W: (-1, 0)}[code]
        g = [ag.pos[0] + d[0], ag.pos[1] + d[1]]
        g = [max(1, min(w.n - 2, g[0])), max(1, min(w.n - 2, g[1]))]
        clamped = (g == list(ag.pos))
    return {
        "pos": list(ag.pos), "carry": ag.carry, "clamped": clamped,
        "adj_locked": bool(adj and adj.locked),
        "n_locked": sum(1 for o in w.objs if o.locked),
    }


def run_python(seed: int, cfg: dict, cov: Coverage):
    rng = random.Random(seed)
    w = GridWorld(cfg, rng)
    spawn = ([(0 if a.team == "hider" else 1, a.pos[0], a.pos[1]) for a in w.agents],
             [(0 if o.kind == "box" else 1, o.pos[0], o.pos[1]) for o in w.objs])

    arng = random.Random(seed ^ 0x5EED)
    codes = [arng.choices(range(ep.N_ACTIONS), weights=ep._WEIGHTS, k=len(w.agents))
             for _ in range(cfg["sim"]["turns"])]

    for turn_codes in codes:
        pre = [snapshot(w, a, c) for a, c in zip(w.agents, turn_codes)]
        w.resolve_turn({a.id: ep.decode(c, a.pos)
                        for a, c in zip(w.agents, turn_codes)})
        for a, c, b in zip(w.agents, turn_codes, pre):
            observe(w, b, snapshot(w, a, c), c, cov)

    for f in w.frames:
        if any(a["seen"] for a in f["agents"]):
            cov.note("seen")
    return w, codes, spawn


def spec_ints(w, codes, spawn) -> str:
    agents, objs = spawn
    ints = [w.n, w.vision, w.move_budget, w.prep_turns, w.total_turns,
            len(w.agents), len(w.objs), len(w.walls), len(codes)]
    for t, x, y in agents:
        ints += [t, x, y]
    for k, x, y in objs:
        ints += [k, x, y]
    for x, y in sorted(w.walls):
        ints += [x, y]
    for tc in codes:
        ints += tc
    return " ".join(map(str, ints)) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--grid", type=int, default=12)
    a = p.parse_args()

    build_mojo()
    tmp = ROOT / "data" / "_parity"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = ep.make_cfg(grid=a.grid, turns=a.turns)
    cov = Coverage()

    failures = []
    for seed in range(a.seeds):
        w, codes, spawn = run_python(seed, cfg, cov)
        spec = tmp / f"s{seed}.spec"
        spec.write_text(spec_ints(w, codes, spawn))
        want = "\n".join(ep.digest_frame(w, f) for f in w.frames) + "\n"

        out = tmp / f"s{seed}.mojo"
        r = subprocess.run([str(BIN), str(spec), str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            failures.append((seed, "mojo exited " + str(r.returncode) + r.stderr))
            continue
        got = out.read_text()
        if got != want:
            wl, gl = want.splitlines(), got.splitlines()
            first = next((i for i in range(min(len(wl), len(gl))) if wl[i] != gl[i]),
                         min(len(wl), len(gl)))
            failures.append((seed, "frame {}\n  py  {}\n  mjo {}".format(
                first, wl[first] if first < len(wl) else "<eof>",
                gl[first] if first < len(gl) else "<eof>")))

    print("=" * 62)
    print("PARITY  {} seeds x {} turns, grid {}".format(a.seeds, a.turns, a.grid))
    print("=" * 62)
    for seed, msg in failures[:5]:
        print("FAIL seed {}: {}".format(seed, msg))
    print("{}/{} identical".format(a.seeds - len(failures), a.seeds))

    print("\nCOVERAGE (paths exercised across the sweep)")
    for k, v in sorted(cov.hit.items()):
        print("  {:18s} {:>8d} {}".format(k, v, "" if v else "  <== NEVER FIRED"))
    print("  note: bfs unreachable-goal fallback cannot fire at move_budget=1")
    print("        (goal is adjacent and discarded from `blocked`) — untested here.")

    bad = cov.missing()
    if failures:
        sys.exit("\nGATE FAILED: {} seed(s) diverged.".format(len(failures)))
    if bad:
        sys.exit("\nGATE FAILED: paths never exercised: {}".format(", ".join(bad)))
    print("\n✅ M1 GATE PASSED — parity holds and every tracked path fired.")


if __name__ == "__main__":
    main()
