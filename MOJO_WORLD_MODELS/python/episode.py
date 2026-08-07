"""Reference episode driver — the parity oracle for mojo/gridworld.mojo.

Runs the vendored Python sim at move_budget=1 (PLAN.md decision 7: one tick =
one transition, so states are Markov) under a purely state-INDEPENDENT random
policy. State independence matters: it lets the whole action sequence be
sampled up front and exported, so Mojo replays identical actions without ever
reimplementing Python's RNG (PLAN.md: "RNG never crosses the language
boundary").

Two artifacts per episode:
  <stem>.spec    flat whitespace-separated ints: map + action sequence.
                 Deliberately keyword-free so the Mojo side needs no parser.
  <stem>.digest  one canonical line per frame — what the M1 gate diffs.

Usage:
    python python/episode.py --seed 0 --turns 24 --out data/ep000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from gridworld import GridWorld

# ---------------------------------------------------------------------------
# Primitive action set (PLAN.md decision 7). move_to is deliberately excluded:
# it is a BFS macro, and expanding it would reintroduce hidden path state.
# ---------------------------------------------------------------------------
WAIT, MOVE_N, MOVE_S, MOVE_E, MOVE_W = 0, 1, 2, 3, 4
GRAB, DROP, LOCK, UNLOCK = 5, 6, 7, 8
LOCK_N, LOCK_S, LOCK_E, LOCK_W = 9, 10, 11, 12
N_ACTIONS = 13

_DIRS = {MOVE_N: "N", MOVE_S: "S", MOVE_E: "E", MOVE_W: "W"}
_LOCK_DELTA = {LOCK_N: (0, 1), LOCK_S: (0, -1), LOCK_E: (1, 0), LOCK_W: (-1, 0)}

# Sampling weights. Movement dominates so the random walker actually covers the
# grid; build actions stay frequent enough to generate fort-shaped states.
_WEIGHTS = [
    1,              # wait
    6, 6, 6, 6,     # move N S E W
    4,              # grab
    1,              # drop
    2,              # lock in place
    1,              # unlock
    3, 3, 3, 3,     # lock N S E W (place-and-lock — the fort-building action)
]
assert len(_WEIGHTS) == N_ACTIONS


def make_cfg(grid=12, hiders=1, seekers=1, boxes=3, ramps=1,
             turns=24, prep_frac=0.5, vision=6) -> dict:
    """move_budget is pinned to 1 — that is the whole point (decision 7)."""
    return {"sim": {
        "grid": grid, "hiders": hiders, "seekers": seekers,
        "num_boxes": boxes, "num_ramps": ramps,
        "prep_frac": prep_frac, "turns": turns,
        "move_budget": 1, "vision_range": vision,
    }}


def decode(code: int, pos: list) -> dict:
    """Action code -> the dict shape gridworld.resolve_turn expects."""
    if code in _DIRS:
        return {"action": "move", "dir": _DIRS[code]}
    if code in _LOCK_DELTA:
        dx, dy = _LOCK_DELTA[code]
        return {"action": "lock", "target": [pos[0] + dx, pos[1] + dy]}
    return {"action": {WAIT: "wait", GRAB: "grab", DROP: "drop",
                       LOCK: "lock", UNLOCK: "unlock"}[code]}


def digest_frame(w: GridWorld, f: dict) -> str:
    """Canonical one-line frame encoding. Compared byte-for-byte against Mojo.

    Text, not JSON: the gate should fail on simulation divergence, not on float
    formatting or key ordering.
    """
    oid = {o.id: i for i, o in enumerate(w.objs)}
    aid = {a.id: i for i, a in enumerate(w.agents)}
    seen_by = {by_hid[1]: by_hid[0] for by_hid in f["sight"]}

    agents = ";".join(
        "{},{},{},{},{},{}".format(
            a["pos"][0], a["pos"][1], a["face"][0], a["face"][1],
            oid[a["carry"]] if a["carry"] else -1,
            aid[seen_by[a["id"]]] if a["id"] in seen_by else -1,
        )
        for a in f["agents"]
    )
    # Frames split boxes/ramps; the digest walks w.objs so ordering is the sim's.
    by_id = {o["id"]: o for o in f["boxes"] + f["ramps"]}
    objs = ";".join(
        "{},{},{}".format(by_id[o.id]["pos"][0], by_id[o.id]["pos"][1],
                          int(by_id[o.id]["locked"]))
        for o in w.objs
    )
    phase = 0 if f["phase"] == "prep" else 1
    return "{},{},{}|{}|{}".format(f["t"], f["turn"], phase, agents, objs)


def run(seed: int, cfg: dict) -> tuple[GridWorld, list[list[int]]]:
    """Sample the full action sequence up front, then replay it."""
    rng = random.Random(seed)
    w = GridWorld(cfg, rng)
    # Actions come from a SEPARATE stream so map layout and policy don't
    # perturb each other — same seed + different turn count keeps the same map.
    arng = random.Random(seed ^ 0x5EED)
    codes = [arng.choices(range(N_ACTIONS), weights=_WEIGHTS, k=len(w.agents))
             for _ in range(cfg["sim"]["turns"])]
    for turn_codes in codes:
        w.resolve_turn({a.id: decode(c, a.pos)
                        for a, c in zip(w.agents, turn_codes)})
    return w, codes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--turns", type=int, default=24)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--out", type=str, required=True)
    a = p.parse_args()

    cfg = make_cfg(grid=a.grid, turns=a.turns)

    # Spawn layout must be captured BEFORE the episode mutates positions.
    spawn = GridWorld(cfg, random.Random(a.seed))
    spawn_agents = [(0 if ag.team == "hider" else 1, ag.pos[0], ag.pos[1])
                    for ag in spawn.agents]
    spawn_objs = [(0 if o.kind == "box" else 1, o.pos[0], o.pos[1])
                  for o in spawn.objs]

    w, codes = run(a.seed, cfg)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    s = cfg["sim"]
    ints = [w.n, w.vision, w.move_budget, w.prep_turns, w.total_turns,
            len(w.agents), len(w.objs), len(w.walls), len(codes)]
    for team, x, y in spawn_agents:
        ints += [team, x, y]
    for kind, x, y in spawn_objs:
        ints += [kind, x, y]
    for x, y in sorted(w.walls):
        ints += [x, y]
    for turn_codes in codes:
        ints += turn_codes
    out.with_suffix(".spec").write_text(" ".join(map(str, ints)) + "\n")

    lines = [digest_frame(w, f) for f in w.frames]
    out.with_suffix(".digest").write_text("\n".join(lines) + "\n")

    print("seed={} frames={} reward={} -> {}.spec/.digest".format(
        a.seed, len(w.frames), w.hider_reward(), out))


if __name__ == "__main__":
    main()
