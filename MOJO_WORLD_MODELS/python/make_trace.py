"""M7 — render a dream beside the reality it was predicting.

Produces two trace.json files in HIDE_SEEK_LLM's exact schema, for the SAME
episode: one from the real sim, one from the dynamics model rolled out closed
loop on the same action sequence. Because the schema is unchanged, the vendored
three.js viewer renders both with no modifications — the structural bet recorded
in PLAN.md's architecture section, cashed in.

Open viewer/compare.html afterwards to watch them drift apart.

    python python/make_trace.py --seed 424242 --steps 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import episode as ep
from dataset import load
from gen_gate import LCG, GenGrid, arm_weights
from gridworld import GridWorld
from model import DynamicsMLP, Spec

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viewer"


class TraceGrid(GenGrid):
    """LCG-spawned like the dataset, but also keeps the frame dicts that
    GridWorld.export() needs. GenGrid alone only accumulates state vectors."""

    def emit_frame(self):
        GridWorld.emit_frame(self)      # the trace frames
        GenGrid.emit_frame(self)        # the state vector


def state_to_frame(s, t, prep_turns, n_agents, n_objs, n_boxes):
    """Predicted state vector -> a frame dict in the viewer's schema."""
    agents, sight = [], []
    for i in range(n_agents):
        b = 2 + i * 6
        team = "hider" if i == 0 else "seeker"
        seen_by = int(s[b + 5])
        seen = team == "hider" and seen_by >= 0
        if seen:
            sight.append([f"s{seen_by - 1}", f"h{i}"])
        carry = int(s[b + 4])
        agents.append({
            "id": f"h{i}" if i == 0 else f"s{i - 1}",
            "team": team,
            "pos": [int(s[b]), int(s[b + 1])],
            "face": [int(s[b + 2]), int(s[b + 3])],
            "carry": (f"box{carry}" if carry < n_boxes else
                      f"ramp{carry - n_boxes}") if carry >= 0 else None,
            "act": None,
            "seen": bool(seen),
        })
    boxes, ramps = [], []
    for i in range(n_objs):
        b = 2 + n_agents * 6 + i * 3
        o = {"id": f"box{i}" if i < n_boxes else f"ramp{i - n_boxes}",
             "pos": [int(s[b]), int(s[b + 1])], "locked": bool(s[b + 2])}
        (boxes if i < n_boxes else ramps).append(o)
    return {"t": t, "turn": int(s[0]),
            "phase": "prep" if int(s[1]) == 0 else "seek",
            "agents": agents, "boxes": boxes, "ramps": ramps, "sight": sight}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--model", default="../data/dynamics")
    p.add_argument("--data", default="../data/train")
    a = p.parse_args()

    _, _, meta = load(a.data)
    spec = Spec(meta)
    grid, turns = meta["grid"], meta["turns"]
    steps = min(a.steps, turns)
    n_ag, n_box = meta["n_agents"], meta["n_boxes"]
    n_obj = n_box + meta["n_ramps"]

    # ---- real episode -----------------------------------------------------
    cfg = ep.make_cfg(grid=grid, turns=turns)
    lcg = LCG(a.seed)
    w = TraceGrid(cfg, lcg, 0)
    wt = arm_weights(0)
    total = sum(wt)
    codes = [[lcg.pick(wt, total) for _ in range(len(w.agents))]
             for _ in range(turns)]
    for tc in codes[:steps]:
        w.resolve_turn({ag.id: ep.decode(c, ag.pos)
                        for ag, c in zip(w.agents, tc)})
    real = w.export({})
    truth = np.array(w.states, dtype=np.int64).reshape(-1, spec.state_dim)

    # ---- the same episode, dreamed ----------------------------------------
    dev = torch.device("cpu")
    model = DynamicsMLP(spec, 1536, 3).to(dev)
    model.load_state_dict(torch.load(a.model + ".pt", map_location=dev))
    model.eval()

    s = torch.from_numpy(truth[0:1].copy()).to(dev)
    acts = np.array(codes[:steps], dtype=np.int64)
    dream_states = [truth[0].copy()]
    with torch.no_grad():
        for t in range(steps):
            logits = model(spec.encode(s, torch.from_numpy(acts[t:t + 1]).to(dev)))
            s = spec.decode(logits, s, t)
            dream_states.append(s[0].cpu().numpy().copy())

    # ---- where they part company ------------------------------------------
    diverged = steps
    for t in range(1, steps + 1):
        if not np.array_equal(dream_states[t], truth[t]):
            diverged = t - 1
            break

    dream = {
        "meta": dict(real["meta"], frames=len(dream_states)),
        "walls": real["walls"],
        "frames": [state_to_frame(st, i, w.prep_turns, n_ag, n_obj, n_box)
                   for i, st in enumerate(dream_states)],
        "result": {"hider_reward": -1.0, "winner": "dream"},
        "memos": {},
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "trace_real.json").write_text(json.dumps(real))
    (OUT / "trace_dream.json").write_text(json.dumps(dream))
    (OUT / "compare_meta.json").write_text(json.dumps({
        "seed": a.seed, "steps": steps, "diverged_after": diverged,
        "real_reward": real["result"]["hider_reward"],
    }))

    print("seed {}  steps {}".format(a.seed, steps))
    print("dream stayed exact for {} steps, then diverged at frame {}".format(
        diverged, diverged + 1))
    print("wrote viewer/trace_real.json, viewer/trace_dream.json")
    print("")
    print("The viewer fetch()es its trace, which file:// blocks on CORS. Serve it:")
    print("    python3 -m http.server 8731")
    print("    open http://localhost:8731/viewer/compare.html")


if __name__ == "__main__":
    main()
