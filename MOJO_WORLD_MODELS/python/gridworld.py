"""Discrete hide-and-seek gridworld.

VENDORED — do not treat as shared code.
    source : DEMO/HIDE_SEEK_LLM/sim/gridworld.py
    sha256 : 4591b960d1af4c1e89bd8aa440e6c63de874a7f0a232ed866cf0df636a71e046
    copied : 2026-08-06, unmodified

MOJO_WORLD_MODELS owns this copy. HIDE_SEEK_LLM is not upstream of anything and
DEMO/ gets cleared out periodically, so this project is deliberately
self-contained (PLAN.md decision 8). Edits here do not propagate anywhere, and
nothing upstream propagates in.

This file is the PARITY ORACLE for mojo/gridworld.mojo: it defines "correct".
If it has a quirk, Mojo reproduces the quirk. Change it only with a matching
change on the Mojo side and a re-run of the M1 parity gate.

Authoritative sim. Agents submit ONE high-level action per turn; the sim resolves
all agents' actions in lockstep, tick by tick, emitting a frame each tick. Frames
are serialized to a trace JSON consumed by the three.js viewer.

Coordinates are integer (x, y) grid cells. The viewer maps (x, y) -> world (x, 0, y).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Obj:
    id: str
    pos: list           # [x, y]
    kind: str           # "box" | "ramp"
    locked: bool = False
    locked_by: str = "" # team that locked it


@dataclass
class Agent:
    id: str
    team: str           # "hider" | "seeker"
    pos: list           # [x, y]
    face: list = field(default_factory=lambda: [0, 1])
    carry: str | None = None   # object id being carried


class GridWorld:
    def __init__(self, cfg: dict, rng: random.Random):
        s = cfg["sim"]
        self.n = s["grid"]
        self.vision = s["vision_range"]
        self.move_budget = s["move_budget"]
        self.prep_turns = max(1, int(round(s["turns"] * s["prep_frac"])))
        self.total_turns = s["turns"]
        self.rng = rng

        self.walls: set[tuple[int, int]] = set()
        self.agents: list[Agent] = []
        self.objs: list[Obj] = []
        self.frames: list[dict] = []
        self.turn = 0
        self._last_actions: dict = {}
        self._spawn(s)
        self.emit_frame()   # record the true spawn layout as frame 0

    # ---- setup ----------------------------------------------------------
    def _free_cell(self, occupied: set) -> list:
        while True:
            c = (self.rng.randrange(1, self.n - 1), self.rng.randrange(1, self.n - 1))
            if c not in occupied and c not in self.walls:
                occupied.add(c)
                return [c[0], c[1]]

    def _spawn(self, s: dict):
        # border walls + a couple of interior wall stubs to make shelter meaningful
        for x in range(self.n):
            self.walls.add((x, 0)); self.walls.add((x, self.n - 1))
            self.walls.add((0, x)); self.walls.add((self.n - 1, x))
        mid = self.n // 2
        for dy in range(-1, 2):
            self.walls.add((mid, mid + dy))

        occ = set(self.walls)
        for i in range(s["hiders"]):
            self.agents.append(Agent(f"h{i}", "hider", self._free_cell(occ)))
        for i in range(s["seekers"]):
            self.agents.append(Agent(f"s{i}", "seeker", self._free_cell(occ)))
        for i in range(s["num_boxes"]):
            self.objs.append(Obj(f"box{i}", self._free_cell(occ), "box"))
        for i in range(s["num_ramps"]):
            self.objs.append(Obj(f"ramp{i}", self._free_cell(occ), "ramp"))

    # ---- queries --------------------------------------------------------
    def obj(self, oid: str) -> Obj | None:
        return next((o for o in self.objs if o.id == oid), None)

    def occupied_cells(self, ignore: set | None = None) -> set:
        ignore = ignore or set()
        cells = set(self.walls)
        for o in self.objs:
            if o.id not in ignore:
                cells.add(tuple(o.pos))
        for a in self.agents:
            cells.add(tuple(a.pos))
        return cells

    def occludes(self, cell: tuple) -> bool:
        if cell in self.walls:
            return True
        return any(tuple(o.pos) == cell and o.kind == "box" for o in self.objs)

    def line_clear(self, a: tuple, b: tuple) -> bool:
        """Bresenham LOS; blocked by walls and boxes (endpoints excluded)."""
        x0, y0 = a; x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cx, cy = x0, y0
        while (cx, cy) != (x1, y1):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; cx += sx
            if e2 < dx:
                err += dx; cy += sy
            if (cx, cy) == (x1, y1):
                break
            if self.occludes((cx, cy)):
                return False
        return True

    def seen_map(self) -> dict:
        """hider_id -> (seen, seeker_id) for current frame (only counts post-prep)."""
        out = {}
        active = self.turn >= self.prep_turns
        seekers = [a for a in self.agents if a.team == "seeker"]
        for h in self.agents:
            if h.team != "hider":
                continue
            seen, by = False, None
            if active:
                for sk in seekers:
                    d = abs(h.pos[0] - sk.pos[0]) + abs(h.pos[1] - sk.pos[1])
                    if d <= self.vision and self.line_clear(tuple(sk.pos), tuple(h.pos)):
                        seen, by = True, sk.id
                        break
            out[h.id] = (seen, by)
        return out

    def bfs_path(self, start: list, goal: list, carry_ignore: set) -> list:
        """Path of cells (excluding start) to goal, avoiding occupied cells."""
        start_t, goal_t = tuple(start), tuple(goal)
        if start_t == goal_t:
            return []
        blocked = self.occupied_cells(ignore=carry_ignore)
        blocked.discard(goal_t)  # allow stepping adjacent / onto target approach
        q = deque([start_t]); prev: dict = {start_t: None}
        while q:
            cur = q.popleft()
            if cur == goal_t:
                break
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = cur[0] + dx, cur[1] + dy
                nc = (nx, ny)
                if not (0 <= nx < self.n and 0 <= ny < self.n):
                    continue
                if nc in prev or (nc in blocked and nc != goal_t):
                    continue
                prev[nc] = cur; q.append(nc)
        target = goal_t
        if goal_t not in prev:
            # goal unreachable (e.g. a sealed fort): head for the reachable cell
            # closest to it, so the agent still advances instead of freezing.
            reachable = [c for c in prev if c != start_t]
            if not reachable:
                return []
            def gdist(c):
                return abs(c[0] - goal_t[0]) + abs(c[1] - goal_t[1])
            target = min(reachable, key=gdist)
            if gdist(target) >= abs(start_t[0] - goal_t[0]) + abs(start_t[1] - goal_t[1]):
                return []          # can't get any closer than we already are
        path = []; cur = target
        while cur != start_t:
            path.append([cur[0], cur[1]]); cur = prev[cur]
        path.reverse()
        return path

    def nearest_obj(self, pos: list, kind: str | None = None, free_only=True) -> Obj | None:
        cands = [o for o in self.objs
                 if (kind is None or o.kind == kind) and (not free_only or not o.locked)]
        if not cands:
            return None
        return min(cands, key=lambda o: abs(o.pos[0] - pos[0]) + abs(o.pos[1] - pos[1]))

    # ---- turn resolution ------------------------------------------------
    def resolve_turn(self, actions: dict):
        """actions: agent_id -> {"action": ..., ...}. Resolves in lockstep ticks."""
        self._last_actions = actions
        plans = {}     # agent_id -> list of cells to walk this turn
        for ag in self.agents:
            if self.turn < self.prep_turns and ag.team == "seeker":
                plans[ag.id] = []          # seekers frozen during prep
                continue
            plans[ag.id] = self._plan(ag, actions.get(ag.id, {"action": "wait"}))

        budget = min(self.move_budget, max((len(p) for p in plans.values()), default=1))
        budget = max(1, budget)
        for tick in range(budget):
            for ag in self.agents:
                path = plans.get(ag.id, [])
                if tick < len(path):
                    target = path[tick]
                    if tuple(target) not in self.occupied_cells(ignore={ag.carry} if ag.carry else None):
                        prev = list(ag.pos)
                        ag.face = [target[0] - ag.pos[0], target[1] - ag.pos[1]]
                        ag.pos = list(target)
                        if ag.carry:
                            o = self.obj(ag.carry)
                            if o:
                                o.pos = prev
            self.emit_frame()
        self.turn += 1

    def _plan(self, ag: Agent, action: dict) -> list:
        """Translate a high-level action into a walk path; apply instant ops now."""
        kind = action.get("action", "wait")
        if kind in ("move_to", "move"):
            if kind == "move":
                d = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}.get(action.get("dir", ""), (0, 0))
                goal = [ag.pos[0] + d[0] * self.move_budget, ag.pos[1] + d[1] * self.move_budget]
                goal = [max(1, min(self.n - 2, goal[0])), max(1, min(self.n - 2, goal[1]))]
            else:
                goal = action.get("target", ag.pos)
            return self.bfs_path(ag.pos, goal, {ag.carry} if ag.carry else set())[: self.move_budget]
        if kind == "grab" and ag.carry is None:
            oid = action.get("object")
            cand = self.obj(oid) if oid else None
            if not (cand and not cand.locked
                    and abs(cand.pos[0] - ag.pos[0]) + abs(cand.pos[1] - ag.pos[1]) <= 1):
                cand = self._adjacent_obj(ag)
            if cand and not cand.locked:
                ag.carry = cand.id
        elif kind == "drop" and ag.carry:
            ag.carry = None
        elif kind == "lock":
            target = action.get("target")
            carried = self.obj(ag.carry) if ag.carry else None
            if (carried and target and self._adjacent_cell(ag.pos, target)
                    and self._placeable(target, ignore_id=carried.id)):
                # place the carried box onto the adjacent target cell and lock it there
                carried.pos = [int(target[0]), int(target[1])]
                carried.locked = True; carried.locked_by = ag.team
                ag.carry = None
            else:
                o = carried or self._adjacent_obj(ag)
                if o:
                    o.locked = True; o.locked_by = ag.team
                    if ag.carry == o.id:
                        ag.carry = None
        elif kind == "unlock":
            o = self._adjacent_obj(ag)
            if o and o.locked and o.locked_by == ag.team:
                o.locked = False
        return []

    def _adjacent_obj(self, ag: Agent) -> Obj | None:
        for o in self.objs:
            if abs(o.pos[0] - ag.pos[0]) + abs(o.pos[1] - ag.pos[1]) <= 1:
                return o
        return None

    def _adjacent_cell(self, pos, cell) -> bool:
        return abs(pos[0] - int(cell[0])) + abs(pos[1] - int(cell[1])) == 1

    def _placeable(self, cell, ignore_id=None) -> bool:
        x, y = int(cell[0]), int(cell[1])
        if not (0 <= x < self.n and 0 <= y < self.n):
            return False
        if (x, y) in self.walls:
            return False
        if any(tuple(o.pos) == (x, y) for o in self.objs if o.id != ignore_id):
            return False
        if any(tuple(a.pos) == (x, y) for a in self.agents):
            return False
        return True

    # ---- output ---------------------------------------------------------
    def emit_frame(self):
        seen = self.seen_map()
        phase = "prep" if self.turn < self.prep_turns else "seek"
        self.frames.append({
            "t": len(self.frames),
            "turn": self.turn,
            "phase": phase,
            "agents": [{
                "id": a.id, "team": a.team, "pos": list(a.pos),
                "face": list(a.face), "carry": a.carry,
                "act": (self._last_actions.get(a.id) or {}).get("action"),
                "seen": seen.get(a.id, (False, None))[0] if a.team == "hider" else False,
            } for a in self.agents],
            "boxes": [{"id": o.id, "pos": list(o.pos), "locked": o.locked}
                      for o in self.objs if o.kind == "box"],
            "ramps": [{"id": o.id, "pos": list(o.pos), "locked": o.locked}
                      for o in self.objs if o.kind == "ramp"],
            "sight": [[by, hid] for hid, (s, by) in seen.items() if s],
        })

    def hider_reward(self) -> float:
        post = [f for f in self.frames if f["phase"] == "seek"]
        if not post:
            return 0.0
        clean = sum(1 for f in post if not any(a["seen"] for a in f["agents"]))
        return round(clean / len(post), 3)

    def export(self, memos: dict) -> dict:
        r = self.hider_reward()
        return {
            "meta": {"grid": self.n, "prep_turns": self.prep_turns,
                     "total_turns": self.total_turns, "frames": len(self.frames)},
            "walls": [[x, y] for (x, y) in sorted(self.walls)],
            "frames": self.frames,
            "result": {"hider_reward": r,
                       "winner": "hider" if r >= 0.5 else "seeker"},
            "memos": memos,
        }
