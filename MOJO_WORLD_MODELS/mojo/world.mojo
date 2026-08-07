# world.mojo — Target A: the hide-and-seek sim, in Mojo, on CPU.
#
# A line-for-line reimplementation of python/gridworld.py. That file is the
# PARITY ORACLE (PLAN.md M1): where it has a quirk, this reproduces the quirk.
# Quirks deliberately preserved are marked "QUIRK:" — do not "fix" them here
# without changing the Python side and re-running python/parity.py.
#
# SCOPE: move_budget == 1 only (PLAN.md decision 7 — one tick, one transition,
# so states are Markov). load_world() rejects anything else rather than
# silently producing a path that was never parity-checked.
#
# Two output modes, because the profile showed Python spends 14% of its time in
# emit_frame building trace dicts:
#   emit_trace=True   append the canonical digest line (compared against Python)
#   emit_trace=False  simulate only; LOS still runs (see emit_frame)
# So "core" timings measure simulation rather than trace serialization.

from std.sys import argv

# Primitive action set — must match python/episode.py exactly.
comptime WAIT = 0
comptime MOVE_N = 1
comptime MOVE_S = 2
comptime MOVE_E = 3
comptime MOVE_W = 4
comptime GRAB = 5
comptime DROP = 6
comptime LOCK = 7
comptime UNLOCK = 8
comptime LOCK_N = 9
comptime LOCK_S = 10
comptime LOCK_E = 11
comptime LOCK_W = 12

comptime UNVISITED = -2
comptime NO_PREV = -1


struct World(Copyable, Movable):
    var n: Int
    var vision: Int
    var move_budget: Int
    var prep_turns: Int
    var total_turns: Int
    var n_turns: Int
    var turn: Int
    var n_agents: Int
    var n_objs: Int

    var wall: List[Bool]        # flat n*n
    var a_team: List[Int]       # 0 hider, 1 seeker
    var a_x: List[Int]
    var a_y: List[Int]
    var a_fx: List[Int]
    var a_fy: List[Int]
    var a_carry: List[Int]      # obj index, or -1
    var o_kind: List[Int]       # 0 box, 1 ramp
    var o_x: List[Int]
    var o_y: List[Int]
    var o_locked: List[Bool]
    var o_lockedby: List[Int]   # team that locked, or -1

    var codes: List[Int]        # flat n_turns * n_agents

    # Preallocated scratch — kept as fields so the hot loop never allocates.
    var plan_x: List[Int]
    var plan_y: List[Int]
    var plan_has: List[Bool]
    var bfs_prev: List[Int]     # n*n, reset lazily via bfs_order
    var bfs_order: List[Int]

    var emit_trace: Bool
    var emit_state: Bool
    # M8 fitness: fraction of post-prep frames in which no hider is seen.
    # Mirrors python/gridworld.py::hider_reward without materialising frames.
    var post_frames: Int
    var unseen_frames: Int
    var n_frames: Int
    var acc: Int                # see emit_frame — defeats dead-code elimination
    var out: String

    # M3 dataset buffers. Layout per frame (state_dim = 2 + n_agents*6 + n_objs*3):
    #   turn, phase,
    #   per agent:  x, y, face_x, face_y, carry, seen_by
    #   per obj:    x, y, locked
    # Walls are NOT encoded: _spawn adds only border walls plus a fixed 3-cell
    # centre stub, with no RNG, so the layout is identical for every seed at a
    # given grid size. The model therefore learns dynamics for this one
    # topology — generalising across wall layouts is out of scope for v1.
    var states: List[Int16]

    def __init__(out self):
        self.n = 0
        self.vision = 0
        self.move_budget = 1
        self.prep_turns = 0
        self.total_turns = 0
        self.n_turns = 0
        self.turn = 0
        self.n_agents = 0
        self.n_objs = 0
        self.wall = []
        self.a_team = []
        self.a_x = []
        self.a_y = []
        self.a_fx = []
        self.a_fy = []
        self.a_carry = []
        self.o_kind = []
        self.o_x = []
        self.o_y = []
        self.o_locked = []
        self.o_lockedby = []
        self.codes = []
        self.plan_x = []
        self.plan_y = []
        self.plan_has = []
        self.bfs_prev = []
        self.bfs_order = []
        self.emit_trace = True
        self.emit_state = False
        self.post_frames = 0
        self.unseen_frames = 0
        self.n_frames = 0
        self.acc = 0
        self.out = String("")
        self.states = []

    # ---- queries --------------------------------------------------------
    @always_inline
    def cell(self, x: Int, y: Int) -> Int:
        return y * self.n + x

    @always_inline
    def in_bounds(self, x: Int, y: Int) -> Bool:
        return x >= 0 and x < self.n and y >= 0 and y < self.n

    @always_inline
    def is_wall(self, x: Int, y: Int) -> Bool:
        if not self.in_bounds(x, y):
            return False
        return self.wall[self.cell(x, y)]

    def occludes(self, x: Int, y: Int) -> Bool:
        # walls, and boxes only — ramps do not block sight
        if self.is_wall(x, y):
            return True
        for i in range(self.n_objs):
            if self.o_kind[i] == 0 and self.o_x[i] == x and self.o_y[i] == y:
                return True
        return False

    def occupied(self, x: Int, y: Int, ignore_obj: Int) -> Bool:
        """Python's occupied_cells() membership test, without building the set.

        Python rebuilds a ~53-element set per call, 43k times in a 300-episode
        profile run. This is the same predicate as a direct scan.
        """
        if self.is_wall(x, y):
            return True
        for i in range(self.n_objs):
            if i != ignore_obj and self.o_x[i] == x and self.o_y[i] == y:
                return True
        for i in range(self.n_agents):
            if self.a_x[i] == x and self.a_y[i] == y:
                return True
        return False

    def line_clear(self, x0: Int, y0: Int, x1: Int, y1: Int) -> Bool:
        """Bresenham LOS, endpoints excluded. Ported step-for-step."""
        var dx = abs(x1 - x0)
        var dy = abs(y1 - y0)
        var sx = 1 if x0 < x1 else -1
        var sy = 1 if y0 < y1 else -1
        var err = dx - dy
        var cx = x0
        var cy = y0
        while not (cx == x1 and cy == y1):
            var e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
            if cx == x1 and cy == y1:
                break
            if self.occludes(cx, cy):
                return False
        return True

    def seen_by(self, h: Int) -> Int:
        """Index of the first seeker with a sightline to hider h, else -1."""
        if self.turn < self.prep_turns:
            return -1
        for s in range(self.n_agents):
            if self.a_team[s] != 1:
                continue
            var d = abs(self.a_x[h] - self.a_x[s]) + abs(self.a_y[h] - self.a_y[s])
            if d <= self.vision and self.line_clear(
                self.a_x[s], self.a_y[s], self.a_x[h], self.a_y[h]
            ):
                return s
        return -1

    def adjacent_obj(self, ag: Int) -> Int:
        # QUIRK: first in list order, ignoring locked state and kind.
        for i in range(self.n_objs):
            if abs(self.o_x[i] - self.a_x[ag]) + abs(self.o_y[i] - self.a_y[ag]) <= 1:
                return i
        return -1

    def placeable(self, x: Int, y: Int, ignore_obj: Int) -> Bool:
        if not self.in_bounds(x, y):
            return False
        if self.is_wall(x, y):
            return False
        for i in range(self.n_objs):
            if i != ignore_obj and self.o_x[i] == x and self.o_y[i] == y:
                return False
        for i in range(self.n_agents):
            if self.a_x[i] == x and self.a_y[i] == y:
                return False
        return True

    # ---- pathing --------------------------------------------------------
    def bfs_step(mut self, ag: Int, gx: Int, gy: Int) -> Bool:
        """BFS toward (gx, gy); writes the single step into plan_[xy][ag].

        Returns True if a step was produced. The unreachable-goal fallback is
        ported in full for parity and for B2, but cannot fire at
        move_budget=1 — see PLAN.md "Known gaps".
        """
        var sx = self.a_x[ag]
        var sy = self.a_y[ag]
        if sx == gx and sy == gy:
            return False

        var ignore = self.a_carry[ag]
        var goal = self.cell(gx, gy)
        var start = self.cell(sx, sy)

        self.bfs_order.clear()
        self.bfs_prev[start] = NO_PREV
        self.bfs_order.append(start)

        var head = 0
        while head < len(self.bfs_order):
            var cur = self.bfs_order[head]
            head += 1
            if cur == goal:
                break
            var cx = cur % self.n
            var cy = cur // self.n
            # Neighbour order must match Python: (0,1), (0,-1), (1,0), (-1,0).
            for k in range(4):
                var nx = cx
                var ny = cy
                if k == 0:
                    ny += 1
                elif k == 1:
                    ny -= 1
                elif k == 2:
                    nx += 1
                else:
                    nx -= 1
                if not self.in_bounds(nx, ny):
                    continue
                var nc = self.cell(nx, ny)
                if self.bfs_prev[nc] != UNVISITED:
                    continue
                # Python discards the goal from `blocked`, so it is always enterable.
                if nc != goal and self.occupied(nx, ny, ignore):
                    continue
                self.bfs_prev[nc] = cur
                self.bfs_order.append(nc)

        var target = goal
        var ok = True
        if self.bfs_prev[goal] == UNVISITED:
            # QUIRK: ties go to the earliest-discovered cell, matching Python's
            # min() over an insertion-ordered dict.
            var best = -1
            var best_d = 0
            for i in range(len(self.bfs_order)):
                var c = self.bfs_order[i]
                if c == start:
                    continue
                var d = abs(c % self.n - gx) + abs(c // self.n - gy)
                if best == -1 or d < best_d:
                    best = c
                    best_d = d
            if best == -1 or best_d >= abs(sx - gx) + abs(sy - gy):
                ok = False          # cannot get any closer than we already are
            else:
                target = best

        if ok:
            var cur2 = target
            while self.bfs_prev[cur2] != start:
                cur2 = self.bfs_prev[cur2]
                if cur2 == NO_PREV or cur2 == UNVISITED:
                    ok = False
                    break
            if ok:
                self.plan_x[ag] = cur2 % self.n
                self.plan_y[ag] = cur2 // self.n

        # Reset only what we touched — O(visited), not O(n^2).
        for i in range(len(self.bfs_order)):
            self.bfs_prev[self.bfs_order[i]] = UNVISITED
        return ok

    # ---- turn resolution ------------------------------------------------
    def plan(mut self, ag: Int, code: Int) -> Bool:
        """Translate an action code into a step; apply instant ops now."""
        if code >= MOVE_N and code <= MOVE_W:
            var dx = 0
            var dy = 0
            if code == MOVE_N:
                dy = 1
            elif code == MOVE_S:
                dy = -1
            elif code == MOVE_E:
                dx = 1
            else:
                dx = -1
            var gx = self.a_x[ag] + dx * self.move_budget
            var gy = self.a_y[ag] + dy * self.move_budget
            gx = max(1, min(self.n - 2, gx))
            gy = max(1, min(self.n - 2, gy))
            return self.bfs_step(ag, gx, gy)

        if code == GRAB:
            if self.a_carry[ag] == -1:
                # QUIRK: no object id is ever supplied, so this always falls
                # through to adjacent_obj — and if that object is locked, the
                # grab fails even when another free object is also adjacent.
                var cand = self.adjacent_obj(ag)
                if cand != -1 and not self.o_locked[cand]:
                    self.a_carry[ag] = cand
            return False

        if code == DROP:
            if self.a_carry[ag] != -1:
                self.a_carry[ag] = -1
            return False

        if code == UNLOCK:
            var o = self.adjacent_obj(ag)
            if o != -1 and self.o_locked[o] and self.o_lockedby[o] == self.a_team[ag]:
                self.o_locked[o] = False
            return False

        if code == LOCK or (code >= LOCK_N and code <= LOCK_W):
            var has_target = code >= LOCK_N
            var tx = 0
            var ty = 0
            if has_target:
                if code == LOCK_N:
                    ty = 1
                elif code == LOCK_S:
                    ty = -1
                elif code == LOCK_E:
                    tx = 1
                else:
                    tx = -1
                tx += self.a_x[ag]
                ty += self.a_y[ag]

            var carried = self.a_carry[ag]
            var adj = abs(self.a_x[ag] - tx) + abs(self.a_y[ag] - ty) == 1
            if carried != -1 and has_target and adj and self.placeable(tx, ty, carried):
                # place-and-lock: the fort-building action
                self.o_x[carried] = tx
                self.o_y[carried] = ty
                self.o_locked[carried] = True
                self.o_lockedby[carried] = self.a_team[ag]
                self.a_carry[ag] = -1
            else:
                var o = carried if carried != -1 else self.adjacent_obj(ag)
                if o != -1:
                    self.o_locked[o] = True
                    self.o_lockedby[o] = self.a_team[ag]
                    if self.a_carry[ag] == o:
                        self.a_carry[ag] = -1
            return False

        return False          # wait

    def resolve_turn(mut self, base: Int):
        for ag in range(self.n_agents):
            # Seekers are frozen during prep — and crucially plan() is NOT
            # called, so their grab/lock side effects do not fire either.
            if self.turn < self.prep_turns and self.a_team[ag] == 1:
                self.plan_has[ag] = False
            else:
                self.plan_has[ag] = self.plan(ag, self.codes[base + ag])

        # move_budget == 1, so budget is always exactly one tick.
        for ag in range(self.n_agents):
            if not self.plan_has[ag]:
                continue
            var tx = self.plan_x[ag]
            var ty = self.plan_y[ag]
            if not self.occupied(tx, ty, self.a_carry[ag]):
                var px = self.a_x[ag]
                var py = self.a_y[ag]
                self.a_fx[ag] = tx - px
                self.a_fy[ag] = ty - py
                self.a_x[ag] = tx
                self.a_y[ag] = ty
                var c = self.a_carry[ag]
                if c != -1:
                    self.o_x[c] = px       # the carried box trails behind
                    self.o_y[c] = py
        self.emit_frame()
        self.turn += 1

    def reward(self) -> Float32:
        """hider_reward: share of post-prep frames with no hider in sight."""
        if self.post_frames == 0:
            return 0.0
        return Float32(self.unseen_frames) / Float32(self.post_frames)

    def resolve_turn_direct(mut self, c0: Int, c1: Int):
        """resolve_turn, but with action codes supplied per call.

        For M8, where a controller decides each action from the current state
        rather than replaying a pre-sampled sequence. Assumes n_agents == 2
        (hider, seeker) — the only configuration this project uses.

        The body is duplicated from resolve_turn rather than factored out:
        resolve_turn is parity-critical (M1) and on the M2 benchmark's hot
        path, and the obvious refactor — passing a List of codes — would add
        a per-turn allocation there. Any change here must be mirrored above.
        """
        for ag in range(self.n_agents):
            if self.turn < self.prep_turns and self.a_team[ag] == 1:
                self.plan_has[ag] = False
            else:
                self.plan_has[ag] = self.plan(ag, c0 if ag == 0 else c1)

        for ag in range(self.n_agents):
            if not self.plan_has[ag]:
                continue
            var tx = self.plan_x[ag]
            var ty = self.plan_y[ag]
            if not self.occupied(tx, ty, self.a_carry[ag]):
                var px = self.a_x[ag]
                var py = self.a_y[ag]
                self.a_fx[ag] = tx - px
                self.a_fy[ag] = ty - py
                self.a_x[ag] = tx
                self.a_y[ag] = ty
                var c = self.a_carry[ag]
                if c != -1:
                    self.o_x[c] = px
                    self.o_y[c] = py
        self.emit_frame()
        self.turn += 1

    def run_all(mut self):
        # Frame 0 (the spawn layout) is emitted by load_world, so that both
        # languages have it OUTSIDE the benchmark's timed region — Python's
        # GridWorld.__init__ emits it at construction and cannot easily defer.
        for t in range(self.n_turns):
            self.resolve_turn(t * self.n_agents)

    # ---- output ---------------------------------------------------------
    def emit_frame(mut self):
        """Two modes. LOS runs in BOTH — it is simulation, not serialization.

        The no-trace branch accumulates `sb` into `acc` rather than discarding
        it, so the compiler cannot eliminate seen_by(). One integer add per
        agent per frame: cheap enough not to distort the measurement, and
        cheap in Python too. (An earlier version folded every field into an
        FNV checksum here; in Python that cost 27 function calls per frame and
        made the no-trace mode SLOWER than the trace mode, inflating Mojo's
        apparent win. Correctness of the two sims is established by the M1
        digest gate, not by a checksum computed inside the benchmark.)
        """
        var phase = 0 if self.turn < self.prep_turns else 1
        var any_seen = False
        if self.emit_trace:
            self.out += String(self.n_frames)
            self.out += ","
            self.out += String(self.turn)
            self.out += ","
            self.out += String(phase)
            self.out += "|"
        if self.emit_state:
            self.states.append(Int16(self.turn))
            self.states.append(Int16(phase))

        for i in range(self.n_agents):
            var sb = -1
            if self.a_team[i] == 0:
                sb = self.seen_by(i)
                if sb >= 0:
                    any_seen = True
            if self.emit_state:
                self.states.append(Int16(self.a_x[i]))
                self.states.append(Int16(self.a_y[i]))
                self.states.append(Int16(self.a_fx[i]))
                self.states.append(Int16(self.a_fy[i]))
                self.states.append(Int16(self.a_carry[i]))
                self.states.append(Int16(sb))
            if self.emit_trace:
                if i > 0:
                    self.out += ";"
                self.out += String(self.a_x[i])
                self.out += ","
                self.out += String(self.a_y[i])
                self.out += ","
                self.out += String(self.a_fx[i])
                self.out += ","
                self.out += String(self.a_fy[i])
                self.out += ","
                self.out += String(self.a_carry[i])
                self.out += ","
                self.out += String(sb)
            elif not self.emit_state:
                self.acc += sb

        if self.emit_state:
            for i in range(self.n_objs):
                self.states.append(Int16(self.o_x[i]))
                self.states.append(Int16(self.o_y[i]))
                self.states.append(Int16(1 if self.o_locked[i] else 0))

        if self.emit_trace:
            self.out += "|"
            for i in range(self.n_objs):
                if i > 0:
                    self.out += ";"
                self.out += String(self.o_x[i])
                self.out += ","
                self.out += String(self.o_y[i])
                self.out += ","
                self.out += String(1 if self.o_locked[i] else 0)
            self.out += "\n"

        # M8 fitness accumulator. Only post-prep frames count, matching
        # python/gridworld.py::hider_reward.
        if phase == 1:
            self.post_frames += 1
            if not any_seen:
                self.unseen_frames += 1
        self.n_frames += 1


def read_ints(path: String) raises -> List[Int]:
    var text = open(path, "r").read()
    var toks = text.split()
    var vals = List[Int]()
    for i in range(len(toks)):
        vals.append(Int(String(toks[i])))
    return vals^


def load_world(v: List[Int], mut p: Int, emit_trace: Bool) raises -> World:
    """Parse one episode spec starting at v[p]; advances p past it.

    Does not run the episode — the benchmark times run_all() alone.
    """
    var w = World()
    w.emit_trace = emit_trace
    w.n = v[p]
    w.vision = v[p + 1]
    w.move_budget = v[p + 2]
    w.prep_turns = v[p + 3]
    w.total_turns = v[p + 4]
    w.n_agents = v[p + 5]
    w.n_objs = v[p + 6]
    var n_walls = v[p + 7]
    w.n_turns = v[p + 8]
    p += 9

    if w.move_budget != 1:
        raise Error("move_budget must be 1 (PLAN.md decision 7); got " +
                    String(w.move_budget))

    w.wall = List[Bool](length=w.n * w.n, fill=False)
    w.bfs_prev = List[Int](length=w.n * w.n, fill=UNVISITED)

    for _ in range(w.n_agents):
        w.a_team.append(v[p])
        w.a_x.append(v[p + 1])
        w.a_y.append(v[p + 2])
        w.a_fx.append(0)         # spawn facing, matches the Python dataclass default
        w.a_fy.append(1)
        w.a_carry.append(-1)
        w.plan_x.append(0)
        w.plan_y.append(0)
        w.plan_has.append(False)
        p += 3

    for _ in range(w.n_objs):
        w.o_kind.append(v[p])
        w.o_x.append(v[p + 1])
        w.o_y.append(v[p + 2])
        w.o_locked.append(False)
        w.o_lockedby.append(-1)
        p += 3

    for _ in range(n_walls):
        w.wall[w.cell(v[p], v[p + 1])] = True
        p += 2

    for _ in range(w.n_turns * w.n_agents):
        w.codes.append(v[p])
        p += 1

    w.emit_frame()          # frame 0: the spawn layout (see run_all)
    return w^
