# gen.mojo — M3: generate the transition dataset, entirely Mojo-side.
#
# M1/M2 kept the RNG on the Python side and shipped specs across the boundary.
# That does not scale to 10^5 episodes, so here Mojo generates maps AND actions
# itself, using the SAME 32-bit LCG as MOJO_CURRICULUM/04_train_mlp.mojo and
# train_mlp_reference.py:  s = (1103515245*s + 12345) mod 2^31.
#
# python/gen_ref.py reimplements every draw below in the same call order, so a
# sample of episodes can be digest-compared against Python (python/gen_gate.py).
# That is what keeps self-generated data trustworthy.
#
# Four arms (PLAN.md "Data mix"):
#   A  random policy          — bulk state coverage; the random walker is the
#                               right default (Ha collects with a random policy)
#   B  pre-locked fort spawns — boxes locked into a corner seal at t=0. LOAD-
#                               BEARING: random and build-biased play produced
#                               ZERO seals in 80k episodes, so without this the
#                               dataset has no fort-shaped states at all.
#   C  build-biased policy    — grab/lock reweighted up, meant to produce fort-
#                               building TRAJECTORIES. Measured as a failure:
#                               still zero seals. Kept for action-mix diversity.
#   D  box within reach       — box parked beside the hider so grab succeeds,
#                               with weights favouring HOLDING. Added after M4
#                               measured carry as the worst subset.
#
# Run: uv run mojo gen.mojo <grid> <turns> <seed> <out-prefix> <nA> <nB> <nC> <nD>

from std.sys import argv, num_performance_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize
from world import World

comptime N_ACTIONS = 13
comptime CHUNK = 2000          # episodes per chunk; bounds peak memory


struct LCG(Copyable, Movable):
    """32-bit LCG. Same constants and modulus as the curriculum's reference."""

    var s: Int

    def __init__(out self, seed: Int):
        # Reduce mod 2^31 at construction. Without this, per-episode seeds
        # (base + idx*7919) exceed 2^31 past ~271k episodes and the first
        # 1103515245*s overflows Int64 — silently, and only at scale.
        var s = seed % (1 << 31)
        self.s = s if s != 0 else 1

    def next(mut self) -> Int:
        self.s = (1103515245 * self.s + 12345) % (1 << 31)
        return self.s

    def below(mut self, n: Int) -> Int:
        return self.next() % n

    def pick(mut self, w: List[Int], total: Int) -> Int:
        """Weighted choice by cumulative walk — matches gen_ref.py exactly."""
        var r = self.below(total)
        var c = 0
        for i in range(len(w)):
            c += w[i]
            if r < c:
                return i
        return len(w) - 1


def arm_weights(arm: Int) -> List[Int]:
    if arm == 3:
        # D: carry-biased. grab boosted, locks SUPPRESSED. The first attempt
        # reused C's weights and barely moved carry coverage (1.0% -> 1.0%),
        # because locks were 52% of C's distribution so the hider dropped the
        # box ~2 turns after grabbing it. Holding is what needs sampling.
        return [1, 8, 8, 8, 8, 10, 1, 1, 1, 2, 2, 2, 2]
    if arm == 2:
        # C: build-biased. grab and place-and-lock boosted.
        return [1, 4, 4, 4, 4, 8, 1, 4, 1, 6, 6, 6, 6]
    # A and B: the same mix python/episode.py uses.
    return [1, 6, 6, 6, 6, 4, 1, 2, 1, 3, 3, 3, 3]


def make_episode(
    grid: Int, vision: Int, turns: Int, prep: Int,
    n_hiders: Int, n_seekers: Int, n_boxes: Int, n_ramps: Int,
    seed: Int, arm: Int,
) raises -> World:
    var lcg = LCG(seed)
    var w = World()
    w.emit_trace = False
    w.emit_state = True
    w.n = grid
    w.vision = vision
    w.move_budget = 1
    w.prep_turns = prep
    w.total_turns = turns
    w.n_turns = turns
    w.n_agents = n_hiders + n_seekers
    w.n_objs = n_boxes + n_ramps

    w.wall = List[Bool](length=grid * grid, fill=False)
    w.bfs_prev = List[Int](length=grid * grid, fill=-2)

    # Border walls plus a fixed 3-cell centre stub — no RNG, exactly as
    # python/gridworld.py::_spawn does it.
    for x in range(grid):
        w.wall[w.cell(x, 0)] = True
        w.wall[w.cell(x, grid - 1)] = True
        w.wall[w.cell(0, x)] = True
        w.wall[w.cell(grid - 1, x)] = True
    var mid = grid // 2
    for dy in range(-1, 2):
        w.wall[w.cell(mid, mid + dy)] = True

    var occ = List[Bool](length=grid * grid, fill=False)
    for i in range(grid * grid):
        occ[i] = w.wall[i]

    # _free_cell: draw x then y in [1, grid-2], reject if occupied.
    var placed_x = List[Int]()
    var placed_y = List[Int]()
    for _ in range(w.n_agents + w.n_objs):
        while True:
            var x = 1 + lcg.below(grid - 2)
            var y = 1 + lcg.below(grid - 2)
            if not occ[w.cell(x, y)]:
                occ[w.cell(x, y)] = True
                placed_x.append(x)
                placed_y.append(y)
                break

    for i in range(w.n_agents):
        w.a_team.append(0 if i < n_hiders else 1)
        w.a_x.append(placed_x[i])
        w.a_y.append(placed_y[i])
        w.a_fx.append(0)
        w.a_fy.append(1)
        w.a_carry.append(-1)
        w.plan_x.append(0)
        w.plan_y.append(0)
        w.plan_has.append(False)

    for i in range(w.n_objs):
        w.o_kind.append(0 if i < n_boxes else 1)
        w.o_x.append(placed_x[w.n_agents + i])
        w.o_y.append(placed_y[w.n_agents + i])
        w.o_locked.append(False)
        w.o_lockedby.append(-1)

    # ---- arm B: pre-locked corner seal -----------------------------------
    # Put the hider in a corner and lock boxes across the two open sides plus
    # the diagonal, i.e. the arrangement that scored reward 1.0 in HIDE_SEEK.
    # The two border walls cover the other sides.
    if arm == 1 and n_boxes >= 3:
        var corner = lcg.below(4)
        var cx = 1 if (corner & 1) == 0 else grid - 2
        var cy = 1 if (corner & 2) == 0 else grid - 2
        var ix = 1 if cx == 1 else -1        # inward direction
        var iy = 1 if cy == 1 else -1
        var sx = [cx + ix, cx, cx + ix]
        var sy = [cy, cy + iy, cy + iy]
        var k = 1 + lcg.below(3)             # 1..3 boxes of the seal present
        for b in range(k):
            w.o_x[b] = sx[b]
            w.o_y[b] = sy[b]
            w.o_locked[b] = True
            w.o_lockedby[b] = 0
        w.a_x[0] = cx                        # hider inside the seal
        w.a_y[0] = cy
        # Hand the hider the next box half the time, so a single lock_[NSEW]
        # can COMPLETE the seal. Without this the dataset contains sealed
        # states but never a transition INTO one — measured: random and
        # build-biased play produced zero seals in 80k episodes, so
        # seal-completion would otherwise be entirely unobserved.
        if k < n_boxes and lcg.below(2) == 0:
            w.a_carry[0] = k

    # ---- arm D: a box within reach ---------------------------------------
    # M4 measured carry transitions at the worst exact-match of any subset,
    # because carrying is only ~1% of frames: `grab` fires 10% of the time but
    # almost always fails, since the vendored _adjacent_obj quirk needs an
    # object at Manhattan distance <= 1 and one rarely is. Parking box 0 beside
    # the hider lets grab actually succeed, so genuine grab -> carry -> move ->
    # place-and-lock sequences appear instead of near-zero carry coverage.
    # Fixed direction order (no extra LCG draw) keeps gen_ref.py in step.
    if arm == 3:
        for k in range(4):
            var nx = w.a_x[0]
            var ny = w.a_y[0]
            if k == 0:
                ny += 1
            elif k == 1:
                ny -= 1
            elif k == 2:
                nx += 1
            else:
                nx -= 1
            if w.placeable(nx, ny, 0):
                w.o_x[0] = nx
                w.o_y[0] = ny
                break

    # ---- action sequence --------------------------------------------------
    var wt = arm_weights(arm)
    var total = 0
    for i in range(len(wt)):
        total += wt[i]
    for _ in range(turns):
        for _a in range(w.n_agents):
            w.codes.append(lcg.pick(wt, total))

    w.emit_frame()          # frame 0 — the spawn layout
    return w^


@always_inline
def push_i16(mut buf: List[UInt8], v: Int):
    buf.append(UInt8(v & 0xFF))
    buf.append(UInt8((v >> 8) & 0xFF))


def main() raises:
    var args = argv()
    if len(args) < 9:
        print("usage: gen <grid> <turns> <seed> <out-prefix> <nA> <nB> <nC> <nD>")
        return

    var grid = Int(String(args[1]))
    var turns = Int(String(args[2]))
    var base = Int(String(args[3]))
    var prefix = String(args[4])
    var counts = [Int(String(args[5])), Int(String(args[6])),
                  Int(String(args[7])), Int(String(args[8]))]
    var n_eps = counts[0] + counts[1] + counts[2] + counts[3]

    var vision = 6
    var prep = turns // 2
    var n_hiders = 1
    var n_seekers = 1
    var n_boxes = 3
    var n_ramps = 1
    var n_agents = n_hiders + n_seekers
    var n_objs = n_boxes + n_ramps
    var state_dim = 2 + n_agents * 6 + n_objs * 3
    var workers = num_performance_cores()

    # Truncate outputs, then append per chunk.
    with open(prefix + ".states.bin", "w") as f:
        f.write("")
    with open(prefix + ".actions.bin", "w") as f:
        f.write("")

    var t0 = perf_counter_ns()
    var done = 0
    var frames_total = 0

    while done < n_eps:
        var this = min(CHUNK, n_eps - done)
        var ws = List[World]()
        for k in range(this):
            var idx = done + k
            var arm = 0
            if idx >= counts[0] + counts[1] + counts[2]:
                arm = 3
            elif idx >= counts[0] + counts[1]:
                arm = 2
            elif idx >= counts[0]:
                arm = 1
            # Seed per episode so parallel scheduling cannot change the data.
            ws.append(make_episode(grid, vision, turns, prep, n_hiders,
                                   n_seekers, n_boxes, n_ramps,
                                   base + idx * 7919, arm))

        @parameter
        def work(i: Int):
            ws[i].run_all()

        parallelize[work](this, workers)

        var sbuf = List[UInt8]()
        var abuf = List[UInt8]()
        for ref w in ws:
            for i in range(len(w.states)):
                push_i16(sbuf, Int(w.states[i]))
            for i in range(len(w.codes)):
                push_i16(abuf, w.codes[i])
            frames_total += w.n_frames
        with open(prefix + ".states.bin", "a") as f:
            f.write_bytes(Span(sbuf))
        with open(prefix + ".actions.bin", "a") as f:
            f.write_bytes(Span(abuf))

        done += this

    var t1 = perf_counter_ns()

    # Sidecar so the dataset is self-describing (shapes, mix, provenance).
    var meta = String("{\n")
    meta += '  "episodes": ' + String(n_eps) + ",\n"
    meta += '  "turns": ' + String(turns) + ",\n"
    meta += '  "frames_per_episode": ' + String(turns + 1) + ",\n"
    meta += '  "grid": ' + String(grid) + ",\n"
    meta += '  "vision": ' + String(vision) + ",\n"
    meta += '  "prep_turns": ' + String(prep) + ",\n"
    meta += '  "n_agents": ' + String(n_agents) + ",\n"
    meta += '  "n_boxes": ' + String(n_boxes) + ",\n"
    meta += '  "n_ramps": ' + String(n_ramps) + ",\n"
    meta += '  "state_dim": ' + String(state_dim) + ",\n"
    meta += '  "dtype": "<i2",\n'
    meta += '  "base_seed": ' + String(base) + ",\n"
    meta += '  "seed_stride": 7919,\n'
    meta += '  "arm_counts": {"random": ' + String(counts[0])
    meta += ', "prelocked_fort": ' + String(counts[1])
    meta += ', "build_biased": ' + String(counts[2])
    meta += ', "box_in_reach": ' + String(counts[3]) + "},\n"
    meta += '  "states_shape": [' + String(n_eps) + ", " + String(turns + 1)
    meta += ", " + String(state_dim) + "],\n"
    meta += '  "actions_shape": [' + String(n_eps) + ", " + String(turns)
    meta += ", " + String(n_agents) + "],\n"
    meta += '  "state_layout": "turn, phase, then per agent (x,y,face_x,'
    meta += 'face_y,carry,seen_by), then per obj (x,y,locked)",\n'
    meta += '  "walls": "border ring plus a 3-cell centre stub; identical for '
    meta += 'every seed, so not encoded in the state",\n'
    meta += '  "gen_ms": ' + String(Int(t1 - t0) // 1000000) + "\n}\n"
    with open(prefix + ".meta.json", "w") as f:
        f.write(meta)

    print("GEN episodes=", n_eps,
          " turns=", turns,
          " frames=", frames_total,
          " state_dim=", state_dim,
          " n_agents=", n_agents,
          " workers=", workers,
          " ms=", Int(t1 - t0) // 1000000)
