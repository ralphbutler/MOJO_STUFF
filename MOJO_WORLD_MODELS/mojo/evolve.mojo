# evolve.mojo — M8: evolve a hider controller inside the real Mojo sim.
#
# PLAN.md decision 9: evolve against the REAL simulator, not the dream. M5
# measured the crossover — evaluating hundreds of candidates per generation is
# the large-batch regime where PyTorch beats Mojo 3x, while against the real
# sim Mojo is 300x (M2). The dream is also only ~6% faithful over a full
# episode, so a controller trained there would optimise against a fiction.
#
# Known cost of that choice (PLAN.md open question 7): this is no longer Ha's
# "sever the environment" claim. It demonstrates Mojo throughput. The
# world-model result rests on M5's fidelity numbers.
#
# Optimiser: SEPARABLE CMA-ES (Ros & Hansen 2008) — diagonal covariance, O(n)
# per generation. Full CMA-ES would need a 598x598 eigendecomposition every
# generation and LAPACK; the optimiser is not what this project measures.
#
# Opponent: a greedy full-knowledge seeker. Deterministic, so fitness has no
# opponent noise, and strong enough that hiding in the open does not work —
# occlusion is the only counter.
#
# Run: uv run mojo evolve.mojo <gens> <pop> <maps> <seed> [out.txt]

from std.sys import argv, num_performance_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize
from std.math import sqrt, log, exp
from world import World
from gen import LCG, make_episode
from dream import Dreamer, load_dreamer

comptime N_ACT = 13
comptime GRID = 12
comptime TURNS = 100
comptime N_FEAT = 45
comptime N_PARAM = N_FEAT * N_ACT + N_ACT      # 598
comptime STATE_DIM = 26                        # matches M3 / model.py


# ---------------------------------------------------------------------------
# State-vector helpers.
#
# OPEN QUESTION 7: both arms — real sim and dream — must share EXACTLY these
# functions, or the measured delta would be my featuriser rather than the
# dynamics. So the real-sim arm also extracts a state vector each turn and
# reads features from it, even though it could read World directly.
# ---------------------------------------------------------------------------
def static_walls(grid: Int) -> List[Bool]:
    """Border ring + 3-cell centre stub. No RNG, identical for every seed."""
    var w = List[Bool](length=grid * grid, fill=False)
    for x in range(grid):
        w[0 * grid + x] = True
        w[(grid - 1) * grid + x] = True
        w[x * grid + 0] = True
        w[x * grid + grid - 1] = True
    var mid = grid // 2
    for dy in range(-1, 2):
        w[(mid + dy) * grid + mid] = True
    return w^


def state_of(w: World) -> List[Int]:
    var s = List[Int](length=STATE_DIM, fill=0)
    s[0] = w.turn
    s[1] = 1 if w.turn >= w.prep_turns else 0
    var p = 2
    for i in range(w.n_agents):
        s[p] = w.a_x[i]
        s[p + 1] = w.a_y[i]
        s[p + 2] = w.a_fx[i]
        s[p + 3] = w.a_fy[i]
        s[p + 4] = w.a_carry[i]
        s[p + 5] = w.seen_by(i) if w.a_team[i] == 0 else -1
        p += 6
    for i in range(w.n_objs):
        s[p] = w.o_x[i]
        s[p + 1] = w.o_y[i]
        s[p + 2] = 1 if w.o_locked[i] else 0
        p += 3
    return s^


def occ_s(s: List[Int], wall: List[Bool], x: Int, y: Int) -> Bool:
    if x < 0 or x >= GRID or y < 0 or y >= GRID:
        return True
    if wall[y * GRID + x]:
        return True
    for i in range(4):                       # objects at cols 14,17,20,23
        if s[14 + i * 3] == x and s[15 + i * 3] == y:
            return True
    for i in range(2):                       # agents at cols 2, 8
        if s[2 + i * 6] == x and s[3 + i * 6] == y:
            return True
    return False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
def features(s: List[Int], wall: List[Bool]) -> List[Float32]:
    """45 features for the hider, read from a STATE VECTOR so the real-sim and
    dream arms are featurised identically. Compact by design: one-hot position
    plus RELATIVE offsets to seeker and objects, so a linear policy can express
    "move away from the seeker" without a hidden layer."""
    var f = List[Float32](length=N_FEAT, fill=0.0)
    var hx = s[2]
    var hy = s[3]
    var g = Float32(GRID)

    f[hx] = 1.0                      # 0..11   hider x one-hot
    f[GRID + hy] = 1.0               # 12..23  hider y one-hot
    f[24] = Float32(s[8] - hx) / g   # seeker offset
    f[25] = Float32(s[9] - hy) / g
    f[26] = Float32(s[1])            # phase
    f[27] = 1.0 if s[6] != -1 else 0.0     # carrying
    f[28] = 1.0 if s[7] >= 0 else 0.0      # currently seen
    var p = 29
    for i in range(4):               # 29..40  relative object offsets + locked
        f[p] = Float32(s[14 + i * 3] - hx) / g
        f[p + 1] = Float32(s[15 + i * 3] - hy) / g
        f[p + 2] = Float32(s[16 + i * 3])
        p += 3
    # 41..44  is each orthogonal neighbour blocked? Without this the policy is
    # blind to walls and wastes most actions bumping into them.
    f[41] = 1.0 if occ_s(s, wall, hx, hy + 1) else 0.0
    f[42] = 1.0 if occ_s(s, wall, hx, hy - 1) else 0.0
    f[43] = 1.0 if occ_s(s, wall, hx + 1, hy) else 0.0
    f[44] = 1.0 if occ_s(s, wall, hx - 1, hy) else 0.0
    return f^


def policy(theta: List[Float32], f: List[Float32]) -> Int:
    """argmax of a linear layer — Ha's controller is linear too.

    SIMD-widened, because python/evolve_ref.py hands this matrix-vector
    product to NumPy. A scalar loop here against a tuned library there is the
    same reverse-strawman the dream kernel hit at M5; the first version of
    this benchmark measured 6.3x for exactly that reason.
    """
    var best = 0
    var bestv = Float32(-1.0e30)
    var tp = theta.unsafe_ptr()
    var fp = f.unsafe_ptr()
    for a in range(N_ACT):
        var base = a * N_FEAT
        var vacc = SIMD[DType.float32, 8](0)
        var i = 0
        while i + 8 <= N_FEAT:
            vacc += tp.load[width=8](base + i) * fp.load[width=8](i)
            i += 8
        var acc = tp[N_FEAT * N_ACT + a] + vacc.reduce_add()
        while i < N_FEAT:
            acc += tp[base + i] * fp[i]
            i += 1
        if acc > bestv:
            bestv = acc
            best = a
    return best


def greedy_seeker(s: List[Int], wall: List[Bool]) -> Int:
    """Move to whichever free orthogonal cell most reduces Manhattan distance
    to the hider. Full knowledge, deterministic — a fixed, noiseless opponent."""
    var sx = s[8]
    var sy = s[9]
    var hx = s[2]
    var hy = s[3]
    var best = 0                                  # wait
    var bestd = abs(sx - hx) + abs(sy - hy)
    for k in range(4):
        var nx = sx
        var ny = sy
        if k == 0:
            ny += 1
        elif k == 1:
            ny -= 1
        elif k == 2:
            nx += 1
        else:
            nx -= 1
        if occ_s(s, wall, nx, ny):
            continue
        var d = abs(nx - hx) + abs(ny - hy)
        if d < bestd:
            bestd = d
            best = k + 1                          # MOVE_N..MOVE_W are 1..4
    return best


def rollout(theta: List[Float32], wall: List[Bool], seed: Int) raises -> Float32:
    """REAL-sim backend."""
    var w = make_episode(GRID, 6, TURNS, TURNS // 2, 1, 1, 3, 1, seed, 0)
    w.emit_state = False
    w.emit_trace = False
    for _ in range(TURNS):
        var s = state_of(w)
        w.resolve_turn_direct(policy(theta, features(s, wall)),
                              greedy_seeker(s, wall))
    return w.reward()


def rollout_dream(theta: List[Float32], wall: List[Bool], d: Dreamer,
                  seed: Int) raises -> Float32:
    """DREAM backend — open question 7.

    Identical policy, identical seeker, identical featuriser, identical true
    spawn state. The ONLY difference is that the next state comes from the
    learned model instead of the simulator. Reward is read from the dream's own
    predicted `seen` field, so the controller is optimising the world model's
    belief about being seen — which is exactly the quantity a world-model agent
    would be optimising."""
    var w = make_episode(GRID, 6, TURNS, TURNS // 2, 1, 1, 3, 1, seed, 0)
    var s = state_of(w)
    var sa = List[Float32](length=1536, fill=0.0)
    var sb = List[Float32](length=1536, fill=0.0)
    var post = 0
    var unseen = 0
    for t in range(TURNS):
        var a0 = policy(theta, features(s, wall))
        var a1 = greedy_seeker(s, wall)
        s = d.step(s, a0, a1, t, sa, sb)
        if s[1] == 1:
            post += 1
            if s[7] < 0:
                unseen += 1
    if post == 0:
        return 0.0
    return Float32(unseen) / Float32(post)


def fitness(theta: List[Float32], wall: List[Bool], base_seed: Int,
            n_maps: Int) raises -> Float32:
    """Mean reward over a fixed map set — same maps for every candidate, so
    ranking within a generation is not corrupted by map luck."""
    var tot = Float32(0.0)
    for m in range(n_maps):
        tot += rollout(theta, wall, base_seed + m * 7919)
    return tot / Float32(n_maps)


def fitness_dream(theta: List[Float32], wall: List[Bool], d: Dreamer,
                  base_seed: Int, n_maps: Int) raises -> Float32:
    var tot = Float32(0.0)
    for m in range(n_maps):
        tot += rollout_dream(theta, wall, d, base_seed + m * 7919)
    return tot / Float32(n_maps)


# ---------------------------------------------------------------------------
# Separable CMA-ES
# ---------------------------------------------------------------------------
struct SepCMA(Copyable, Movable):
    var n: Int
    var lam: Int
    var mu: Int
    var sigma: Float64
    var mean: List[Float64]
    var c: List[Float64]        # diagonal covariance
    var pc: List[Float64]
    var ps: List[Float64]
    var wts: List[Float64]
    var mueff: Float64
    var cc: Float64
    var cs: Float64
    var c1: Float64
    var cmu: Float64
    var damps: Float64
    var chiN: Float64
    var gen: Int

    def __init__(out self, n: Int, lam: Int, sigma0: Float64):
        self.n = n
        self.lam = lam
        self.mu = lam // 2
        self.sigma = sigma0
        self.mean = List[Float64](length=n, fill=0.0)
        self.c = List[Float64](length=n, fill=1.0)
        self.pc = List[Float64](length=n, fill=0.0)
        self.ps = List[Float64](length=n, fill=0.0)
        self.gen = 0

        var w = List[Float64](length=self.mu, fill=0.0)
        var s = 0.0
        var s2 = 0.0
        for i in range(self.mu):
            w[i] = log(Float64(self.mu) + 0.5) - log(Float64(i + 1))
            s += w[i]
        for i in range(self.mu):
            w[i] /= s
            s2 += w[i] * w[i]
        self.wts = w^
        self.mueff = 1.0 / s2

        var nf = Float64(n)
        self.cc = 4.0 / (nf + 4.0)
        self.cs = (self.mueff + 2.0) / (nf + self.mueff + 3.0)
        # Separable variant: learning rates scaled by (n+2)/3 (Ros & Hansen).
        var base1 = 2.0 / ((nf + 1.3) * (nf + 1.3) + self.mueff)
        var scale = (nf + 2.0) / 3.0
        self.c1 = base1 * scale
        var basemu = 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / (
            (nf + 2.0) * (nf + 2.0) + self.mueff)
        self.cmu = basemu * scale
        if self.c1 + self.cmu > 1.0:
            var t = self.c1 + self.cmu
            self.c1 /= t
            self.cmu /= t
        var extra = sqrt((self.mueff - 1.0) / (nf + 1.0)) - 1.0
        self.damps = 1.0 + self.cs + 2.0 * (extra if extra > 0.0 else 0.0)
        self.chiN = sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf))


def gauss(mut r: LCG) -> Float64:
    """Box-Muller from the shared LCG — no new RNG enters the project."""
    var u1 = (Float64(r.next()) + 1.0) / Float64(1 << 31)
    var u2 = Float64(r.next()) / Float64(1 << 31)
    return sqrt(-2.0 * log(u1)) * cos_approx(6.283185307179586 * u2)


def cos_approx(x: Float64) -> Float64:
    # exp(ix) via the identity cos(x) = sin(x + pi/2); Mojo's std.math has cos,
    # but keeping this explicit avoids an import ambiguity with SIMD overloads.
    var t = x
    while t > 3.141592653589793:
        t -= 6.283185307179586
    while t < -3.141592653589793:
        t += 6.283185307179586
    var t2 = t * t
    # 8th-order Taylor — ample for Box-Muller
    return 1.0 - t2 / 2.0 + t2 * t2 / 24.0 - t2 * t2 * t2 / 720.0 + (
        t2 * t2 * t2 * t2) / 40320.0


def main() raises:
    var args = argv()
    # bench mode: time `reps` fitness evaluations over deterministic thetas.
    # This is the head-to-head against python/evolve_ref.py. It measures the
    # ROLLOUT ENGINE, which is >99% of a search's cost, with parameters that
    # are always finite -- so the comparison cannot be skewed by degenerate
    # controllers behaving differently in the simulator.
    if String(args[1]) == "bench":
        var bw = static_walls(GRID)
        var bbase = Int(String(args[2]))
        var bmaps = Int(String(args[3]))
        var breps = Int(String(args[4]))
        var chk = Float32(0.0)
        var bt0 = perf_counter_ns()
        for r in range(breps):
            var th = List[Float32](length=N_PARAM, fill=0.0)
            for i in range(N_PARAM):
                th[i] = Float32(((i * 37 + r * 13) % 101) - 50) / 100.0
            chk += fitness(th, bw, bbase, bmaps)
        var bt1 = perf_counter_ns()
        var nroll = breps * bmaps
        print("MJ rollouts=", nroll, " turns=", nroll * TURNS,
              " sec=", Float64(Int(bt1 - bt0)) / 1.0e9,
              " checksum=", chk)
        return

    # probe mode: score two fixed controllers and exit. Gives python/evolve_ref.py
    # something exact to verify against before any speed is compared.
    if String(args[1]) == "probe":
        var pw = static_walls(GRID)
        var pbase = Int(String(args[2]))
        var pmaps = Int(String(args[3]))
        var z = List[Float32](length=N_PARAM, fill=0.0)
        var det = List[Float32](length=N_PARAM, fill=0.0)
        for i in range(N_PARAM):
            det[i] = Float32(((i * 37) % 101) - 50) / 100.0
        print("PROBE zero=", fitness(z, pw, pbase, pmaps),
              " det=", fitness(det, pw, pbase, pmaps))
        return

    if len(args) < 6:
        print("usage: evolve <gens> <pop> <maps> <seed> <real|dream> [out.txt]")
        return

    var gens = Int(String(args[1]))
    var lam = Int(String(args[2]))
    var n_maps = Int(String(args[3]))
    var seed = Int(String(args[4]))
    var backend = String(args[5])
    var use_dream = backend == "dream"
    var workers = num_performance_cores()
    var wall = static_walls(GRID)

    # Loaded either way so both runs pay the same startup; only used if dreaming.
    var dre = load_dreamer("data/dynamics.weights.bin", 1536, GRID, TURNS, 2, 4)

    var es = SepCMA(N_PARAM, lam, 0.5)
    var rng = LCG(seed)

    # Held-out maps, disjoint from the training set, for honest reporting.
    var train_base = seed
    var test_base = seed + 777_000_000

    var hist = String("gen best mean sigma test\n")
    var t0 = perf_counter_ns()
    var best_theta = List[Float32](length=N_PARAM, fill=0.0)
    var best_fit = Float32(-1.0)

    for g in range(gens):
        # --- sample -------------------------------------------------------
        var zs = List[List[Float64]]()
        var thetas = List[List[Float32]]()
        for _ in range(lam):
            var z = List[Float64](length=N_PARAM, fill=0.0)
            var th = List[Float32](length=N_PARAM, fill=0.0)
            for i in range(N_PARAM):
                z[i] = gauss(rng)
                th[i] = Float32(es.mean[i] + es.sigma * sqrt(es.c[i]) * z[i])
            zs.append(z^)
            thetas.append(th^)

        # --- evaluate (the part Mojo makes cheap) --------------------------
        var fits = List[Float32](length=lam, fill=0.0)

        @parameter
        def work(k: Int):
            try:
                if use_dream:
                    fits[k] = fitness_dream(thetas[k], wall, dre, train_base, n_maps)
                else:
                    fits[k] = fitness(thetas[k], wall, train_base, n_maps)
            except:
                fits[k] = 0.0

        parallelize[work](lam, workers)

        # --- rank (descending: reward is maximised) ------------------------
        var order = List[Int](length=lam, fill=0)
        for i in range(lam):
            order[i] = i
        for i in range(1, lam):
            var key = order[i]
            var j = i - 1
            while j >= 0 and fits[order[j]] < fits[key]:
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = key

        # --- recombine ------------------------------------------------------
        var old_mean = es.mean.copy()
        var zmean = List[Float64](length=N_PARAM, fill=0.0)
        for i in range(N_PARAM):
            var acc = 0.0
            for k in range(es.mu):
                acc += es.wts[k] * zs[order[k]][i]
            zmean[i] = acc
            es.mean[i] = old_mean[i] + es.sigma * sqrt(es.c[i]) * acc

        # --- step-size and covariance ---------------------------------------
        var psnorm = 0.0
        for i in range(N_PARAM):
            es.ps[i] = (1.0 - es.cs) * es.ps[i] + sqrt(
                es.cs * (2.0 - es.cs) * es.mueff) * zmean[i]
            psnorm += es.ps[i] * es.ps[i]
        psnorm = sqrt(psnorm)

        var hsig = 1.0 if psnorm / sqrt(
            1.0 - (1.0 - es.cs) ** (2.0 * Float64(g + 1))
        ) / es.chiN < 1.4 + 2.0 / (Float64(N_PARAM) + 1.0) else 0.0

        for i in range(N_PARAM):
            es.pc[i] = (1.0 - es.cc) * es.pc[i] + hsig * sqrt(
                es.cc * (2.0 - es.cc) * es.mueff) * sqrt(es.c[i]) * zmean[i]
            var rank_mu = 0.0
            for k in range(es.mu):
                var d = sqrt(es.c[i]) * zs[order[k]][i]
                rank_mu += es.wts[k] * d * d
            es.c[i] = (1.0 - es.c1 - es.cmu) * es.c[i] \
                + es.c1 * es.pc[i] * es.pc[i] \
                + es.cmu * rank_mu
            if es.c[i] < 1e-12:
                es.c[i] = 1e-12
        es.sigma *= exp((es.cs / es.damps) * (psnorm / es.chiN - 1.0))

        # --- report ---------------------------------------------------------
        var mean_fit = Float32(0.0)
        for i in range(lam):
            mean_fit += fits[i]
        mean_fit /= Float32(lam)
        var gbest = fits[order[0]]
        if gbest > best_fit:
            best_fit = gbest
            best_theta = thetas[order[0]].copy()
        # Held-out is always scored in the REAL sim, whichever backend trained.
        # A dream-trained controller that looks great inside the dream and fails
        # in reality is exactly the finding open question 7 is after.
        var test = fitness(thetas[order[0]], wall, test_base, n_maps)

        hist += String(g) + " " + String(gbest) + " " + String(mean_fit)
        hist += " " + String(es.sigma) + " " + String(test) + "\n"
        print("gen", g, " best", gbest, " mean", mean_fit,
              " sigma", es.sigma, " heldout", test)

    var t1 = perf_counter_ns()
    var rollouts = gens * (lam + 1) * n_maps

    # --- baselines, same held-out maps -------------------------------------
    var zero = List[Float32](length=N_PARAM, fill=0.0)
    var b_static = fitness(zero, wall, test_base, n_maps)   # argmax ties -> wait

    var rnd = LCG(seed + 12345)
    var rand_theta = List[Float32](length=N_PARAM, fill=0.0)
    for i in range(N_PARAM):
        rand_theta[i] = Float32(gauss(rnd))
    var b_rand = fitness(rand_theta, wall, test_base, n_maps)
    var b_best = fitness(best_theta, wall, test_base, n_maps)
    # What the dream BELIEVED the champion scores, vs what it really scores.
    var b_in_dream = fitness_dream(best_theta, wall, dre, test_base, n_maps)

    print("")
    print("RESULT gens=", gens, " pop=", lam, " maps=", n_maps,
          " params=", N_PARAM,
          " rollouts=", rollouts,
          " turns=", rollouts * TURNS,
          " ms=", Int(t1 - t0) // 1000000,
          " workers=", workers)
    print("BASELINE backend=", backend, " static=", b_static, " random=", b_rand,
          " evolved_in_REAL=", b_best, " same_controller_scored_in_DREAM=", b_in_dream)

    # What did it actually learn? An action histogram over held-out maps
    # separates "found a corner and sat in it" from "built something".
    var hist_a = List[Int](length=N_ACT, fill=0)
    var locked_end = 0
    var sealed_end = 0
    for m in range(n_maps):
        var w = make_episode(GRID, 6, TURNS, TURNS // 2, 1, 1, 3, 1,
                             test_base + m * 7919, 0)
        w.emit_state = False
        w.emit_trace = False
        for _ in range(TURNS):
            var st = state_of(w)
            var a0 = policy(best_theta, features(st, wall))
            hist_a[a0] += 1
            w.resolve_turn_direct(a0, greedy_seeker(st, wall))
        for i in range(w.n_objs):
            if w.o_locked[i]:
                locked_end += 1
        var hx = w.a_x[0]
        var hy = w.a_y[0]
        if (w.occupied(hx, hy + 1, -1) and w.occupied(hx, hy - 1, -1)
                and w.occupied(hx + 1, hy, -1) and w.occupied(hx - 1, hy, -1)):
            sealed_end += 1
    print("ACTIONS wait=", hist_a[0], " N=", hist_a[1], " S=", hist_a[2],
          " E=", hist_a[3], " W=", hist_a[4], " grab=", hist_a[5],
          " drop=", hist_a[6], " lock=", hist_a[7], " unlock=", hist_a[8],
          " lockN=", hist_a[9], " lockS=", hist_a[10], " lockE=", hist_a[11],
          " lockW=", hist_a[12])
    print("ENDSTATE boxes_locked=", locked_end, " of", n_maps * 3,
          "  hider_sealed=", sealed_end, " of", n_maps)

    var tb = String("")
    for i in range(N_PARAM):
        tb += String(best_theta[i])
        tb += "\n"
    with open("data/best_controller_" + backend + ".txt", "w") as f:
        f.write(tb)

    if len(args) > 6:
        with open(String(args[6]), "w") as f:
            f.write(hist)
