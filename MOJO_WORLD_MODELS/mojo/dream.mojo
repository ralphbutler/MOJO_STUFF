# dream.mojo — M5/Target B: run the learned dynamics model as a simulator.
#
# The world model's payoff: sever the real environment and step the LEARNED
# model instead. Closed loop — each predicted state becomes the next input, so
# errors compound rather than being reset by ground truth every step.
#
# This is the small-sequential regime the whole project is about: a 208->1536->
# 1536->1536->180 MLP stepped one state at a time, millions of times. Tiny
# matmuls where per-call dispatch overhead, not FLOPs, is the cost.
#
# Weights come from python/train.py's export: flat float32, per Linear a
# weight [out,in] row-major then bias [out]. No parser needed (same principle
# as M1's keyword-free .spec files); shapes are hard-passed from the caller.
#
# Run: uv run mojo dream.mojo <weights.bin> <bundle> <steps> <mode>
#   mode = verify  -> dump predicted states for the Python gate
#          bench    -> time closed-loop rollouts, no output

from std.sys import argv, num_performance_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize
from std.math import exp

comptime N_ACTIONS = 13
comptime N_LAYERS = 4
comptime W = 16                # SIMD lanes for the dot product


struct Net(Copyable, Movable):
    """The exported MLP. Flat weight storage, one contiguous buffer."""

    var w: List[List[Float32]]      # per layer, [out*in] row-major
    var b: List[List[Float32]]
    var dims_in: List[Int]
    var dims_out: List[Int]

    def __init__(out self):
        self.w = []
        self.b = []
        self.dims_in = []
        self.dims_out = []

    def forward(self, x: List[Float32], mut scratch_a: List[Float32],
                mut scratch_b: List[Float32]) -> List[Float32]:
        """ReLU MLP, no allocation in the loop — scratch buffers are reused."""
        var cur = x.copy()
        for L in range(len(self.dims_in)):
            var n_in = self.dims_in[L]
            var n_out = self.dims_out[L]
            var out = List[Float32](length=n_out, fill=0.0)
            var wp = self.w[L].unsafe_ptr()
            var cp = cur.unsafe_ptr()
            var last = L == len(self.dims_in) - 1
            for o in range(n_out):
                # SIMD-widened dot product. Comparing a naive scalar loop
                # against tuned BLAS would be a strawman in reverse.
                var base = o * n_in
                var vacc = SIMD[DType.float32, W](0)
                var i = 0
                while i + W <= n_in:
                    vacc += wp.load[width=W](base + i) * cp.load[width=W](i)
                    i += W
                var acc = self.b[L][o] + vacc.reduce_add()
                while i < n_in:
                    acc += wp[base + i] * cp[i]
                    i += 1
                out[o] = acc if (last or acc > 0.0) else 0.0
            cur = out^
        return cur^


struct Field(ImplicitlyCopyable, Movable):
    var col: Int
    var classes: Int
    var shift: Int

    def __init__(out self, col: Int, classes: Int, shift: Int):
        self.col = col
        self.classes = classes
        self.shift = shift


struct Dreamer(Copyable, Movable):
    var net: Net
    var fields: List[Field]
    var grid: Int
    var turns: Int
    var prep_turns: Int
    var n_agents: Int
    var n_objs: Int
    var state_dim: Int
    var in_dim: Int

    def __init__(out self):
        self.net = Net()
        self.fields = []
        self.grid = 0
        self.turns = 0
        self.prep_turns = 0
        self.n_agents = 0
        self.n_objs = 0
        self.state_dim = 0
        self.in_dim = 0

    def build_fields(mut self):
        """Must match python/model.py::Spec exactly — same order, same shifts."""
        var col = 2
        for _ in range(self.n_agents):
            self.fields.append(Field(col + 0, self.grid, 0))
            self.fields.append(Field(col + 1, self.grid, 0))
            self.fields.append(Field(col + 2, 3, 1))
            self.fields.append(Field(col + 3, 3, 1))
            self.fields.append(Field(col + 4, self.n_objs + 1, 1))
            self.fields.append(Field(col + 5, self.n_agents + 1, 1))
            col += 6
        for _ in range(self.n_objs):
            self.fields.append(Field(col + 0, self.grid, 0))
            self.fields.append(Field(col + 1, self.grid, 0))
            self.fields.append(Field(col + 2, 2, 0))
            col += 3
        self.state_dim = col

    def encode(self, s: List[Int], a0: Int, a1: Int) -> List[Float32]:
        var x = List[Float32](length=self.in_dim, fill=0.0)
        x[0] = Float32(s[0]) / Float32(self.turns)     # turn, normalised
        x[1] = Float32(s[1])                           # phase
        var p = 2
        for f in range(len(self.fields)):
            var fl = self.fields[f]
            var v = s[fl.col] + fl.shift
            if v >= 0 and v < fl.classes:
                x[p + v] = 1.0
            p += fl.classes
        x[p + a0] = 1.0
        p += N_ACTIONS
        x[p + a1] = 1.0
        return x^

    def step(self, s: List[Int], a0: Int, a1: Int, turn_next: Int,
             mut sa: List[Float32], mut sb: List[Float32]) -> List[Int]:
        """One dream step: (state, joint action) -> predicted next state.

        turn_next is PASSED IN, not derived as s[0]+1. GOTCHA: the sim emits a
        frame BEFORE incrementing turn, so frame k carries turn k-1 and frames
        0 and 1 both have turn 0. Incrementing is right at every step but the
        first — enough to diverge a closed loop at step 0 and never recover.
        At dream step t the produced frame is t+1, whose turn is exactly t."""
        var logits = self.net.forward(self.encode(s, a0, a1), sa, sb)
        var out = List[Int](length=self.state_dim, fill=0)
        # turn and phase are deterministic — not predicted (see model.py).
        out[0] = turn_next
        out[1] = 1 if turn_next >= self.prep_turns else 0
        var off = 0
        for f in range(len(self.fields)):
            var fl = self.fields[f]
            var best = 0
            var bestv = logits[off]
            for c in range(1, fl.classes):
                if logits[off + c] > bestv:
                    bestv = logits[off + c]
                    best = c
            out[fl.col] = best - fl.shift
            off += fl.classes
        return out^


def read_f32(path: String, count: Int) raises -> List[Float32]:
    var raw = open(path, "r").read_bytes()
    var out = List[Float32](length=count, fill=0.0)
    var p = UnsafePointer(to=raw[0]).bitcast[Float32]()
    for i in range(count):
        out[i] = p[i]
    return out^


def read_ints(path: String) raises -> List[Int]:
    var text = open(path, "r").read()
    var toks = text.split()
    var vals = List[Int]()
    for i in range(len(toks)):
        vals.append(Int(String(toks[i])))
    return vals^


def load_dreamer(weights: String, hidden: Int, grid: Int, turns: Int,
                 n_agents: Int, n_objs: Int) raises -> Dreamer:
    """Build a Dreamer from an exported M4 weight file. Factored out of main so
    M8's evolution harness can use the dream as a rollout backend."""
    var d = Dreamer()
    d.grid = grid
    d.turns = turns
    d.prep_turns = turns // 2
    d.n_agents = n_agents
    d.n_objs = n_objs
    d.build_fields()

    var out_dim = 0
    for f in range(len(d.fields)):
        out_dim += d.fields[f].classes
    d.in_dim = 2 + out_dim + n_agents * N_ACTIONS

    var dims_in = [d.in_dim, hidden, hidden, hidden]
    var dims_out = [hidden, hidden, hidden, out_dim]
    var total = 0
    for L in range(N_LAYERS):
        total += dims_in[L] * dims_out[L] + dims_out[L]

    var flat = read_f32(weights, total)
    var p = 0
    for L in range(N_LAYERS):
        var nw = dims_in[L] * dims_out[L]
        var w = List[Float32](length=nw, fill=0.0)
        for i in range(nw):
            w[i] = flat[p + i]
        p += nw
        var b = List[Float32](length=dims_out[L], fill=0.0)
        for i in range(dims_out[L]):
            b[i] = flat[p + i]
        p += dims_out[L]
        d.net.w.append(w^)
        d.net.b.append(b^)
        d.net.dims_in.append(dims_in[L])
        d.net.dims_out.append(dims_out[L])
    return d^


def main() raises:
    var args = argv()
    if len(args) < 5:
        print("usage: dream <weights.bin> <bundle> <steps> <verify|bench> [out]")
        return

    # Shapes are fixed by the M4 export; the caller passes the episode data.
    var hidden = 1536
    var grid = 12
    var turns = 100
    var n_agents = 2
    var n_objs = 4

    var d = Dreamer()
    d.grid = grid
    d.turns = turns
    d.prep_turns = turns // 2
    d.n_agents = n_agents
    d.n_objs = n_objs
    d.build_fields()

    var out_dim = 0
    for f in range(len(d.fields)):
        out_dim += d.fields[f].classes
    d.in_dim = 2 + out_dim + n_agents * N_ACTIONS

    var dims_in = [d.in_dim, hidden, hidden, hidden]
    var dims_out = [hidden, hidden, hidden, out_dim]
    var total = 0
    for L in range(N_LAYERS):
        total += dims_in[L] * dims_out[L] + dims_out[L]

    var flat = read_f32(String(args[1]), total)
    var p = 0
    for L in range(N_LAYERS):
        var nw = dims_in[L] * dims_out[L]
        var w = List[Float32](length=nw, fill=0.0)
        for i in range(nw):
            w[i] = flat[p + i]
        p += nw
        var b = List[Float32](length=dims_out[L], fill=0.0)
        for i in range(dims_out[L]):
            b[i] = flat[p + i]
        p += dims_out[L]
        d.net.w.append(w^)
        d.net.b.append(b^)
        d.net.dims_in.append(dims_in[L])
        d.net.dims_out.append(dims_out[L])

    # Bundle: n_eps, then per episode  state_dim ints (start state)
    # followed by steps*2 action codes.
    var v = read_ints(String(args[2]))
    var steps = Int(String(args[3]))
    var mode = String(args[4])
    var n_eps = v[0]
    var workers = num_performance_cores()

    var starts = List[List[Int]]()
    var acts = List[List[Int]]()
    var q = 1
    for _ in range(n_eps):
        var s = List[Int]()
        for i in range(d.state_dim):
            s.append(v[q + i])
        q += d.state_dim
        var a = List[Int]()
        for i in range(steps * 2):
            a.append(v[q + i])
        q += steps * 2
        starts.append(s^)
        acts.append(a^)

    var results = List[List[Int]]()
    for _ in range(n_eps):
        results.append(List[Int]())

    var t0 = perf_counter_ns()

    @parameter
    def work(e: Int):
        var sa = List[Float32](length=hidden, fill=0.0)
        var sb = List[Float32](length=hidden, fill=0.0)
        var s = starts[e].copy()
        var rec = List[Int]()
        for t in range(steps):
            s = d.step(s, acts[e][t * 2], acts[e][t * 2 + 1], t, sa, sb)
            if mode == "verify":
                for i in range(d.state_dim):
                    rec.append(s[i])
        results[e] = rec^

    parallelize[work](n_eps, workers)
    var t1 = perf_counter_ns()

    if mode == "verify" and len(args) > 5:
        var txt = String("")
        for e in range(n_eps):
            for i in range(len(results[e])):
                txt += String(results[e][i])
                txt += " " if (i + 1) % d.state_dim != 0 else "\n"
        with open(String(args[5]), "w") as f:
            f.write(txt)

    print("DREAM episodes=", n_eps, " steps=", steps,
          " total_steps=", n_eps * steps,
          " workers=", workers,
          " ns=", Int(t1 - t0),
          " us_per_step=", Float64(Int(t1 - t0)) / 1000.0 / Float64(n_eps * steps))
