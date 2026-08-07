# bench_sim.mojo — M2: time Target A over a bundle of episodes.
#
# Parsing and world construction are deliberately OUTSIDE the timed region:
# the benchmark measures simulation, not file I/O or integer parsing. Frame 0
# is emitted by load_world for the same reason (Python's GridWorld.__init__
# emits it at construction and cannot easily defer). Worlds are rebuilt fresh
# every rep because run_all() mutates them.
#
# Two modes, matching python/bench.py exactly:
#   core    emit_trace=False — simulation only; LOS still runs
#   digest  emit_trace=True  — the same, plus the canonical trace text
#
# In digest mode the concatenated trace is written out so the harness can diff
# it against Python's. Correctness is proven by that diff, not by anything
# measured inside the timed region.
#
# Run: uv run mojo bench_sim.mojo <bundle> <reps> <core|digest> [digest-out]

from std.sys import argv, num_performance_cores
from std.time import perf_counter_ns
from std.algorithm import parallelize
from world import World, load_world, read_ints


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: bench_sim <bundle> <reps> <core|digest> [digest-out]")
        return

    var v = read_ints(String(args[1]))
    var reps = Int(String(args[2]))
    var mode = String(args[3])
    var emit = mode == "digest"
    var parallel = mode == "par"
    var n_eps = v[0]
    var workers = num_performance_cores()

    var best = 0
    var total = 0
    var acc = 0
    var frames = 0
    var trace = String("")

    # rep 0 is warmup (page faults, first-touch allocation) and is discarded.
    for rep in range(reps + 1):
        var ws = List[World]()
        var p = 1
        for _ in range(n_eps):
            ws.append(load_world(v, p, emit))

        var t0 = perf_counter_ns()
        if parallel:
            # Episodes are fully independent — no shared state, no reduction.
            # This is the configuration M3 data generation actually uses.
            @parameter
            def work(i: Int):
                ws[i].run_all()

            parallelize[work](n_eps, workers)
        else:
            for ref w in ws:
                w.run_all()
        var t1 = perf_counter_ns()

        if rep == 0:
            acc = 0
            frames = 0
            for ref w in ws:
                acc += w.acc
                frames += w.n_frames
            if emit and len(args) > 4:
                for ref w in ws:
                    trace += w.out
                with open(String(args[4]), "w") as f:
                    f.write(trace)
            continue

        var elapsed = Int(t1 - t0)
        if best == 0 or elapsed < best:
            best = elapsed
        total += elapsed

    print("RESULT lang=mojo mode=", mode,
          " episodes=", n_eps,
          " frames=", frames,
          " workers=", workers if parallel else 1,
          " best_ns=", best,
          " mean_ns=", total // reps,
          " acc=", acc)
