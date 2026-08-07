"""M2 — Target A benchmark: the gridworld sim, Mojo vs Python, CPU vs CPU.

Protocol (PLAN.md "Benchmark protocol"):
  * Device pinned and printed. Mojo is CPU here, so Python is CPU. No MPS.
  * World construction and spec parsing are OUTSIDE the timed region on both
    sides; only simulation is timed. Frame 0 is emitted at construction on both
    sides for the same reason.
  * Warmup rep discarded; `best` reported (least noisy), `mean` shown alongside.
  * Correctness is proven by DIFFING the digest text both languages produce —
    not by anything computed inside the timed region.

Two modes:
  core    simulation only. LOS still runs; the result is summed into `acc` so
          neither compiler can eliminate it. One integer add per agent per
          frame. (An earlier version folded every field into an FNV checksum
          here. In Python that was 27 function calls per frame, which made
          core SLOWER than digest and inflated Mojo's apparent win. Don't put
          instrumentation in the hot loop.)
  digest  the same, plus the canonical trace text.

Three Python rows, because "which Python?" is the fairness question:
  core        the fairest pairing against Mojo core
  digest      formats the canonical digest text directly
  as-written  the vendored HIDE_SEEK code, building a frame dict per tick

Note on scope: action-code -> action-dict translation (`decode`) is inside
Python's timed region and the equivalent integer unpacking is inside Mojo's.
That dict allocation is a genuine cost of the Python sim's API, not harness
overhead bolted on for effect.

    python python/bench.py --episodes 300 --turns 100 --reps 5
"""
from __future__ import annotations

import argparse
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import episode as ep
from gridworld import GridWorld

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "bench_sim"


class CoreGrid(GridWorld):
    """Simulate only. seen_map() still runs; `acc` keeps it from being dead."""

    def __init__(self, cfg, rng):
        self.acc = 0
        self.n_frames = 0
        super().__init__(cfg, rng)

    def emit_frame(self):
        seen = self.seen_map()
        acc = self.acc
        for a in self.agents:
            if a.team == "hider" and seen.get(a.id, (False, None))[0]:
                acc += 1
        self.acc = acc
        self.n_frames += 1


class DigestGrid(GridWorld):
    """Format the canonical digest line directly.

    Deliberately skips the vendored emit_frame's dict construction, so this
    row measures the same job Mojo's digest mode does rather than Python doing
    it twice.
    """

    def __init__(self, cfg, rng):
        self.parts = []
        self.n_frames = 0
        self._idx = None
        self._oidx = None
        super().__init__(cfg, rng)

    def emit_frame(self):
        if self._idx is None:
            self._idx = {a.id: i for i, a in enumerate(self.agents)}
            self._oidx = {o.id: i for i, o in enumerate(self.objs)}
        seen = self.seen_map()
        ag = []
        for a in self.agents:
            sb = -1
            if a.team == "hider":
                s, by = seen.get(a.id, (False, None))
                if s:
                    sb = self._idx[by]
            c = self._oidx[a.carry] if a.carry else -1
            ag.append("{},{},{},{},{},{}".format(
                a.pos[0], a.pos[1], a.face[0], a.face[1], c, sb))
        ob = ["{},{},{}".format(o.pos[0], o.pos[1], int(o.locked))
              for o in self.objs]
        self.parts.append("{},{},{}|{}|{}".format(
            self.n_frames, self.turn,
            0 if self.turn < self.prep_turns else 1,
            ";".join(ag), ";".join(ob)))
        self.n_frames += 1


def gen_codes(seed: int, n_agents: int, turns: int) -> list[list[int]]:
    arng = random.Random(seed ^ 0x5EED)
    return [arng.choices(range(ep.N_ACTIONS), weights=ep._WEIGHTS, k=n_agents)
            for _ in range(turns)]


def drive(w: GridWorld, codes: list[list[int]]) -> None:
    """The timed region. Mirrors Mojo's run_all()."""
    for turn_codes in codes:
        w.resolve_turn({a.id: ep.decode(c, a.pos)
                        for a, c in zip(w.agents, turn_codes)})


def time_python(cls, cfg, n_eps: int, reps: int):
    best, total, frames, text = 0, 0, 0, ""
    for rep in range(reps + 1):
        worlds = []
        for seed in range(n_eps):
            w = cls(cfg, random.Random(seed))
            worlds.append((w, gen_codes(seed, len(w.agents), cfg["sim"]["turns"])))

        t0 = time.perf_counter_ns()
        for w, codes in worlds:
            drive(w, codes)
        t1 = time.perf_counter_ns()

        if rep == 0:
            frames = sum(getattr(w, "n_frames", len(w.frames)) for w, _ in worlds)
            if hasattr(worlds[0][0], "parts"):
                text = "".join("\n".join(w.parts) + "\n" for w, _ in worlds)
            continue
        el = t1 - t0
        best = el if best == 0 else min(best, el)
        total += el
    return best, total // reps, frames, text


def build_bundle(cfg, n_eps: int, path: Path) -> None:
    ints = [n_eps]
    for seed in range(n_eps):
        w = GridWorld(cfg, random.Random(seed))
        codes = gen_codes(seed, len(w.agents), cfg["sim"]["turns"])
        ints += [w.n, w.vision, w.move_budget, w.prep_turns, w.total_turns,
                 len(w.agents), len(w.objs), len(w.walls), len(codes)]
        for a in w.agents:
            ints += [0 if a.team == "hider" else 1, a.pos[0], a.pos[1]]
        for o in w.objs:
            ints += [0 if o.kind == "box" else 1, o.pos[0], o.pos[1]]
        for x, y in sorted(w.walls):
            ints += [x, y]
        for tc in codes:
            ints += tc
    path.write_text(" ".join(map(str, ints)) + "\n")


def run_mojo(bundle: Path, reps: int, mode: str, out: Path | None = None) -> dict:
    cmd = [str(BIN), str(bundle), str(reps), mode]
    if out:
        cmd.append(str(out))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("mojo bench failed:\n" + r.stdout + r.stderr)
    line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))
    # Mojo's print() inserts spaces around arguments; re-pair the tokens.
    return dict(t.split("=", 1) for t in line.replace("= ", "=").split() if "=" in t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--grid", type=int, default=12)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()

    cfg = ep.make_cfg(grid=a.grid, turns=a.turns)
    ROOT.joinpath("build").mkdir(exist_ok=True)
    r = subprocess.run(["uv", "run", "mojo", "build", "mojo/bench_sim.mojo",
                        "-o", str(BIN)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("mojo build failed:\n" + r.stdout + r.stderr)

    bundle = ROOT / "data" / "bench.bundle"
    mj_trace = ROOT / "data" / "bench.mojo.digest"
    build_bundle(cfg, a.episodes, bundle)

    py_core = time_python(CoreGrid, cfg, a.episodes, a.reps)
    py_dig = time_python(DigestGrid, cfg, a.episodes, a.reps)
    py_raw = time_python(GridWorld, cfg, a.episodes, a.reps)
    mj_core = run_mojo(bundle, a.reps, "core")
    mj_par = run_mojo(bundle, a.reps, "par")
    mj_dig = run_mojo(bundle, a.reps, "digest", mj_trace)

    # Parallel must produce the same accumulator as serial, or the threads
    # raced / dropped work.
    if mj_par["acc"] != mj_core["acc"]:
        sys.exit("PARALLEL MISMATCH: acc {} vs serial {}".format(
            mj_par["acc"], mj_core["acc"]))

    # --- correctness: the two sims must have produced identical traces -----
    if mj_trace.read_text() != py_dig[3]:
        sys.exit("TRACE MISMATCH — the two sims did different work. "
                 "Run python/parity.py to localise.")
    if int(mj_dig["frames"]) != py_dig[2]:
        sys.exit("FRAME COUNT MISMATCH: mojo {} vs python {}".format(
            mj_dig["frames"], py_dig[2]))

    turns_total = a.episodes * a.turns

    def row(name, ns):
        return "  {:34s} {:>10.2f} ms {:>11.3f} us/turn".format(
            name, ns / 1e6, ns / 1e3 / turns_total)

    def speed(name, num, den):
        return "  {:34s} {:>10.1f}x".format(name, num / den)

    print("=" * 74)
    print("M2 — TARGET A BENCHMARK   {} episodes x {} turns, grid {}  ({:,} turns)".format(
        a.episodes, a.turns, a.grid, turns_total))
    print("device: CPU both sides ({} {})   reps: {}, warmup discarded, best shown".format(
        platform.machine(), platform.system(), a.reps))
    print("correctness: {:,} frames of trace text identical across languages".format(
        py_dig[2]))
    print("=" * 74)
    print("\nCORE  (simulation only — LOS runs, no trace produced)")
    print(row("Python", py_core[0]))
    print(row("Mojo, 1 thread", int(mj_core["best_ns"])))
    print(row("Mojo, {} threads".format(mj_par["workers"]), int(mj_par["best_ns"])))
    print(speed("--> speedup, 1 thread", py_core[0], int(mj_core["best_ns"])))
    print(speed("--> speedup, {} threads".format(mj_par["workers"]),
                py_core[0], int(mj_par["best_ns"])))
    print("  {:34s} {:>10.1f}x of {}x ideal".format(
        "--> parallel efficiency",
        int(mj_core["best_ns"]) / int(mj_par["best_ns"]), mj_par["workers"]))
    print("\nDIGEST  (simulation + canonical trace text)")
    print(row("Python", py_dig[0]))
    print(row("Mojo", int(mj_dig["best_ns"])))
    print(speed("--> speedup", py_dig[0], int(mj_dig["best_ns"])))
    print("\nREFERENCE  (the vendored HIDE_SEEK code, unmodified)")
    print(row("Python as-written (frame dicts)", py_raw[0]))
    print(speed("--> vs Mojo digest", py_raw[0], int(mj_dig["best_ns"])))
    print()


if __name__ == "__main__":
    main()
