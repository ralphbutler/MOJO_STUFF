# gridworld.mojo — single-episode runner, used by the M1 parity gate.
#
# Reads one spec, runs it with trace emission on, writes the digest.
# The simulation itself lives in world.mojo.
#
# Run: uv run mojo mojo/gridworld.mojo <spec-in> <digest-out>

from std.sys import argv
from world import World, load_world, read_ints


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: gridworld <spec-in> <digest-out>")
        return

    var v = read_ints(String(args[1]))
    var p = 0
    var w = load_world(v, p, True)
    w.run_all()

    with open(String(args[2]), "w") as f:
        f.write(w.out)
    print("frames=", w.n_frames)
