"""M3 coverage report — does the generated dataset actually contain what M4 needs?

Volume is easy; coverage is the thing that can silently fail. A random walker
covers the grid well but rarely builds a fort, and a dynamics model that never
saw a sealed corner cannot dream one. This quantifies that rather than assuming
it, and is the gate PLAN.md requires for M3.

Also the loader M4 will import: `load(prefix)` returns memmapped arrays plus
the meta dict, so a 500 MB dataset costs nothing to open.

    python python/dataset.py data/train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# State vector column offsets (see meta["state_layout"]).
TURN, PHASE = 0, 1
AG = 2                     # per agent: x, y, fx, fy, carry, seen_by
AG_W = 6


def load(prefix: str):
    meta = json.loads(Path(prefix + ".meta.json").read_text())
    s = np.memmap(prefix + ".states.bin", dtype=meta["dtype"], mode="r"
                  ).reshape(meta["states_shape"])
    a = np.memmap(prefix + ".actions.bin", dtype=meta["dtype"], mode="r"
                  ).reshape(meta["actions_shape"])
    return s, a, meta


def wall_mask(n: int) -> np.ndarray:
    """Reconstruct the static layout: border ring + 3-cell centre stub."""
    w = np.zeros((n, n), dtype=bool)
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
    mid = n // 2
    for dy in (-1, 0, 1):
        w[mid, mid + dy] = True
    return w


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    s, a, meta = load(prefix)
    n_eps, n_frames, dim = s.shape
    n_ag = meta["n_agents"]
    n_obj = meta["n_boxes"] + meta["n_ramps"]
    grid = meta["grid"]
    OB = AG + n_ag * AG_W          # first object column

    flat = np.asarray(s).reshape(-1, dim)
    total = flat.shape[0]

    hx, hy = flat[:, AG + 0], flat[:, AG + 1]
    hcarry, hseen = flat[:, AG + 4], flat[:, AG + 5]
    sx, sy = flat[:, AG + AG_W + 0], flat[:, AG + AG_W + 1]

    locked = np.stack([flat[:, OB + 3 * i + 2] for i in range(n_obj)], axis=1)
    n_locked = locked.sum(axis=1)

    # --- fort detection: every orthogonal neighbour of the hider blocked ----
    walls = wall_mask(grid)
    blocked = np.zeros((total, 4), dtype=bool)
    box_xy = [(flat[:, OB + 3 * i], flat[:, OB + 3 * i + 1])
              for i in range(meta["n_boxes"])]
    for k, (dx, dy) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        nx, ny = hx + dx, hy + dy
        inb = (nx >= 0) & (nx < grid) & (ny >= 0) & (ny < grid)
        b = np.zeros(total, dtype=bool)
        b[inb] = walls[nx[inb], ny[inb]]
        for bx, by in box_xy:
            b |= (bx == nx) & (by == ny)
        blocked[:, k] = b | ~inb
    sealed = blocked.all(axis=1)

    free_cells = int((~walls).sum())
    hcells = len(set(zip(hx.tolist(), hy.tolist())))
    acts, acounts = np.unique(np.asarray(a).reshape(-1), return_counts=True)

    # per-arm slices
    arms = meta["arm_counts"]
    bounds, off = {}, 0
    for k, v in arms.items():
        bounds[k] = (off, off + v)
        off += v

    def pct(x):
        return 100.0 * float(x) / total

    print("=" * 68)
    print("M3 DATASET COVERAGE — {}".format(prefix))
    print("=" * 68)
    print("episodes {:,}   frames {:,}   state_dim {}   generated in {} ms".format(
        n_eps, total, dim, meta["gen_ms"]))
    print("mix: " + ", ".join("{} {:,}".format(k, v) for k, v in arms.items()))
    print("on disk: {:.0f} MB states + {:.0f} MB actions".format(
        s.nbytes / 1e6, a.nbytes / 1e6))

    print("\nSTATE-SPACE COVERAGE")
    print("  hider cells visited        {:>4d} / {:<4d} free  ({:.1f}%)".format(
        hcells, free_cells, 100.0 * hcells / free_cells))
    print("  distinct full states       {:,} of {:,} frames ({:.1f}% unique)".format(
        len(np.unique(flat, axis=0)), total,
        100.0 * len(np.unique(flat, axis=0)) / total))

    print("\nFORT-SHAPED STATES  (the coverage that random play misses)")
    print("  frames with >=1 box locked {:>12,}  ({:.1f}%)".format(
        int((n_locked >= 1).sum()), pct((n_locked >= 1).sum())))
    print("  frames with >=2 boxes      {:>12,}  ({:.1f}%)".format(
        int((n_locked >= 2).sum()), pct((n_locked >= 2).sum())))
    print("  frames with  3 boxes       {:>12,}  ({:.1f}%)".format(
        int((n_locked >= 3).sum()), pct((n_locked >= 3).sum())))
    print("  HIDER FULLY SEALED         {:>12,}  ({:.1f}%)".format(
        int(sealed.sum()), pct(sealed.sum())))
    seal2d = sealed.reshape(n_eps, n_frames)
    for k, (lo, hi) in bounds.items():
        m = seal2d[lo:hi]
        print("    via {:16s} {:>12,}  ({:.1f}% of arm)".format(
            k, int(m.sum()), 100.0 * m.mean()))
    # Transitions INTO a seal. Sealed states alone are not enough: without
    # these the model never sees the lock that completes a fort, which is
    # exactly what B2 would need it to dream.
    completions = (~seal2d[:, :-1]) & seal2d[:, 1:]
    print("  seal COMPLETIONS (unsealed->sealed) {:>7,}  in {:,} episodes".format(
        int(completions.sum()), int(completions.any(axis=1).sum())))

    print("\nDYNAMICS")
    print("  hider seen by seeker       {:>12,}  ({:.1f}%)".format(
        int((hseen >= 0).sum()), pct((hseen >= 0).sum())))
    print("  hider carrying an object   {:>12,}  ({:.1f}%)".format(
        int((hcarry >= 0).sum()), pct((hcarry >= 0).sum())))
    changed = (np.asarray(s)[:, 1:, :] != np.asarray(s)[:, :-1, :]).any(axis=2)
    print("  transitions changing state {:>12,}  ({:.1f}%)".format(
        int(changed.sum()), 100.0 * changed.mean()))

    print("\nACTION DISTRIBUTION")
    names = ["wait", "move_N", "move_S", "move_E", "move_W", "grab", "drop",
             "lock", "unlock", "lock_N", "lock_S", "lock_E", "lock_W"]
    for code, cnt in zip(acts.tolist(), acounts.tolist()):
        print("  {:<8s} {:>12,}  ({:.1f}%)".format(
            names[code], cnt, 100.0 * cnt / acounts.sum()))

    missing = [n for n in range(13) if n not in acts.tolist()]
    if missing:
        sys.exit("\nGATE FAILED: actions never issued: {}".format(missing))
    if sealed.sum() == 0:
        sys.exit("\nGATE FAILED: no fort-shaped states — arm B is not working.")
    print("\n✅ M3 GATE PASSED — every action fires and fort states are present.")


if __name__ == "__main__":
    main()
