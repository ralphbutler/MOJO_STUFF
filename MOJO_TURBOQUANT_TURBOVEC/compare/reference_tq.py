#!/usr/bin/env python3
"""Independent numpy reference of PLAIN 4-bit TurboQuant (no TQ+).

Mirrors the Mojo algorithm exactly so we can tell whether Mojo has a residual
bug or whether this is simply the quality of the un-calibrated method:
  normalize -> random-orthogonal rotate -> per-coord Gaussian Lloyd-Max
  quantize -> per-vector scale = ||v|| / <u_rot, x_hat> -> asymmetric score.
"""
import json
import os
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
K = 64
BITS = 4


def lloyd_max_gaussian(bits, iters=200):
    """Standard-normal Lloyd-Max levels (closed-form truncated mean)."""
    L = 1 << bits
    c = np.linspace(-3, 3, L)
    for _ in range(iters):
        b = (c[:-1] + c[1:]) / 2
        edges = np.concatenate([[-40.0], b, [40.0]])
        lo, hi = edges[:-1], edges[1:]
        mass = norm.cdf(hi) - norm.cdf(lo)
        mean = norm.pdf(lo) - norm.pdf(hi)
        newc = np.where(mass < 1e-15, c, mean / np.maximum(mass, 1e-300))
        if np.max(np.abs(newc - c)) < 1e-12:
            c = newc
            break
        c = newc
    b = (c[:-1] + c[1:]) / 2
    return b.astype(np.float32), c.astype(np.float32)


def recall_at_1_in_topk(true_top1, pred, k):
    return float(np.mean([true_top1[i] in pred[i, :k] for i in range(len(true_top1))]))


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    dim = meta["dim"]
    base = np.load(os.path.join(DATA, "base.npy"))       # (N, dim) unit
    queries = np.load(os.path.join(DATA, "queries.npy")) # (NQ, dim) unit
    true_top1 = np.fromfile(os.path.join(DATA, "true_top1.i32"), dtype=np.int32)
    N = base.shape[0]

    sigma = 1.0 / np.sqrt(dim)
    b_std, c_std = lloyd_max_gaussian(BITS)
    boundaries = b_std * sigma
    centroids = c_std * sigma

    # random orthogonal rotation (QR of seeded Gaussian)
    rng = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    Q = Q.astype(np.float32)

    t0 = time.time()
    # encode base
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    u = base / np.maximum(norms, 1e-10)
    rot = u @ Q.T                                   # (N, dim), ~N(0,1/dim)
    codes = np.searchsorted(boundaries, rot).astype(np.int32)  # 0..L-1
    xhat = centroids[codes]                         # reconstructed rotated unit
    inner = np.einsum("nd,nd->n", rot, xhat)        # <u_rot, x_hat>
    inner = np.maximum(inner, 1e-10)
    scale = norms[:, 0] / inner                     # (N,)
    print(f"encode {time.time()-t0:.1f}s   mean scale={scale.mean():.3f}")

    # search: score_v(q) = scale_v * <q_rot, xhat_v>
    t0 = time.time()
    qrot = queries @ Q.T                            # (NQ, dim)
    # raw = qrot @ xhat_v^T  -> (NQ, N); do in chunks to bound memory
    NQ = queries.shape[0]
    pred = np.empty((NQ, K), dtype=np.int32)
    CH = 100
    for i in range(0, NQ, CH):
        j = min(i + CH, NQ)
        raw = qrot[i:j] @ xhat.T                    # (chunk, N)
        raw *= scale[None, :]
        # top-K by score, sorted desc
        part = np.argpartition(-raw, K - 1, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(raw, part, axis=1), axis=1)
        pred[i:j] = np.take_along_axis(part, order, axis=1)
    print(f"search {time.time()-t0:.1f}s")

    recalls = {str(k): round(recall_at_1_in_topk(true_top1, pred, k), 4) for k in K_VALUES}
    print("PLAIN 4-bit TurboQuant (numpy reference) recall@1-in-top-k:", recalls)


if __name__ == "__main__":
    main()
