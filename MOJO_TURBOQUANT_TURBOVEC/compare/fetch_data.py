#!/usr/bin/env python3
"""Fetch DBpedia OpenAI-1536 embeddings and export for both FAISS and Mojo.

Mirrors turbovec's benchmark methodology (same dataset/metric/split shape) so
our TurboQuant-in-Mojo and FAISS numbers are directly comparable:
  - 100k database + 1k query vectors from the same corpus
  - both L2-normalized (inner product == cosine)
  - ground truth = exact nearest neighbor (argmax q @ db.T) per query

Writes raw little-endian float32/int32 blobs the Mojo side reads natively:
  data/base.f32      (N*DIM  float32, row-major)
  data/queries.f32   (NQ*DIM float32, row-major)
  data/true_top1.i32 (NQ     int32)
  data/meta.json     ({n, dim, nq})
Also caches data/base.npy + data/queries.npy for the FAISS script.
"""
import json
import os

import numpy as np

DIM = 1536
N = 100_000
NQ = 1_000
SEED = 42
NAME = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
COL = "text-embedding-3-large-1536-embedding"

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)


def load_embeddings(n_needed):
    """Stream the first n_needed rows so we don't pull the full ~6GB corpus."""
    from datasets import load_dataset

    out = np.empty((n_needed, DIM), dtype=np.float32)
    print(f"streaming {n_needed} rows from {NAME} ...")
    ds = load_dataset(NAME, split="train", streaming=True)
    got = 0
    for row in ds:
        out[got] = np.asarray(row[COL], dtype=np.float32)
        got += 1
        if got % 20_000 == 0:
            print(f"  {got}/{n_needed}")
        if got >= n_needed:
            break
    if got < n_needed:
        raise RuntimeError(f"only got {got} rows, needed {n_needed}")
    return out


def main():
    total = N + NQ
    vecs = load_embeddings(total)

    # Deterministic split (our own; internally consistent for FAISS vs Mojo).
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(total)
    base = vecs[perm[:N]].copy()
    queries = vecs[perm[N:N + NQ]].copy()

    # L2-normalize both (cosine == inner product).
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Exact ground-truth nearest neighbor per query.
    print("computing exact ground truth (argmax q @ db.T) ...")
    true_top1 = np.empty(NQ, dtype=np.int32)
    CH = 200  # chunk queries to bound the score matrix memory
    for i in range(0, NQ, CH):
        j = min(i + CH, NQ)
        sims = queries[i:j] @ base.T          # (chunk, N)
        true_top1[i:j] = np.argmax(sims, axis=1).astype(np.int32)

    # Raw blobs for Mojo.
    base.tofile(os.path.join(DATA, "base.f32"))
    queries.tofile(os.path.join(DATA, "queries.f32"))
    true_top1.tofile(os.path.join(DATA, "true_top1.i32"))
    # npy caches for FAISS script.
    np.save(os.path.join(DATA, "base.npy"), base)
    np.save(os.path.join(DATA, "queries.npy"), queries)
    with open(os.path.join(DATA, "meta.json"), "w") as f:
        json.dump({"n": N, "dim": DIM, "nq": NQ, "seed": SEED}, f, indent=2)

    print(f"wrote base.f32 ({base.nbytes/1e6:.0f} MB), queries.f32, true_top1.i32")
    print(f"DIM={DIM} N={N} NQ={NQ}")


if __name__ == "__main__":
    main()
