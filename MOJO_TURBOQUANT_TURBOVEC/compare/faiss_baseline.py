#!/usr/bin/env python3
"""FAISS PQ baseline on DBpedia-1536, equal-memory with 4-bit TurboQuant.

FAISS IndexPQ(dim, m=dim/2, nbits=8, INNER_PRODUCT) stores m*nbits/8 = 768
bytes/vector at dim=1536 -- exactly the same as 4-bit scalar (1536*4/8=768).
Metric: recall@1-in-top-k (true nearest neighbor found within top-k), the same
metric used for the Mojo side, so the two are directly comparable.
"""
import json
import os
import time

import faiss
import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
K = 64


def recall_at_1_in_topk(true_top1, pred, k):
    return float(np.mean([true_top1[i] in pred[i, :k] for i in range(len(true_top1))]))


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    dim = meta["dim"]
    base = np.load(os.path.join(DATA, "base.npy"))
    queries = np.load(os.path.join(DATA, "queries.npy"))
    true_top1 = np.fromfile(os.path.join(DATA, "true_top1.i32"), dtype=np.int32)

    m = dim // 2
    nbits = 8
    print(f"=== FAISS IndexPQ(dim={dim}, m={m}, nbits={nbits}, INNER_PRODUCT) ===")
    print(f"    {m * nbits // 8} bytes/vector (== 4-bit TurboQuant)")

    t0 = time.time()
    index = faiss.IndexPQ(dim, m, nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(base)
    t_train = time.time() - t0
    t0 = time.time()
    index.add(base)
    t_add = time.time() - t0
    t0 = time.time()
    _, ids = index.search(queries, K)
    t_search = time.time() - t0

    recalls = {str(k): round(recall_at_1_in_topk(true_top1, ids, k), 4) for k in K_VALUES}
    print(f"    train {t_train:.1f}s | add {t_add:.1f}s | search {t_search:.2f}s "
          f"({1000*t_search/len(queries):.2f} ms/query)")
    print("    recall@1-in-top-k:", recalls)

    out = {
        "engine": "faiss-IndexPQ",
        "bytes_per_vector": m * nbits // 8,
        "t_train_s": round(t_train, 3),
        "t_add_s": round(t_add, 3),
        "t_search_s": round(t_search, 3),
        "recalls": recalls,
    }
    os.makedirs(os.path.join(DATA, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "results", "faiss.json"), "w"), indent=2)
    print("saved data/results/faiss.json")


if __name__ == "__main__":
    main()
