#!/usr/bin/env python3
"""Ryan Codrai's Rust `turbovec` on DBpedia-1536, 4-bit == 768 B/vector.

Same data, same metric (recall@1-in-top-k), same K sweep as faiss_baseline.py
and the Mojo side -- so all three are directly comparable. turbovec is
data-oblivious (like our Mojo impl): no train step, "build" = add + prepare.
"""
import json
import os
import time

import numpy as np
import turbovec

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
K = 64
BITS = 4


def recall_at_1_in_topk(true_top1, pred, k):
    return float(np.mean([true_top1[i] in pred[i, :k] for i in range(len(true_top1))]))


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    dim = meta["dim"]
    base = np.ascontiguousarray(np.load(os.path.join(DATA, "base.npy")), dtype=np.float32)
    queries = np.ascontiguousarray(np.load(os.path.join(DATA, "queries.npy")), dtype=np.float32)
    true_top1 = np.fromfile(os.path.join(DATA, "true_top1.i32"), dtype=np.int32)

    print(f"=== turbovec.TurboQuantIndex(dim={dim}, bit_width={BITS}) ===")
    print(f"    {dim * BITS // 8} bytes/vector (== FAISS PQ, == Mojo 4-bit)")

    index = turbovec.TurboQuantIndex(dim=dim, bit_width=BITS)
    t0 = time.time()
    index.add(base)
    t_add = time.time() - t0
    t0 = time.time()
    index.prepare()
    t_prepare = time.time() - t0
    t0 = time.time()
    _, ids = index.search(queries, K)
    t_search = time.time() - t0
    ids = ids.astype(np.int64)

    recalls = {str(k): round(recall_at_1_in_topk(true_top1, ids, k), 4) for k in K_VALUES}
    t_build = t_add + t_prepare
    print(f"    add {t_add:.1f}s | prepare {t_prepare:.1f}s | build {t_build:.1f}s "
          f"| search {t_search:.2f}s ({1000*t_search/len(queries):.2f} ms/query)")
    print("    recall@1-in-top-k:", recalls)

    out = {
        "engine": "turbovec-rust",
        "bit_width": BITS,
        "bytes_per_vector": dim * BITS // 8,
        "t_add_s": round(t_add, 3),
        "t_prepare_s": round(t_prepare, 3),
        "t_build_s": round(t_build, 3),
        "t_search_s": round(t_search, 3),
        "ms_per_query": round(1000 * t_search / len(queries), 3),
        "recalls": recalls,
    }
    os.makedirs(os.path.join(DATA, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(DATA, "results", "turbovec.json"), "w"), indent=2)
    print("saved data/results/turbovec.json")


if __name__ == "__main__":
    main()
