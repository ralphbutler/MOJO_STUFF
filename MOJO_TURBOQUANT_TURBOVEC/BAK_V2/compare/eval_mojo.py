#!/usr/bin/env python3
"""Score the Mojo TurboQuant output with the same recall@1-in-top-k metric."""
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
K = 64


def recall_at_1_in_topk(true_top1, pred, k):
    return float(np.mean([true_top1[i] in pred[i, :k] for i in range(len(true_top1))]))


def main():
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    nq, dim = meta["nq"], meta["dim"]
    true_top1 = np.fromfile(os.path.join(DATA, "true_top1.i32"), dtype=np.int32)
    pred = np.fromfile(os.path.join(DATA, "results", "mojo_topk.i32"), dtype=np.int32)
    pred = pred.reshape(nq, K)

    recalls = {str(k): round(recall_at_1_in_topk(true_top1, pred, k), 4) for k in K_VALUES}
    bytes_per_vector = dim * 4 // 8  # 4-bit
    print("=== TurboQuant (Mojo) ===")
    print(f"    {bytes_per_vector} bytes/vector (4-bit)")
    print("    recall@1-in-top-k:", recalls)

    out = {"engine": "mojo-TurboQuant", "bytes_per_vector": bytes_per_vector, "recalls": recalls}
    json.dump(out, open(os.path.join(DATA, "results", "mojo.json"), "w"), indent=2)
    print("saved data/results/mojo.json")


if __name__ == "__main__":
    main()
