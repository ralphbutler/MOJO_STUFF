#!/usr/bin/env python3
"""Print the head-to-head recall table: TurboQuant (Mojo) vs FAISS PQ."""
import json
import os

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "data", "results"))
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    mojo = load("mojo.json")
    faiss = load("faiss.json")
    rust = load("turbovec.json")

    print("\n  DBpedia OpenAI-1536 · 100k base / 1k queries · 4-bit == 768 B/vec")
    print("  recall@1-in-top-k (true nearest neighbor found within top-k)\n")
    hdr = "  {:<22}".format("k") + "".join(f"{k:>8}" for k in K_VALUES)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    engines = (
        ("TurboQuant (Mojo)", mojo),
        ("turbovec (Rust)", rust),
        ("FAISS IndexPQ", faiss),
    )
    for label, res in engines:
        if res is None:
            print(f"  {label:<22} (no results yet)")
            continue
        row = "  {:<22}".format(label) + "".join(
            f"{res['recalls'][str(k)]:>8.4f}" for k in K_VALUES
        )
        print(row)
    if mojo and faiss:
        print("  " + "-" * (len(hdr) - 2))
        delta = "  {:<22}".format("Δ (Mojo − FAISS)") + "".join(
            f"{mojo['recalls'][str(k)] - faiss['recalls'][str(k)]:>+8.4f}" for k in K_VALUES
        )
        print(delta)
    if mojo and rust:
        delta = "  {:<22}".format("Δ (Mojo − Rust)") + "".join(
            f"{mojo['recalls'][str(k)] - rust['recalls'][str(k)]:>+8.4f}" for k in K_VALUES
        )
        print(delta)
    print()

    # Speed / build summary
    def ms(res):
        if not res:
            return "-"
        if "ms_per_query" in res:
            return f"{res['ms_per_query']:.2f}"
        if "t_search_s" in res:
            return f"{1000 * res['t_search_s'] / 1000:.2f}"
        return "-"

    def build_s(res):
        if not res:
            return None
        if "t_build_s" in res:
            return res["t_build_s"]
        if "t_train_s" in res or "t_add_s" in res:
            return res.get("t_train_s", 0) + res.get("t_add_s", 0)
        return None

    print("  {:<22}{:>12}{:>12}".format("", "search ms/q", "build s"))
    print("  " + "-" * 44)
    for label, res in engines:
        if res is None:
            continue
        b = build_s(res)
        bstr = f"{b:.1f}" if b is not None else "-"
        print("  {:<22}{:>12}{:>12}".format(label, ms(res), bstr))
    print()


if __name__ == "__main__":
    main()
