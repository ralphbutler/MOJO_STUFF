"""CPU vs PyTorch-MPS matmul benchmark — the framework-tier yardstick for 03*.

Match N to the Mojo matmul files (03a_matmul_naive.mojo etc., default N=2048)
for a fair comparison: your hand-written Mojo GPU kernel vs PyTorch's shipped one.
Run from an env that has torch (e.g. your BASE env), NOT the mojo-learning venv
(the MOJO_CURRICULUM/.venv holds MAX, not torch):

    python bench_torch.py            # N=2048, 50 iters (matches 03*)
    python bench_torch.py 1024 50    # custom size / iters
"""

import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
DTYPE = torch.float32


def bench(device_str: str):
    device = torch.device(device_str)
    a = torch.randn(N, N, dtype=DTYPE, device=device)
    b = torch.randn(N, N, dtype=DTYPE, device=device)

    # Warmup — first ops include allocation / kernel setup.
    for _ in range(5):
        a @ b
    if device_str == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        a @ b
    if device_str == "mps":
        torch.mps.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / ITERS * 1e3
    gflops = (2.0 * N**3) / (avg_ms / 1e3) / 1e9
    return avg_ms, gflops


print(f"N={N}, iters={ITERS}, dtype=float32\n")

avg, gf = bench("cpu")
print("--- PyTorch CPU ---")
print(f"  avg time: {avg:.3f} ms")
print(f"  perf    : {gf:.1f} GFLOP/s\n")

if torch.backends.mps.is_available():
    avg, gf = bench("mps")
    print("--- PyTorch MPS (Apple GPU) ---")
    print(f"  avg time: {avg:.3f} ms")
    print(f"  perf    : {gf:.1f} GFLOP/s")
else:
    print("MPS not available on this build")
