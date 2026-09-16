"""bench_torch_xpu.py — vendor-library matmul yardstick for G3 on one PVC tile.

Aurora analog of POLARIS/bench_torch.py (which measured cuBLAS through torch on the
A100). `a @ b` on an XPU tensor dispatches to oneMKL SGEMM. Same timing method and
GFLOP/s formula as the Mojo kernels: warmup, ITERS launches, one synchronize,
2*N^3 / seconds-per-pass.

Run on a next-eval compute node with the ALCF frameworks module:
    module load frameworks
    ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 python bench_torch_xpu.py 2048 50
"""

import os
import sys
import time

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
DTYPE = torch.float32

print(f"torch {torch.__version__}, N={N}, iters={ITERS}, dtype=float32")
print(f"ZE_FLAT_DEVICE_HIERARCHY={os.environ.get('ZE_FLAT_DEVICE_HIERARCHY')} "
      f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    sys.exit("torch.xpu not available")
print(f"xpu devices visible: {torch.xpu.device_count()}  device 0: {torch.xpu.get_device_name(0)}")

device = torch.device("xpu:0")
a = torch.randn(N, N, dtype=DTYPE, device=device)
b = torch.randn(N, N, dtype=DTYPE, device=device)

for _ in range(5):
    a @ b
torch.xpu.synchronize()

t0 = time.perf_counter()
for _ in range(ITERS):
    a @ b
torch.xpu.synchronize()
t1 = time.perf_counter()

avg_ms = (t1 - t0) / ITERS * 1e3
gflops = (2.0 * N**3) / (avg_ms / 1e3) / 1e9
print("--- PyTorch XPU (oneMKL SGEMM, one PVC tile) ---")
print(f"  avg time: {avg_ms:.3f} ms")
print(f"  perf    : {gflops:.1f} GFLOP/s")
