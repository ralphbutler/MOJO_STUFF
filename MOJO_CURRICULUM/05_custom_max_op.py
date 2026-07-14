"""05_custom_max_op.py — the capstone: a custom Mojo op running inside MAX.

The relu we hand-wrote in 04/04b is registered as a MAX graph operation in
custom_op_kernels/relu.mojo (@compiler.register("relu")). This driver builds a one-op
graph around it, compiles it, and runs it on the accelerator. It closes the whole journey:

  Mojo  — write the kernel (00–04b)
  MAX   — run models (max_generate.sh / max_serve_litellm.sh)
  ...   — and HERE the two meet: MAX runs a kernel YOU wrote in Mojo.

No training here — just proof that a hand-written Mojo op executes as a first-class MAX
operation, CPU or GPU (MAX picks the target).

Run:  uv run python 05_custom_max_op.py
"""

from pathlib import Path

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

kernels = Path(__file__).parent / "custom_op_kernels"   # dir MAX compiles our .mojo from
dtype = DType.float32
device = CPU() if accelerator_count() == 0 else Accelerator()
dev = DeviceRef.from_device(device)
print(f"device: {'GPU (Metal)' if accelerator_count() else 'CPU'}")

# A mix of negatives and positives so relu's effect is obvious.
data = np.array([[-2.0, -0.5, 0.0, 0.5, 2.0, -1.0, 3.0, -4.0]], dtype=np.float32)

# One-op graph: input -> custom "relu" -> output.
with Graph(
    "relu_custom",
    input_types=[TensorType(dtype, shape=list(data.shape), device=dev)],
    custom_extensions=[kernels],          # compile + link our Mojo op
) as graph:
    x = graph.inputs[0]
    y = ops.custom(
        name="relu",                      # must match @compiler.register("relu")
        device=dev,
        values=[x],
        out_types=[TensorType(dtype=x.tensor.dtype, shape=x.tensor.shape, device=dev)],
    )[0].tensor
    graph.output(y)

session = InferenceSession(devices=[device])
model = session.load(graph)               # compiles the graph incl. our Mojo kernel

xt = Buffer.from_numpy(data).to(device)
result = model.execute(xt)[0].to(CPU()).to_numpy()

expected = np.maximum(data, 0.0)
print("input:   ", data)
print("relu(x): ", result)
print("expected:", expected)
print("RESULT:  ", "PASS" if np.allclose(result, expected) else "FAIL")
