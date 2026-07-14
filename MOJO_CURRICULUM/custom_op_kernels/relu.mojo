# relu.mojo — a custom Mojo op registered into MAX. THE CAPSTONE.
#
# The relu activation we hand-wrote inside the training loops (04/04b) is here promoted
# to a first-class operation inside MAX's graph/inference engine. This is the answer to
# "why is Mojo a language and not just a demo": when a shipped MAX op doesn't fit, you
# write the kernel in Mojo, annotate it with @compiler.register, and MAX compiles and runs
# it as a graph node — CPU or GPU, chosen by `target`. The Python driver (05_custom_max_op.py)
# builds a one-op graph around this and executes it.

import compiler

from std.gpu.host import DeviceContext
from std.math import max

from extensibility import InputTensor, OutputTensor, foreach

from std.utils.coord import Coord


@compiler.register("relu")
struct Relu:
    @staticmethod
    def execute[
        target: StaticString,                     # "cpu" or "gpu" — MAX fills this in
    ](
        output: OutputTensor,
        x: InputTensor[dtype = output.dtype, rank = output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        # One elementwise rule, applied across the whole tensor by `foreach`.
        # Same math as the relu in 04/04b: pass positives through, clamp the rest to 0.
        @parameter
        @always_inline
        def relu_elementwise[width: Int](idx: Coord) -> SIMD[x.dtype, width]:
            var v = x.load[width](idx)
            return max(v, SIMD[x.dtype, width](0))   # relu = max(x, 0), per lane

        foreach[relu_elementwise, target=target](output, ctx)
