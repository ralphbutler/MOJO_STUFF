# codebook.mojo — Lloyd-Max scalar quantizer for the post-rotation coordinate.
#
# After a random orthogonal rotation, each coordinate of a unit vector on the
# (d-1)-sphere follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. For high d this is
# extremely close to a zero-mean Gaussian with variance 1/d (the variance of
# that Beta on [-1,1] is exactly 1/(2a+1) = 1/d). We therefore fit the optimal
# (Lloyd-Max) scalar quantizer to a *standard normal* once, then scale the
# levels by sigma = 1/sqrt(d). This mirrors turbovec's codebook.rs but replaces
# its Beta+adaptive-Simpson machinery with the closed-form truncated-normal
# conditional mean — exact for the Gaussian limit, near-identical to the Beta
# fit at d=1536.
#
# The quantizer is defined by:
#   boundaries[0..L-1]  ascending thresholds
#   centroids[0..L]     reconstruction level per cell,  L = 2^bits cells
# A value's code = number of boundaries it exceeds (a monotone bucket index).

from std.math import sqrt, exp, erf

comptime PI = 3.141592653589793
comptime SQRT2 = 1.4142135623730951
comptime BIG = 40.0  # stand-in for +/- infinity at the outer cell edges


struct Codebook(Copyable, Movable):
    var boundaries: List[Float32]  # length L-1
    var centroids: List[Float32]   # length L
    var bits: Int

    def __init__(out self, var boundaries: List[Float32], var centroids: List[Float32], bits: Int):
        self.boundaries = boundaries^
        self.centroids = centroids^
        self.bits = bits


def std_normal_pdf(x: Float64) -> Float64:
    return exp(-0.5 * x * x) / sqrt(2.0 * PI)


def std_normal_cdf(x: Float64) -> Float64:
    return 0.5 * (1.0 + erf(x / SQRT2))


# Closed-form conditional mean of N(0,1) restricted to [lo, hi]:
#   E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
def truncated_normal_mean(lo: Float64, hi: Float64) -> Float64:
    var mass = std_normal_cdf(hi) - std_normal_cdf(lo)
    if mass < 1e-15:
        return 0.5 * (lo + hi)
    return (std_normal_pdf(lo) - std_normal_pdf(hi)) / mass


# Lloyd-Max levels for a *standard normal*, written into the provided lists
# (cleared first). boundaries -> length L-1, centroids -> length L, L = 2^bits.
def lloyd_max_std_normal(
    bits: Int,
    mut boundaries: List[Float64],
    mut centroids: List[Float64],
    max_iter: Int = 200,
    tol: Float64 = 1e-12,
):
    var n_levels = 1 << bits
    boundaries.clear()
    centroids.clear()

    # Initialize centroids evenly over +/- 3 sigma.
    var spread = 3.0
    for i in range(n_levels):
        var t = Float64(i) / Float64(n_levels - 1)
        centroids.append(-spread + 2.0 * spread * t)

    for _ in range(max_iter):
        # Cell edges: -inf, midpoints between consecutive centroids, +inf.
        var edges = List[Float64](capacity=n_levels + 1)
        edges.append(-BIG)
        for i in range(n_levels - 1):
            edges.append(0.5 * (centroids[i] + centroids[i + 1]))
        edges.append(BIG)

        var max_change = 0.0
        for i in range(n_levels):
            var m = truncated_normal_mean(edges[i], edges[i + 1])
            var change = m - centroids[i]
            if change < 0.0:
                change = -change
            if change > max_change:
                max_change = change
            centroids[i] = m

        if max_change < tol:
            break

    for i in range(n_levels - 1):
        boundaries.append(0.5 * (centroids[i] + centroids[i + 1]))


# Codebook for the given bit width and dimension, scaling standard-normal
# levels by sigma = 1/sqrt(d).
def codebook(bits: Int, dim: Int) -> Codebook:
    var b64 = List[Float64]()
    var c64 = List[Float64]()
    lloyd_max_std_normal(bits, b64, c64)
    var sigma = 1.0 / sqrt(Float64(dim))

    var boundaries = List[Float32](capacity=len(b64))
    for i in range(len(b64)):
        boundaries.append(Float32(b64[i] * sigma))
    var centroids = List[Float32](capacity=len(c64))
    for i in range(len(c64)):
        centroids.append(Float32(c64[i] * sigma))
    return Codebook(boundaries^, centroids^, bits)


def main() raises:
    # Validate against textbook standard-normal Lloyd-Max values.
    print("== standard-normal Lloyd-Max (unit sigma) ==")
    for bits in range(1, 5):
        var b = List[Float64]()
        var c = List[Float64]()
        lloyd_max_std_normal(bits, b, c)
        print("bits =", bits, " levels =", len(c))
        var cs = String("  centroids: ")
        for i in range(len(c)):
            cs += String(c[i]) + " "
        print(cs)
        if bits <= 2:
            var bs = String("  boundaries: ")
            for i in range(len(b)):
                bs += String(b[i]) + " "
            print(bs)

    print("\n== d=1536, 4-bit codebook (scaled by 1/sqrt(d)) ==")
    var cb = codebook(4, 1536)
    print("n_boundaries:", len(cb.boundaries), " n_centroids:", len(cb.centroids))
    print("centroid[0]:", cb.centroids[0], " centroid[15]:", cb.centroids[15])
    print("sigma = 1/sqrt(1536) =", 1.0 / sqrt(1536.0))
