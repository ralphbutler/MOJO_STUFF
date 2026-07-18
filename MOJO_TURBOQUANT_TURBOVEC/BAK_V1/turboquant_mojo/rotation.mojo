# rotation.mojo — deterministic random orthogonal rotation matrix.
#
# TurboQuant's key trick: rotate every vector by a fixed random orthogonal
# matrix Q so each coordinate becomes ~ N(0, 1/d), letting one shared scalar
# quantizer (see codebook.mojo) fit every coordinate.
#
# turbovec builds Q via QR of a seeded ChaCha8 Gaussian matrix. We don't need
# *his* exact matrix — recall is invariant to which random orthogonal basis we
# pick — so we use our own seeded PRNG (SplitMix64 -> Box-Muller Gaussian) and
# Householder QR. Deterministic given the seed, so index and query share Q.
#
# Q is returned row-major, length dim*dim. Applying it is a dense mat-vec:
#   rotated = unit . Q^T   (encode/search both use Q^T).

from std.math import sqrt, log, cos, sin

comptime PI = 3.141592653589793
comptime ROTATION_SEED: UInt64 = 0x9E3779B97F4A7C15


struct SplitMix64(Copyable, Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next_u64(mut self) -> UInt64:
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    # Uniform in [0, 1).
    def next_f64(mut self) -> Float64:
        # top 53 bits -> double in [0,1)
        return Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    # Standard normal via Box-Muller (one of the pair).
    def next_gaussian(mut self) -> Float64:
        var u1 = self.next_f64()
        var u2 = self.next_f64()
        if u1 < 1e-300:
            u1 = 1e-300
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2)


# Build a dim x dim orthogonal matrix (row-major) via Householder QR of a seeded
# Gaussian matrix. Returns the Q factor as List[Float32] of length dim*dim.
#
# We compute in Float64 for numerical quality, then downcast. Q is formed by
# applying the Householder reflectors to the identity. Column signs are fixed so
# the result is deterministic (sign of the pivot made negative, matching the
# usual R-diagonal-positive convention up to sign; the exact convention is
# irrelevant to us, only determinism is).
def make_rotation_matrix(dim: Int, seed: UInt64 = ROTATION_SEED) -> List[Float32]:
    var rng = SplitMix64(seed)

    # A = column-major dim x dim Gaussian matrix, stored flat as A[col*dim + row].
    var a = List[Float64](capacity=dim * dim)
    for _ in range(dim * dim):
        a.append(rng.next_gaussian())

    # Householder vectors stored per column; also accumulate Q by applying
    # reflectors to identity. We use the standard "form Q explicitly" approach.
    # betas[k] and the reflector v_k (stored in the lower part of column k of A).
    var betas = List[Float64](capacity=dim)
    for _ in range(dim):
        betas.append(0.0)

    for k in range(dim):
        # Compute Householder reflector for column k, rows k..dim.
        var norm_sq = 0.0
        for i in range(k, dim):
            var v = a[k * dim + i]
            norm_sq += v * v
        var alpha = sqrt(norm_sq)
        var akk = a[k * dim + k]
        # Choose sign to avoid cancellation.
        if akk > 0.0:
            alpha = -alpha
        if alpha == 0.0:
            betas[k] = 0.0
            continue
        # v = x - alpha e_k, with v[k] = akk - alpha.
        var vk = akk - alpha
        # beta = 2 / (v^T v). Store v in A's lower column (v[k]=vk, rest unchanged).
        var vnorm_sq = vk * vk
        for i in range(k + 1, dim):
            vnorm_sq += a[k * dim + i] * a[k * dim + i]
        if vnorm_sq < 1e-300:
            betas[k] = 0.0
            continue
        betas[k] = 2.0 / vnorm_sq
        a[k * dim + k] = vk  # overwrite pivot with v[k]
        # Apply reflector to trailing columns k+1..dim of A.
        for j in range(k + 1, dim):
            var dot = 0.0
            for i in range(k, dim):
                dot += a[k * dim + i] * a[j * dim + i]
            var w = betas[k] * dot
            for i in range(k, dim):
                a[j * dim + i] -= w * a[k * dim + i]
        # Stash alpha as the R diagonal (implicit); we only need Q here.
        # Keep v in column k lower part; a[k*dim+k]=vk already set.

    # Form Q = H_0 H_1 ... H_{dim-1} applied to identity, column-major q[col*dim+row].
    var q = List[Float64](capacity=dim * dim)
    for _ in range(dim * dim):
        q.append(0.0)
    for i in range(dim):
        q[i * dim + i] = 1.0
    # Apply reflectors in reverse order to build Q.
    for kk in range(dim):
        var k = dim - 1 - kk
        if betas[k] == 0.0:
            continue
        for j in range(dim):  # each column of Q
            var dot = 0.0
            for i in range(k, dim):
                dot += a[k * dim + i] * q[j * dim + i]
            var w = betas[k] * dot
            for i in range(k, dim):
                q[j * dim + i] -= w * a[k * dim + i]

    # Downcast to row-major Float32: result[row*dim + col] = q[col*dim + row].
    var result = List[Float32](capacity=dim * dim)
    for _ in range(dim * dim):
        result.append(0.0)
    for col in range(dim):
        for row in range(dim):
            result[row * dim + col] = Float32(q[col * dim + row])
    return result^


# --- validation: orthogonality checks on a small matrix ---
def main() raises:
    var dim = 8
    var q = make_rotation_matrix(dim)
    print("built Q for dim =", dim, " (len =", len(q), ")")

    # Check Q^T Q = I : compute max |(Q^T Q)_{ab} - delta_ab|.
    var max_err = Float32(0.0)
    for aa in range(dim):
        for bb in range(dim):
            var s = Float32(0.0)
            for i in range(dim):
                # column aa dot column bb  (Q row-major: Q[i*dim+aa])
                s += q[i * dim + aa] * q[i * dim + bb]
            var target = Float32(1.0) if aa == bb else Float32(0.0)
            var e = s - target
            if e < 0.0:
                e = -e
            if e > max_err:
                max_err = e
    print("max |Q^T Q - I| =", max_err)

    # Check it preserves norm of a test vector.
    var x = List[Float32](capacity=dim)
    for i in range(dim):
        x.append(Float32(i) - 3.5)
    var nx = Float32(0.0)
    for i in range(dim):
        nx += x[i] * x[i]
    # y = x . Q^T  => y[a] = sum_i x[i] Q[a*dim+i]
    var ny = Float32(0.0)
    for aa in range(dim):
        var ya = Float32(0.0)
        for i in range(dim):
            ya += x[i] * q[aa * dim + i]
        ny += ya * ya
    print("||x||^2 =", nx, "  ||xQ^T||^2 =", ny, "  (should match)")
