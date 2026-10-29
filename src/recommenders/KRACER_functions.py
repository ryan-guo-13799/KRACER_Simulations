from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from models.item import Item

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
AngularInterval = tuple[float, float]


def EQ(I: AngularInterval, c: Vector, g_i: Vector, K: int) -> list[Item]:
    A, B = I 

    _validate_eq(A, B, c, g_i, K)

    bin_angle = (B - A) / K
    slate = []
    for j in range(K):
        angle = A + (j + 0.5) * bin_angle
        q = np.cos(angle) * c + np.sin(angle) * g_i
        q /= np.linalg.norm(q)
        slate.append(Item(q))
    return slate


def Restart(
    c: Vector,
    F: Matrix,
    W: Vector,
    i: int,
    I: AngularInterval
) -> tuple[Vector, Matrix, Vector]:
    L, R = I
    
    _validate_restart(c, F, W, i, L, R)
    D = c.size
    d = D - 1

    # whitening
    active_length = np.sqrt(W[i]) # half axis length of the ellipsoid
    left = np.tan(L) / active_length
    right = np.tan(R) / active_length
    _validate_whitened_bounds(left, right)
    left = max(-1.0, left)
    right = min(1.0, right)
    _validate_interval_width(left, right)

    # MVEE
    if left * right <= -1 / d:
        tau_slab, h_active, h_inactive = 0.0, 1.0, 1.0
    else:
        delta_slab = d**2 * (right**2 - left**2) ** 2 + 4 * (1 - left**2) * (1 - right**2)
        tau_slab = 2 * (left + right) * (1 + d * left * right)
        tau_slab /= d * (left + right) ** 2 + 2 * (1 + left * right) + np.sqrt(delta_slab)
        h_active = d * (tau_slab - left) * (right - tau_slab)
        h_inactive = d / (d - 1) * (1 + left * right - tau_slab * (left + right))

    _validate_slab(tau_slab, h_active, h_inactive)

    # lift the ellipsoid to a cone
    xi = tau_slab * active_length
    p = W[i] * h_active
    chi = 1 + xi**2 - p
    root = np.hypot(chi, 2 * np.sqrt(p))
    mu = (chi + root) / 2 if chi >= 0 else 2 * p / (root - chi) # TODO: remove?
    denominator = 1 + p / mu # equivalent to mu + p - xi**2, without subtracting large numbers
    theta = xi / denominator

    _validate_rotation(mu, denominator, theta)
    
    divide = np.hypot(1.0, theta) # sqrt(1+theta**2)
    active_direction = F[:, i]
    new_c = (c + theta * active_direction) / divide
    new_F = F.copy()
    new_F[:, i] = (active_direction - theta * c) / divide
    new_W = W * h_inactive / mu
    new_W[i] = W[i] * h_active / mu**2
    _validate_new_widths(new_W)
    return new_c, new_F, new_W


def Cut(
    I: AngularInterval,
    V: Matrix,
    h: float,
    winner: int,
    c: Vector,
    g_i: Vector,
    K: int,
    W: Vector,
    i: int,
    *,
    require_largest: bool = True,
) -> tuple[AngularInterval, Item]:
    slate = EQ(I, c, g_i, K)
    _validate_cut(I, V, h, winner, c, K, W, i, require_largest)
    A, B = I
    bin_angle = (B - A) / K

    # first and last winners have only one margin
    if winner == 0:
        left = A
        _, right = SafeAngularBounds(I, V, h, A + bin_angle, c, g_i, W, i)
    elif winner == K - 1:
        left, _ = SafeAngularBounds(I, V, h, A + winner * bin_angle, c, g_i, W, i)
        right = B
    else:
        left, _ = SafeAngularBounds(I, V, h, A + winner * bin_angle, c, g_i, W, i)
        _, right = SafeAngularBounds(I, V, h, A + (winner + 1) * bin_angle, c, g_i, W, i)
    return (left, right), slate[winner]


def SafeAngularBounds(
    I: AngularInterval,
    V: Matrix,
    h: float,
    boundary: float,
    c: Vector,
    g_i: Vector,
    W: Vector,
    i: int,
) -> AngularInterval:
    A, B = I
    normal = -np.sin(boundary) * c + np.cos(boundary) * g_i # n(b)
    margin = h * np.sqrt(normal @ np.linalg.solve(V, normal)) # m(b)
    active = W[i]
    transverse = np.max(np.delete(W, i))
    if margin == 0:
        # exact cut (perfect user)
        return boundary, boundary
    if margin >= 1:
        # margin too large
        return A, B

    a = 1 + transverse
    b = 1 - transverse / active
    sin_b, cos_b = np.sin(boundary), np.cos(boundary)
    x_left, x_b, x_right = np.tan(A), np.tan(boundary), np.tan(B)

    # constructing the quadratic equation of candidate x's
    quadratic = cos_b**2 - margin**2 * b
    linear = -2 * sin_b * cos_b
    constant = sin_b**2 - margin**2 * a
    roots = []
    if quadratic == 0:
        # The quadratic equation reduces to linear * x + constant = 0
        if linear != 0:
            roots.append(-constant / linear)
    else:
        # Solve the quadratic with a stable formula, including a repeated root
        discriminant_part = (
            a * cos_b**2
            + b * sin_b**2
            - margin**2 * a * b
        )
        if discriminant_part >= 0:
            root = 2 * margin * np.sqrt(discriminant_part)
            root_numerator = -0.5 * (linear + np.copysign(root, linear))
            roots.append(root_numerator / quadratic)
            if root_numerator != 0:
                roots.append(constant / root_numerator)

    # checks if the x is in the bounds of a side (left/right)
    def feasible(x: float, side: int) -> bool:
        # side=-1 checks left of the boundary; side=1 checks right of it
        profile = side * (x * cos_b - sin_b) / np.sqrt(
            a + b * x**2
        )
        return profile <= margin + 1e-11

    left_roots = [
        x for x in roots if x_left - 1e-10 <= x <= x_b + 1e-10 and feasible(x, -1)
    ]
    right_roots = [
        x for x in roots if x_b - 1e-10 <= x <= x_right + 1e-10 and feasible(x, 1)
    ]

    left = A if feasible(x_left, -1) or not left_roots else np.arctan(min(left_roots))
    right = B if feasible(x_right, 1) or not right_roots else np.arctan(max(right_roots))
    return max(A, float(np.nextafter(left, -np.inf))), min(B, float(np.nextafter(right, np.inf)))


def Observe(
    slate: list[Item],
    winner: int,
    V: Matrix,
    t: int
) -> tuple[Item, Matrix, int]:
    chosen, q_t = _validate_observe(slate, winner, V)
    new_V = V + np.outer(q_t, q_t)
    return chosen, new_V, t + 1


def WinnerTree(I0, V, t, max_depth, K, h_t, cut):
    """Yield each level of hypothetical intervals and information matrices."""
    nodes = [(I0, V.copy())]
    for step in range(max_depth):
        h = h_t(t + step)
        children = []
        for interval, info_matrix in nodes:
            for winner in range(K):
                child_interval, chosen = cut(interval, info_matrix, h, winner)
                child_matrix = info_matrix + np.outer(chosen.emb, chosen.emb)
                children.append((child_interval, child_matrix))
        nodes = children
        yield nodes


def TreeCertificate(
    I0: AngularInterval,
    V: Matrix,
    t: int,
    max_depth: int,
    c: Vector,
    F: Matrix,
    W: Vector,
    i: int,
    K: int,
    h_t: Callable[[int], float],
    gamma_ctr: float,
) -> int | None:
    """Return the first depth that passes, or None if no depth passes."""
    _validate_tree(t, max_depth, c, F, W, i, K)
    d = c.size - 1
    active_direction = F[:, i]
    B_d = d / (d - 1)
    factor = B_d ** ((d - 1) / 2)

    def cut(interval, info_matrix, h, winner):
        return Cut(interval, info_matrix, h, winner, c, active_direction, K, W, i)

    for depth, nodes in enumerate(WinnerTree(I0, V, t, max_depth, K, h_t, cut), start=1):
        # Every possible winner path must pass at this depth.
        worst = 0.0
        for interval, _ in nodes:
            _, _, new_W = Restart(c, F, W, i, interval)
            contraction = factor * np.sqrt(new_W[i] / W[i])
            _validate_contraction(contraction)
            worst = max(worst, float(contraction))
        if worst <= gamma_ctr:
            return depth
    return None


# BEGIN K-RACER PARITY CHANGE (step 1): match the origin's directional
# excitation, candidate ordering, and reference ring.
def ExcitationSlate(
    I0: AngularInterval,
    V: Matrix,
    c: Vector,
    g_i: Vector,
    K: int,
    lambda0: float
) -> list[Item]:
    A, B = I0
    _validate_excitation(A, B, V, c, g_i, K, lambda0)
    D = c.size

    direction, _ = WorstNormal(I0, V, c, g_i)
    center = np.linalg.solve(V + np.eye(D), direction)
    center /= np.linalg.norm(center)

    radius = min(0.25, 1 / np.sqrt(2 * (1 + 1 / lambda0)))
    # BEGIN PAPER ALIGNMENT CHANGE: keep the signed center from Eq. (43).
    # Eq. (44) displays this cap directly, without a global flip toward c.
    return ExcitationRing(center, K, radius, reference=c)
    # END PAPER ALIGNMENT CHANGE.


def WorstNormal(I: AngularInterval, V: Matrix, c: Vector, g_i: Vector):
    """Return the most uncertain normal and its squared V-inverse norm."""
    A, B = I
    # maximize ||n(b)||_{V^{-1}} over the interval
    inverse = np.linalg.inv(V)
    cc = c @ inverse @ c
    gg = g_i @ inverse @ g_i
    cg = c @ inverse @ g_i
    stationary_angle = 0.5 * np.arctan2(2 * cg, cc - gg)
    candidates = [A, B]
    if A <= 0 <= B:
        candidates.append(0.0)
    candidates += [stationary_angle + k * np.pi / 2 for k in range(-3, 4)
                   if A <= stationary_angle + k * np.pi / 2 <= B]
    candidates.sort()

    def normal(b: float) -> Vector: # b: the angle
        return -np.sin(b) * c + np.cos(b) * g_i

    def uncertainty(b: float) -> float: # function we want to maximize
        return float(gg * np.cos(b)**2 + cc * np.sin(b)**2 - 2 * cg * np.sin(b) * np.cos(b))

    values = [uncertainty(angle) for angle in candidates]
    angle = float(candidates[int(np.argmax(values))])
    return normal(angle), uncertainty(angle)


def TangentFrame(center: Vector, reference: Vector | None = None) -> Matrix:
    """Project the reference, then canonical axes, in the origin's order."""
    vectors = []
    candidates = [] if reference is None else [reference]
    candidates += list(np.eye(center.size))
    for axis in candidates:
        tangent = axis - center * (center @ axis)
        for previous in vectors:
            tangent -= previous * (previous @ tangent)
        if np.linalg.norm(tangent) > 1e-10:
            vectors.append(tangent / np.linalg.norm(tangent))
        if len(vectors) == center.size - 1:
            break
    return np.column_stack(vectors)


def ExcitationRing(
    center: Vector, K: int, radius: float, *, reference: Vector | None = None
) -> list[Item]:
    """Place K unit items in a ring around a unit center (shared by ablations)."""
    if reference is not None:
        frame = TangentFrame(center, reference)

        def ring(frame):
            if K == 2:
                return np.array([frame[:, 0], -frame[:, 0]])
            angles = np.arange(K) * 2 * np.pi / K
            return (np.cos(angles)[:, None] * frame[:, 0]
                    + np.sin(angles)[:, None] * frame[:, 1])

        vectors = np.sqrt(1 - radius**2) * center + radius * ring(frame)
        if (not np.all(np.isfinite(vectors))
                or np.max(abs(np.linalg.norm(vectors, axis=1) - 1)) >= 1e-8):
            # Match the origin's fallback for a nearly parallel reference.
            tangents = []
            for axis in np.eye(center.size):
                tangent = axis.copy()
                for _ in range(2):
                    tangent -= center * (center @ tangent)
                    for previous in tangents:
                        tangent -= previous * (previous @ tangent)
                length = np.linalg.norm(tangent)
                if length > 1e-10:
                    tangents.append(tangent / length)
                if len(tangents) == center.size - 1:
                    break
            if len(tangents) != center.size - 1:
                raise FloatingPointError("Unable to construct excitation tangent frame")
            vectors = (np.sqrt(1 - radius**2) * center
                       + radius * ring(np.column_stack(tangents)))
            vectors /= np.linalg.norm(vectors, axis=1)[:, None]
            if (not np.all(np.isfinite(vectors))
                    or np.max(abs(np.linalg.norm(vectors, axis=1) - 1)) >= 1e-12):
                raise FloatingPointError("Invalid repaired excitation slate")
        return [Item(vector) for vector in vectors]

    # Retain the existing no-reference convention for the eigenvector ablation.
    D = center.size
    # Make two perpendicular directions for a ring around the center.
    axes = np.eye(D)
    axis = axes[np.argmin(np.abs(center))]
    first = axis - (axis @ center) * center
    first /= np.linalg.norm(first)
    tangent_axes = axes - np.outer(axes @ center, center) - np.outer(axes @ first, first)
    second = tangent_axes[np.argmax(np.linalg.norm(tangent_axes, axis=1))]
    second /= np.linalg.norm(second)

    slate = []
    for j in range(K):
        angle = 2 * np.pi * j / K
        tangent = np.cos(angle) * first + np.sin(angle) * second
        q = np.sqrt(1 - radius**2) * center + radius * tangent
        slate.append(Item(q))
    return slate
# END K-RACER PARITY CHANGE (step 1).


# Validation is kept here so the algorithms above remain easy to follow.


def _validate_whitened_bounds(left, right):
    if left < -1 - 1e-12 or right > 1 + 1e-12:
        raise ValueError("I must lie inside the cone's angular interval")


def _validate_interval_width(left, right):
    if not left < right:
        raise ValueError("The retained interval must have positive width")


def _validate_slab(tau_slab, h_active, h_inactive):
    if not np.all(np.isfinite([tau_slab, h_active, h_inactive])) or min(h_active, h_inactive) <= 0:
        raise FloatingPointError("Could not calculate a nondegenerate slab ellipsoid")


def _validate_rotation(mu, denominator, theta):
    if not np.isfinite(mu) or mu <= 0:
        raise FloatingPointError("Could not calculate a stable cone restart")
    if not np.isfinite(denominator):
        raise FloatingPointError("Could not calculate a stable cone restart")
    if not np.isfinite(theta):
        raise FloatingPointError("Could not calculate a stable cone rotation")


def _validate_new_widths(new_W):
    if not np.all(np.isfinite(new_W)) or np.any(new_W <= 0):
        raise FloatingPointError("Restart produced invalid cone widths")


def _validate_contraction(contraction):
    if not np.isfinite(contraction):
        raise FloatingPointError("Tree certificate produced a nonfinite value")


def _validate_eq(A, B, c, g_i, K):
    # validations by AI
    if K < 2:
        raise ValueError("EQ needs at least two items")
    if not -np.pi / 2 < A < B < np.pi / 2:
        raise ValueError("EQ needs -pi/2 < A < B < pi/2")
    if c.ndim != 1 or g_i.shape != c.shape:
        raise ValueError("c and g_i must be vectors of the same dimension")
    if not (
        np.isclose(np.linalg.norm(c), 1.0, rtol=0, atol=1e-8)
        and np.isclose(np.linalg.norm(g_i), 1.0, rtol=0, atol=1e-8)
        and np.isclose(c @ g_i, 0.0, rtol=0, atol=1e-8)
    ):
        raise ValueError("c and g_i must be orthonormal unit vectors")



def _validate_restart(c, F, W, i, L, R):
    # validations by AI
    if c.ndim != 1 or c.size < 3:
        raise ValueError("c must be a vector in at least three dimensions")
    D = c.size
    d = D - 1
    if F.shape != (D, d) or W.shape != (d,): # W is the diagnoal of shape matrix of ellipsoid
        raise ValueError("F and W must have shapes (D, D-1) and (D-1,)")
    frame = np.column_stack((c, F))
    if not np.allclose(frame.T @ frame, np.eye(D), rtol=0, atol=1e-8):
        raise ValueError("c and the columns of F must be orthonormal")
    if not np.all(np.isfinite(W)):
        raise ValueError("W must contain finite diagonal entries")
    if np.any(W <= 0):
        raise ValueError("W must have positive diagonal entries")
    if not isinstance(i, (int, np.integer)) or not 0 <= i < d:
        raise ValueError("i must identify an axis of W")
    if not -np.pi / 2 < L < R < np.pi / 2:
        raise ValueError("I must satisfy -pi/2 < L < R < pi/2")



def _validate_cut(I, V, h, winner, c, K, W, i, require_largest=True):
    if not isinstance(winner, (int, np.integer)) or not 0 <= winner < K:
        raise ValueError("winner must be a zero-based position in the slate")

    D = c.size
    if D < 3:
        raise ValueError("Cut needs at least three dimensions")
    if W.shape != (D - 1,) or not np.all(np.isfinite(W)) or np.any(W <= 0):
        raise ValueError("W must contain D-1 positive finite squared axis lengths")
    if not isinstance(i, (int, np.integer)) or not 0 <= i < D - 1:
        raise ValueError("i must identify an axis of W")
    if require_largest and W[i] < np.max(W):
        raise ValueError("i must identify a largest axis")
    if V.shape != (D, D) or not np.all(np.isfinite(V)):
        raise ValueError("V must be a finite D by D matrix")
    if not np.allclose(V, V.T, rtol=0, atol=1e-12):
        raise ValueError("V must be symmetric")
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as exc:
        raise ValueError("V must be positive definite") from exc
    if not np.isfinite(h) or h < 0:
        raise ValueError("h must be finite and nonnegative")

    A, B = I
    alpha = np.arctan(np.sqrt(W[i]))
    if A < -alpha - 1e-12 or B > alpha + 1e-12:
        raise ValueError("I must lie inside the active cone interval")


def _validate_observe(slate, winner, V):
    if not isinstance(winner, (int, np.integer)) or isinstance(winner, (bool, np.bool_)) or not 0 <= winner < len(slate):
        raise ValueError("winner must be a position in the slate")
    chosen = slate[winner]
    q_t = chosen.emb
    if q_t.ndim != 1 or V.shape != (q_t.size, q_t.size) or not np.all(np.isfinite(V)):
        raise ValueError("V must be a finite matrix matching the item dimension")
    if not np.allclose(V, V.T, rtol=0, atol=1e-12):
        raise ValueError("V must be symmetric")
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as exc:
        raise ValueError("V must be positive definite") from exc
    if any(item.emb.shape != q_t.shape or not np.all(np.isfinite(item.emb)) for item in slate):
        raise ValueError("slate items must be finite vectors of the same dimension")

    return chosen, q_t


def _validate_tree(t, depth, c, F, W, i, K):
    if not isinstance(t, (int, np.integer)) or isinstance(t, (bool, np.bool_)) or t < 1:
        raise ValueError("t must be a positive round number")
    if not isinstance(depth, (int, np.integer)) or isinstance(depth, (bool, np.bool_)) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not isinstance(K, (int, np.integer)) or isinstance(K, (bool, np.bool_)) or K < 2:
        raise ValueError("K must be an integer of at least two")
    if c.ndim != 1 or c.size < 3:
        raise ValueError("c must be a vector in at least three dimensions")
    D = c.size
    d = D - 1
    if F.shape != (D, d) or W.shape != (d,):
        raise ValueError("F and W must have shapes (D, D-1) and (D-1,)")
    if not np.all(np.isfinite(W)) or np.any(W <= 0):
        raise ValueError("W must contain positive finite squared axis lengths")
    if not isinstance(i, (int, np.integer)) or isinstance(i, (bool, np.bool_)) or not 0 <= i < d:
        raise ValueError("i must identify an axis of W")
    if W[i] < np.max(W):
        raise ValueError("i must identify a largest axis")



def _validate_excitation(A, B, V, c, g_i, K, lambda0):
    if not -np.pi / 2 < A < B < np.pi / 2:
        raise ValueError("I0 must satisfy -pi/2 < A < B < pi/2")
    if not isinstance(K, (int, np.integer)) or isinstance(K, (bool, np.bool_)) or K < 2:
        raise ValueError("K must be an integer of at least two")
    if c.ndim != 1 or c.size < 3 or g_i.shape != c.shape:
        raise ValueError("c and g_i must be vectors in at least three dimensions")
    if not np.allclose([c @ c, g_i @ g_i, c @ g_i], [1, 1, 0], rtol=0, atol=1e-8):
        raise ValueError("c and g_i must be orthonormal unit vectors")
    if not np.isfinite(lambda0) or lambda0 <= 0:
        raise ValueError("lambda0 must be positive and finite")
    D = c.size
    if V.shape != (D, D) or not np.all(np.isfinite(V)):
        raise ValueError("V must be a finite D by D matrix")
    if not np.allclose(V, V.T, rtol=0, atol=1e-12):
        raise ValueError("V must be symmetric")
    if np.min(np.linalg.eigvalsh(V)) < lambda0 * (1 - 1e-10):
        raise ValueError("V must be at least lambda0 times the identity")

