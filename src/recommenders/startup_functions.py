import math

import numpy as np

# BEGIN K-RACER PARITY CHANGE (step 1): reuse the origin's tangent-frame convention.
from recommenders.KRACER_functions import TangentFrame
# END K-RACER PARITY CHANGE (step 1).


def sign_threshold(dimension: int, half_angle: float) -> float:
    """Return the paper's sign-certification threshold for the target cone."""
    if dimension < 2:
        raise ValueError("startup needs at least two dimensions")
    if not math.acos(1 / math.sqrt(dimension)) < half_angle < math.pi / 2:
        raise ValueError("startup half-angle must lie between acos(1/sqrt(D)) and pi/2")
    return (math.sin(half_angle) / math.sqrt(dimension - 1)
            - math.cos(half_angle)) / math.sqrt(dimension)


def make_cone(signs: np.ndarray, half_angle: float):
    """Construct the target cone from all certified coordinate signs."""
    D = len(signs)
    sign_threshold(D, half_angle)
    if not np.all(np.isin(signs, [-1, 1])):
        raise ValueError("each coordinate needs a certified sign")
    c = np.asarray(signs, dtype=float) / math.sqrt(D)
    # BEGIN K-RACER PARITY CHANGE (step 1): replace QR with canonical-axis
    # projection. Equal cone widths make the basis order affect later queries.
    F = TangentFrame(c)
    # END K-RACER PARITY CHANGE (step 1).
    W = np.full(D - 1, math.tan(half_angle)**2)
    return c, F, W
