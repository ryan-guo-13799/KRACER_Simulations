"""Ellipsoid geometry used by the RAES baseline."""

import numpy as np


def signed(v):
    v = v.copy()
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


def eigenframe(P, reference=None):
    """Descending eigenvalues; tied bases use public reference then canonical axes."""
    vals, U = np.linalg.eigh((P + P.T) / 2)
    vals = vals[::-1]
    U = U[:, ::-1]
    out = np.empty_like(U)
    start = 0
    while start < len(vals):
        end = start + 1
        while end < len(vals) and abs(vals[end] - vals[start]) <= 1e-10 * max(
            abs(vals[start]), 1e-12
        ):
            end += 1
        if end - start == 1:
            out[:, start] = signed(U[:, start])
        else:
            proj = U[:, start:end] @ U[:, start:end].T
            basis = []
            cand = ([] if reference is None else [reference]) + list(np.eye(len(vals)))
            for a in cand:
                v = proj @ a
                for b in basis:
                    v -= b * (b @ v)
                if np.linalg.norm(v) > 1e-9:
                    basis.append(v / np.linalg.norm(v))
                if len(basis) == end - start:
                    break
            out[:, start:end] = np.column_stack(basis)
        start = end
    return vals, out


def cutting_direction(x, P, reference=None):
    _, U = eigenframe(P, reference)
    u, v = U[:, :2].T
    p = np.array([u @ x, v @ x])
    g = u if np.linalg.norm(p) < 1e-12 else (-p[1] * u + p[0] * v) / np.linalg.norm(p)
    return signed(g)


def shallow_cut(x, P, winner_difference, margin):
    D = len(x)
    a = winner_difference
    rho = np.sqrt(a @ P @ a)
    alpha = -(margin + a @ x) / rho
    if not -1 / D - 1e-9 <= alpha < 1:
        raise ValueError("Cut outside shallow-cut domain")
    v = P @ (-a / rho)
    xx = x - (1 + D * alpha) / (D + 1) * v
    PP = (
        D
        * D
        * (1 - alpha * alpha)
        / (D * D - 1)
        * (P - 2 * (1 + D * alpha) / ((D + 1) * (1 + alpha)) * np.outer(v, v))
    )
    return xx, (PP + PP.T) / 2, float(alpha)


def ellipsoid_membership(u, x, P):
    v = u - x
    return float(v @ np.linalg.solve(P, v))
