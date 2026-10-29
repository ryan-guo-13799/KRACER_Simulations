"""Native pairwise RAES on an ambient ellipsoid, with no cone startup."""

import numpy as np
from .agent import Agent
from .geometry import eigenframe, cutting_direction, shallow_cut
from .geometry import ellipsoid_membership


class RAESUpdated(Agent):
    """Pairwise RAES: a supplied ambient prior or the full unit sphere."""

    def __init__(self, cfg, prior):
        self.cfg = cfg
        self.D = cfg.D
        # The full sphere is enclosed by the unit ball. The initial axis is
        # deterministic, with no preference information or startup history.
        self.x = np.zeros(cfg.D) if prior is None else prior.x.copy()
        self.P = np.eye(cfg.D) if prior is None else prior.P.copy()
        self.c0 = np.eye(cfg.D)[0] if prior is None else prior.c.copy()
        self._estimate = self.c0.copy()
        self._diag = {}
        self._mode = None
        self.numerical_inflations = 0
        self.total_numerical_jitter = 0.0
        self._eigvals, self._eigvecs = eigenframe(self.P, self._estimate)
        self._refresh()

    def _refresh(self):
        self._eigvals, self._eigvecs = eigenframe(self.P, self._estimate)
        largest = self._eigvals[0]
        mask = abs(self._eigvals - largest) <= 1e-10 * max(abs(largest), 1e-12)
        U = self._eigvecs[:, mask]
        projection = U @ (U.T @ self._estimate)
        if np.linalg.norm(projection) < 1e-10:
            projection = U @ (U.T @ self.c0)
        if np.linalg.norm(projection) < 1e-10:
            projection = U[:, 0]
        estimate = projection / np.linalg.norm(projection)
        ref = self.x if np.linalg.norm(self.x) > 1e-12 else self.c0
        if estimate @ ref < -1e-12:
            estimate = -estimate
        self._estimate = estimate

    def propose(self, t, V_pre, beta_t, zeta_t):
        self._diag = {"alpha_candidate": float("nan"), "cut_ratio": float("nan")}
        if t > self.cfg.T0:
            self._mode = "exploit"
            return np.array([self._estimate, self._estimate]), "exploit"
        g = cutting_direction(self.x, self.P, self._estimate)
        ratio = (beta_t + zeta_t) * np.sqrt(g @ np.linalg.solve(V_pre, g)) / np.sqrt(g @ self.P @ g)
        self._diag = {"alpha_candidate": -float(ratio), "cut_ratio": float(ratio)}
        if ratio <= 1 / (self.cfg.kappa_cut * self.D) + 1e-12:
            self._mode = "cut"
            return np.array([-g, g]), "localize"
        _, U = eigenframe(V_pre, self._estimate)
        vmax, vmin = U[:, 0], U[:, -1]
        self._mode = "explore"
        return np.array([0.8 * vmax + 0.6 * vmin, 0.8 * vmax - 0.6 * vmin]), "excite"

    def observe(self, t, slate, winner, V_pre, beta_t, zeta_t):
        if self._mode != "cut":
            return
        a = slate[winner] - slate[1 - winner]
        margin = (beta_t + zeta_t) * np.sqrt(a @ np.linalg.solve(V_pre, a))
        self.x, self.P, alpha = shallow_cut(self.x, self.P, a, margin)
        if abs(alpha - self._diag["alpha_candidate"]) > 1e-7:
            raise AssertionError("Cut depth mismatch")
        eig = np.linalg.eigvalsh(self.P)
        # Repeated perfect central cuts can flatten the ellipsoid below double
        # precision. Add a tiny PSD outward inflation, never a truth-dependent
        # recentering or contraction. Log every occurrence for the evaluator.
        floor = 1e-12 * max(eig[-1], 1e-30)
        if eig[0] < floor:
            jitter = floor - eig[0]
            self.P = self.P + jitter * np.eye(self.D)
            self.numerical_inflations += 1
            self.total_numerical_jitter += float(jitter)
        self._refresh()

    def estimate(self):
        return self._estimate.copy()

    def membership(self, u):
        return ellipsoid_membership(u, self.x, self.P)

    def diagnostics(self):
        phase = {"cut": "localize", "explore": "excite", "exploit": "exploit"}.get(self._mode)
        return {
            **self._diag,
            "phase": phase,
            "algorithm_native_phase": "RAES " + str(self._mode),
            "logdet_P": float(np.linalg.slogdet(self.P)[1]),
            "numerical_inflations": self.numerical_inflations,
            "total_numerical_jitter": self.total_numerical_jitter,
            "largest_P_eigenvalue": float(self._eigvals[0]),
            **{f"P_eigenvalue_{i}": float(v) for i, v in enumerate(self._eigvals)},
        }

    def prefix_is_safe(self, t, cutoff):
        """The preparation policy reads T0 only in its stopping condition."""
        return t <= cutoff
