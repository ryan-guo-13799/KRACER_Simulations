"""DBGD with the explicitly approved sphere initialization/projection adaptation."""

import numpy as np
from .dueling import DuelingBaseline, unit


class DBGD(DuelingBaseline):
    phase = "dbgd"

    def __init__(self, cfg, prior=None, *, seed=0, delta=None, gamma=None, initial=None):
        # The supplied preference enclosure is intentionally not used by DBGD.
        self.delta_override, self.gamma_override = delta, gamma
        self.initial = None if initial is None else np.asarray(initial, dtype=float).copy()
        self.reset(seed, cfg)

    def reset(self, seed, config):
        super().reset(seed, config)
        # RAES Section 5.1: its d is the ambient dimension (our D, not D-1).
        self.delta = (
            self.D ** (-0.5) * config.T ** (-0.25)
            if self.delta_override is None
            else float(self.delta_override)
        )
        self.gamma = (
            config.T ** (-0.5) if self.gamma_override is None else float(self.gamma_override)
        )
        if (
            not np.isfinite(self.delta)
            or not np.isfinite(self.gamma)
            or self.delta <= 0
            or self.gamma <= 0
        ):
            raise ValueError("DBGD step sizes must be finite and positive")
        if self.initial is not None and (
            self.initial.shape != (self.D,)
            or not np.isclose(np.linalg.norm(self.initial), 1.0, rtol=0, atol=1e-10)
        ):
            raise ValueError("Initial incumbent must be a unit vector")
        self.w = unit(self.rng.normal(size=self.D)) if self.initial is None else self.initial.copy()
        self.direction = None
        self.winner_updates = self.zero_projection_fallbacks = 0

    def _project(self, vector):
        if np.linalg.norm(vector) <= 1e-15:
            self.zero_projection_fallbacks += 1
        # At zero all sphere points are equally close; choose the incumbent.
        return unit(vector, self.w)

    def _select_pair(self):
        self.direction = unit(self.rng.normal(size=self.D))
        return np.array([self.w, self._project(self.w + self.delta * self.direction)])

    def _learn(self, pair, winner):
        if winner == 1:
            self.w = self._project(self.w + self.gamma * self.direction)
            self.winner_updates += 1

    def estimate(self):
        return self.w.copy()

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "delta": self.delta,
            "gamma": self.gamma,
            "winner_updates": self.winner_updates,
            "zero_projection_fallbacks": self.zero_projection_fallbacks,
            "adaptation": "sphere_normalization_and_unit_initialization",
        }
