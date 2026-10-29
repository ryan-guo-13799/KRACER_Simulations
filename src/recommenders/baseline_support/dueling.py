"""Pairwise public-feedback protocol adapted to the existing Agent interface."""

import numpy as np
from .agent import Agent


def unit(vector, fallback=None):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length):
        raise ValueError("Nonfinite action")
    if length <= 1e-15:
        if fallback is None:
            raise ValueError("Cannot normalize a zero vector")
        return unit(fallback)
    return vector / length


class DuelingBaseline(Agent):
    """Policies ignore V/beta/zeta; only displayed pairs and winners train them.

    Native baselines learn for the entire horizon. Registration declares that
    preparation budgets do not apply; no synthetic T0 is required.
    An evaluator-only membership call returns NaN (no enclosure), without
    retaining or inspecting its argument.
    """

    phase = "duel"

    def reset(self, seed, config):
        if config.K != 2:
            raise ValueError("Traditional baselines require K=2")
        self.cfg, self.D = config, config.D
        self.rng = np.random.default_rng(seed)
        self.rounds = 0
        self._pending = None
        self._last_winner = np.eye(self.D)[0]

    def select_pair(self):
        if self._pending is not None:
            raise RuntimeError("Feedback required before selecting another pair")
        pair = np.asarray(self._select_pair(), dtype=float)
        if pair.shape != (2, self.D) or not np.all(np.isfinite(pair)):
            raise ValueError("Invalid recommendation pair")
        if not np.allclose(np.linalg.norm(pair, axis=1), 1.0, rtol=0, atol=1e-10):
            raise ValueError("Recommendations must lie on the unit sphere")
        self._pending = pair.copy()
        return pair.copy()

    def update(self, pair, winner_index):
        if self._pending is None:
            raise RuntimeError("No outstanding duel")
        if not isinstance(winner_index, (int, np.integer)) or winner_index not in (0, 1):
            raise ValueError("Winner must be index 0 or 1")
        pair = np.asarray(pair, dtype=float)
        if not np.array_equal(pair, self._pending):
            raise ValueError("Feedback must match the displayed pair")
        self._learn(self._pending, int(winner_index))
        self._last_winner = self._pending[winner_index].copy()
        self.rounds += 1
        self._pending = None

    def propose(self, t, V_pre, beta_t, zeta_t):
        if t != self.rounds + 1:
            raise ValueError("Unexpected absolute round")
        return self.select_pair(), self.phase

    def observe(self, t, slate, winner, V_pre, beta_t, zeta_t):
        if t != self.rounds + 1:
            raise ValueError("Unexpected feedback round")
        self.update(slate, winner)

    def estimate(self):
        return self._last_winner.copy()

    def membership(self, u):
        return float("nan")

    def diagnostics(self):
        return {
            "rounds": self.rounds,
            "phase": self.phase,
            "enclosure_applicable": False,
            "learning_lifecycle": "full_horizon",
        }
