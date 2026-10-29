"""Ablation: replace winner-tree admission with the analytical error gate."""

import numpy as np

from recommenders.KRACER import KRACER
from recommenders.KRACER_functions import Restart, WorstNormal


class AnalyticalCertificate(KRACER):
    def __init__(self, state, sim_config, recommender_config, **kwargs):
        super().__init__(state, sim_config, recommender_config, **kwargs)
        self.gate_size = recommender_config.analytical_gate_size
        if not np.isfinite(self.gate_size) or self.gate_size <= 0:
            raise ValueError("analytical_gate_size must be positive and finite")

    # EXTRA FOR THIS ABLATION: test the analytical bound instead of a tree.
    def _find_depth(self, max_depth):
        for depth in range(1, max_depth + 1):
            if self._certificate(depth) <= self.gamma_ctr:
                return depth
        return None

    # EXTRA FOR THIS ABLATION: analytical admission instead of TreeCertificate.
    def _certificate(self, depth):
        state = self.state
        omega = np.sqrt(state.W[state.i])
        alpha = np.arctan(omega)
        d = self.dim - 1
        C_d = np.sqrt(d) * (d / (d - 1)) ** ((d - 1) / 2)
        G = 2 * alpha / np.sin(2 * alpha)
        if C_d * G / self.K**depth > self.gamma_ctr:
            return np.inf

        _, variance = WorstNormal(state.I, state.V, state.c, state.F[:, state.i])
        h_window = max(self.config.h_t(self.t + step) for step in range(depth))
        delta = G / np.cos(alpha) * h_window * np.sqrt(variance)
        error = 2 * C_d * delta / (omega * np.cos(alpha)**2)
        return self.gamma_ctr if error <= self.gate_size else np.inf

    # EXTRA FOR THIS ABLATION: a permissive gate needs a real restart check.
    def _finish_round(self):
        state = self.state
        try:
            c, F, W = Restart(state.c, state.F, state.W, state.i, state.I)
        except (ArithmeticError, ValueError):
            self.cutoff = self.t
            return
        d = self.dim - 1
        factor = (d / (d - 1)) ** ((d - 1) / 2)
        contraction = factor * np.sqrt(W[state.i] / state.W[state.i])
        if contraction <= self.gamma_ctr:
            state.c, state.F, state.W = c, F, W
        else:
            self.cutoff = self.t  # Preserve the completed cone and exploit next round.
