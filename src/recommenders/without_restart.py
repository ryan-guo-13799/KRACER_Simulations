"""Ablation: retain coordinate intervals in the initial, fixed cone chart."""

import math

import numpy as np

from recommenders.KRACER import KRACER
from recommenders.KRACER_functions import Cut, WinnerTree


def half_width(interval):
    return (np.tan(interval[1]) - np.tan(interval[0])) / 2


class WithoutRestart(KRACER):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        angles = np.arctan(np.sqrt(self.state.W))
        self.intervals = np.column_stack((-angles, angles))

    # EXTRA FOR THIS ABLATION: choose by retained width, keeping the original chart.
    def _prepare_interval(self):
        state = self.state
        state.i = int(np.argmax([half_width(I) for I in self.intervals]))
        state.I = tuple(self.intervals[state.i])
        self._root_width = half_width(state.I)

    def _max_depth(self):
        if self._root_width <= 0:
            return 0
        sec_squared = 1 + self.state.W[self.state.i]
        return max(1, math.ceil(math.log(2 * sec_squared / self.gamma_ctr, self.K)))

    def _cut(self, interval, V, h, winner):
        state = self.state
        # A previously narrowed axis can be wider than another remaining axis.
        # Keep the original W for the bounds; relax only the axis-selection check.
        return Cut(interval, V, h, winner, state.c, state.F[:, state.i],
                   self.K, state.W, state.i, require_largest=False)

    # EXTRA FOR THIS ABLATION: score leaves by interval width, without a restart.
    def _find_depth(self, max_depth):
        levels = WinnerTree(self.state.I, self.state.V, self.t, max_depth,
                            self.K, self.config.h_t, self._cut)
        for depth, nodes in enumerate(levels, start=1):
            worst = max(half_width(interval) / self._root_width for interval, _ in nodes)
            if worst <= self.gamma_ctr:
                return depth
        return None

    # EXTRA FOR THIS ABLATION: save the interval instead of replacing the cone.
    def _finish_round(self):
        self.intervals[self.state.i] = self.state.I
        if half_width(self.state.I) / self._root_width > self.gamma_ctr:
            self.cutoff = self.t

    def estimate(self):
        midpoint = np.tan(self.intervals).mean(axis=1)
        direction = self.state.c + self.state.F @ midpoint
        return direction / np.linalg.norm(direction)
