"""Ailon et al. (2014), Algorithm 4, with two independent OFUL learners."""

import numpy as np
from .dueling import DuelingBaseline
from .oful import OFUL


class Sparring(DuelingBaseline):
    phase = "sparring"

    def __init__(self, cfg, prior=None, *, seed=0, oful_config=None):
        self.oful_config = oful_config
        self.reset(seed, cfg)

    def reset(self, seed, config):
        super().reset(seed, config)
        seeds = self.rng.integers(0, 2**63, size=2)
        self.left = OFUL(self.D, seed=int(seeds[0]), config=self.oful_config)
        self.right = OFUL(self.D, seed=int(seeds[1]), config=self.oful_config)

    def _select_pair(self):
        return np.array([self.left.select_action(), self.right.select_action()])

    def _learn(self, pair, winner):
        self.left.update(pair[0], int(winner == 0))
        self.right.update(pair[1], int(winner == 1))

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "left_oful": self.left.diagnostics(),
            "right_oful": self.right.diagnostics(),
        }
