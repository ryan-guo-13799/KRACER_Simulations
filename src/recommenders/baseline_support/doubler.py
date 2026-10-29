"""Ailon et al. (2014), Algorithm 2, with OFUL as in RAES Section 5.1."""

import numpy as np
from .dueling import DuelingBaseline, unit
from .oful import OFUL


class Doubler(DuelingBaseline):
    phase = "doubler"

    def __init__(self, cfg, prior=None, *, seed=0, oful_config=None):
        self.oful_config = oful_config
        self.reset(seed, cfg)

    def reset(self, seed, config):
        super().reset(seed, config)
        self.oful = OFUL(self.D, seed=int(self.rng.integers(0, 2**63)), config=self.oful_config)
        self.epoch_index = 1
        self.epoch_length = 2
        self.epoch_rounds = 0
        self.left_history = np.array([unit(self.rng.normal(size=self.D))])
        self.right_history = []
        self.completed_epochs = 0
        self._new_epoch = False

    def _select_pair(self):
        if self._new_epoch:
            self.oful.reset_statistics()
            self.epoch_index += 1
            self.epoch_length *= 2
            self.epoch_rounds = 0
            self._new_epoch = False
        # Sample indices, preserving multiplicities of repeated historical arms.
        left = self.left_history[int(self.rng.integers(len(self.left_history)))].copy()
        return np.array([left, self.oful.select_action()])

    def _learn(self, pair, winner):
        self.oful.update(pair[1], int(winner == 1))
        self.right_history.append(pair[1].copy())
        self.epoch_rounds += 1
        if self.epoch_rounds == self.epoch_length:
            self.left_history = np.array(self.right_history)
            self.right_history = []
            self.completed_epochs += 1
            self._new_epoch = True

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "epoch_index": self.epoch_index,
            "epoch_length": self.epoch_length,
            "epoch_rounds": self.epoch_rounds,
            "left_multiset_size": len(self.left_history),
            "current_right_history_size": len(self.right_history),
            "completed_epochs": self.completed_epochs,
            "epoch_transition_pending": self._new_epoch,
            "oful": self.oful.diagnostics(),
        }
