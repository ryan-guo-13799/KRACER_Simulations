"""Ablation: excite the direction with the least accumulated information."""

import numpy as np

from recommenders.KRACER import KRACER
from recommenders.KRACER_functions import ExcitationRing


class MinimumEigenvector(KRACER):
    # EXTRA FOR THIS ABLATION: only the excitation center changes.
    def _excitation(self):
        _, vectors = np.linalg.eigh(self.state.V)
        center = vectors[:, 0]
        if center[np.argmax(np.abs(center))] < 0:
            center = -center
        radius = min(0.25, 1 / np.sqrt(2 * (1 + 1 / self.user_lambda0)))
        return ExcitationRing(center, self.K, radius)
