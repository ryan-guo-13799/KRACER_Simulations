import math

import numpy as np

from config import RecommenderConfig, SimConfig
from models.interaction import Interaction
from models.item import Item
from models.phase import Phase
from protocols.recommender import Recommender
from recommenders.KRACER_functions import Observe
from recommenders.startup_functions import make_cone, sign_threshold
from recommenders.startup_state import StartupConstraint, StartupState


class Startup(Recommender):
    """Binary coordinate-sign startup; call recommend then update each round."""

    def __init__(self, sim_config: SimConfig, recommender_config: RecommenderConfig):
        self.config = sim_config
        self.state = StartupState(sim_config)
        self.phase = Phase.STARTUP
        D = sim_config.ambient_dimension
        if D < 3 or recommender_config.slate_size < 2:
            raise ValueError("startup with K-RACER requires D >= 3 and slate_size >= 2")
        if not math.isfinite(sim_config.user_config.lambda0) or sim_config.user_config.lambda0 <= 0:
            raise ValueError("lambda0 must be positive and finite")

        angle = recommender_config.startup_half_angle
        if angle is None:
            angle = (math.acos(1 / math.sqrt(D)) + math.pi / 2) / 2
        self.half_angle = angle
        self.threshold = sign_threshold(D, angle)
        self._slate = None
        self._coordinate = None
        self._margin = None

    @property
    def exhausted(self) -> bool:
        return self.state.status == "PENDING" and self.state.t > self.config.num_steps

    def recommend(self, k: int) -> list[Item]:
        if k < 2:
            raise ValueError("startup needs room for two items")
        if self._slate is not None:
            raise RuntimeError("update the previous recommendation first")
        if self.state.status == "CERTIFIED":
            raise RuntimeError("startup is already certified")
        if self.exhausted:
            raise RuntimeError("simulation horizon ended without a certified cone")

        state = self.state
        remaining_axis = np.flatnonzero(state.signs == 0) # the axis haven't beed cut
        h = self.config.h_t(state.t)
        margins = h / np.sqrt(self.config.user_config.lambda0 + state.counts[remaining_axis]) # margin of each remaining axis
        self._coordinate = int(remaining_axis[np.argmax(margins)])
        self._margin = float(np.max(margins)) # we tackle the largest margin first
        axis = np.eye(self.config.ambient_dimension)[self._coordinate]
        self._slate = [Item(axis), Item(-axis)]
        return self._slate

    def update(self, interaction: Interaction) -> None:
        if self._slate is None:
            raise RuntimeError("recommend a slate before updating")
        winner = interaction.chosen_id
        if not isinstance(winner, (int, np.integer)) or isinstance(winner, (bool, np.bool_)) or winner not in (0, 1):
            raise ValueError("chosen_id must be 0 or 1")
        if not np.array_equal(interaction.chosen_item.emb, self._slate[winner].emb):
            raise ValueError("chosen_item must match the recommended winner")

        state = self.state
        i = self._coordinate
        sign = 1 if winner == 0 else -1
        constraint = StartupConstraint(state.t, i, sign, self._margin)
        _, state.V, state.t = Observe(self._slate, winner, state.V, state.t)
        state.constraints.append(constraint)
        state.counts[i] += 1
        if self._margin <= self.threshold:
            state.signs[i] = sign
            state.margins[i] = self._margin
            if np.all(state.signs != 0):
                state.status = "CERTIFIED"
        self._slate = None
        self._coordinate = None
        self._margin = None

    def cone(self):
        """Return the cone only after every coordinate has a certified sign."""
        if self.state.status != "CERTIFIED":
            raise RuntimeError("startup has not certified a cone")
        return make_cone(self.state.signs, self.half_angle)
