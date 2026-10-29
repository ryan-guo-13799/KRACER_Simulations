import math

import numpy as np

from config import SimConfig, RecommenderConfig
from models.interaction import Interaction
from models.item import Item
from models.phase import Phase
from protocols.recommender import Recommender
from recommenders.KRACER_functions import Cut, EQ, ExcitationSlate, Observe, Restart, TreeCertificate
from recommenders.KRACER_state import KRACERState


class KRACER(Recommender):
    """K-RACER starting from a supplied certified cone.

    Call recommend once, then update with the user's response. The initial
    cone must already contain the user's preference; startup is separate.
    """

    def __init__(
        self,
        state: KRACERState,
        sim_config: SimConfig,
        recommender_config: RecommenderConfig,
        *,
        a_geo: float | None = None,
        start_time: int = 1,
    ) -> None:
        self.state = state
        self.config = sim_config
        if not isinstance(start_time, int) or isinstance(start_time, bool) or not 1 <= start_time <= sim_config.num_steps + 1:
            raise ValueError("start_time must be between 1 and num_steps + 1")
        self.t = start_time
        self.phase: Phase | None = None
        self.cutoff = recommender_config.cutoff # the T_0
        self.K = recommender_config.slate_size
        self.dim = self.config.ambient_dimension
        self._slate: list[Item] | None = None
        self._remaining_depth = 0

        self._validate_cone()

        self.user_lambda0 = sim_config.user_config.lambda0
        self._initialize_information()

        self._validate_exploration(recommender_config.gamma_ctr)
        self.gamma_ctr = recommender_config.gamma_ctr
        self.a_geo = self.gamma_ctr / (4 * self.K) if a_geo is None else a_geo
        self._validate_a_geo()

    def recommend(self, k: int) -> list[Item]:
        """Choose one slate; the next call must be update."""
        self._validate_recommend(k)

        state = self.state
        if self.t > self.cutoff:
            self.phase = Phase.EXPLOIT
            self._slate = [Item(self.estimate()) for _ in range(k)]
        else:
            if self._remaining_depth == 0:
                self._attempt() # may change the _remaining_depth and enter the next if
            if self._remaining_depth:
                self.phase = Phase.LOCALIZE
                self._slate = EQ(state.I, state.c, state.F[:, state.i], k)
            else:
                self.phase = Phase.EXCITE
                self._slate = self._excitation()
        return self._slate

    def _attempt(self) -> None:
        """Freeze the cone and find the first admissible localization depth."""
        self._prepare_interval()
        max_depth = min(self._max_depth(), self.cutoff - self.t + 1)
        if max_depth > 0:
            self._remaining_depth = self._find_depth(max_depth) or 0

    # The following functions are shared functions across ablations. The ablactions override only the function they change.

    def _prepare_interval(self) -> None:
        state = self.state
        state.i = int(np.argmax(state.W))
        alpha = float(np.arctan(np.sqrt(state.W[state.i])))
        state.I = (-alpha, alpha)

    def _max_depth(self) -> int:
        alpha = float(np.arctan(np.sqrt(self.state.W[self.state.i])))
        d = self.dim - 1
        B_d = d / (d - 1)
        C_d = np.sqrt(d) * B_d ** ((d - 1) / 2)
        G = 2 * alpha / np.sin(2 * alpha)
        return math.floor(math.log(C_d * G / self.a_geo, self.K))

    def _find_depth(self, max_depth: int) -> int | None:
        state = self.state
        return TreeCertificate(
            state.I, state.V, self.t, max_depth, state.c, state.F,
            state.W, state.i, self.K, self.config.h_t, self.gamma_ctr,
        )

    def _excitation(self) -> list[Item]:
        state = self.state
        alpha = float(np.arctan(np.sqrt(state.W[state.i])))
        return ExcitationSlate(
            (-alpha, alpha), state.V, state.c, state.F[:, state.i], self.K, self.user_lambda0
        )

    def _cut(self, interval, V, h, winner):
        state = self.state
        return Cut(interval, V, h, winner, state.c, state.F[:, state.i],
                   self.K, state.W, state.i)

    def _finish_round(self) -> None:
        state = self.state
        state.c, state.F, state.W = Restart(state.c, state.F, state.W, state.i, state.I)

    def estimate(self):
        return self.state.c.copy()

    def update(self, interaction: Interaction) -> None:
        """Use the public winner to update the interval and information."""
        winner = self._validate_update(interaction)

        state = self.state
        if self.phase == Phase.LOCALIZE:
            # The cut uses the information matrix before this response.
            state.I, _ = self._cut(state.I, state.V, self.config.h_t(self.t), winner)

        _, state.V, next_t = Observe(self._slate, winner, state.V, self.t)
        if self.phase == Phase.LOCALIZE:
            self._remaining_depth -= 1
            if self._remaining_depth == 0:
                self._finish_round()

        self.t = next_t
        self._slate = None

    # Input and interaction validation.

    def _validate_cone(self) -> None:
        state = self.state
        sim_config = self.config
        if self.dim != sim_config.ambient_dimension or self.dim < 3:
            raise ValueError("state and simulation need the same dimension, at least 3")
        if not isinstance(self.K, int) or self.K < 2:
            raise ValueError("the slate size must be at least 2")
        if state.c is None or state.F is None or state.W is None:
            raise ValueError("provide an initial certified cone in the state")
        D = self.dim
        if state.c.shape != (D,) or state.F.shape != (D, D - 1) or state.W.shape != (D - 1,):
            raise ValueError("cone shapes must be (D,), (D, D-1), and (D-1,)")
        frame = np.column_stack((state.c, state.F))
        if not np.allclose(frame.T @ frame, np.eye(D), rtol=0, atol=1e-8):
            raise ValueError("the cone axis and frame must be orthonormal")
        if not np.all(np.isfinite(state.W)) or np.any(state.W <= 0):
            raise ValueError("cone widths must be positive and finite")


    def _initialize_information(self) -> None:
        state = self.state
        D = self.dim
        if not math.isfinite(self.user_lambda0) or self.user_lambda0 <= 0:
            raise ValueError("the user's initial information must be positive")
        if state.V is None:
            state.V = self.user_lambda0 * np.eye(D)
        if state.V.shape != (D, D) or not np.all(np.isfinite(state.V)):
            raise ValueError("V must be a finite D by D matrix")
        if not np.allclose(state.V, state.V.T, rtol=0, atol=1e-12):
            raise ValueError("V must be symmetric")
        if np.min(np.linalg.eigvalsh(state.V)) < self.user_lambda0 * (1 - 1e-10):
            raise ValueError("V must be at least lambda0 times the identity")


    def _validate_exploration(self, gamma_ctr: float) -> None:
        sim_config = self.config
        if not isinstance(self.cutoff, int) or not 0 <= self.cutoff <= sim_config.num_steps:
            raise ValueError("exploration_steps must be between 0 and num_steps")
        if not 0 < gamma_ctr < 1:
            raise ValueError("gamma_ctr must be between 0 and 1")

    def _validate_a_geo(self) -> None:
        gamma_ctr = self.gamma_ctr
        state = self.state
        if not 0 < self.a_geo < gamma_ctr / self.K:
            raise ValueError("a_geo must be between 0 and gamma_ctr / K")

    def _validate_recommend(self, k: int) -> None:
        if self._slate is not None:
            raise RuntimeError("update the previous recommendation first")
        if k != self.K:
            raise ValueError("k must match the configured slate size")
        if self.t > self.config.num_steps:
            raise RuntimeError("the simulation has finished")

    def _validate_update(self, interaction: Interaction) -> int:
        if self._slate is None:
            raise RuntimeError("recommend a slate before updating")
        winner = interaction.chosen_id
        if not isinstance(winner, (int, np.integer)) or isinstance(winner, (bool, np.bool_)) or not 0 <= winner < len(self._slate):
            raise ValueError("chosen_id must be a position in the recommended slate")
        if not np.array_equal(interaction.chosen_item.emb, self._slate[winner].emb):
            raise ValueError("chosen_item must match the winner in the recommended slate")
        return winner
