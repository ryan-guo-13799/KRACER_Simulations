"""One resumable simulation; algorithms and cutoff search stay separate."""

# BEGIN T0 FORKING CHANGE: extract the existing interaction loop into state
# that can be stepped, completed, or copied without changing the algorithms.
from copy import deepcopy
from dataclasses import dataclass, replace

from models.interaction import Interaction
from recommenders.KRACER import KRACER
from recommenders.KRACER_state import KRACERState
from recommenders.analytical_certificate import AnalyticalCertificate
from recommenders.minimum_eigenvector import MinimumEigenvector
from recommenders.without_restart import WithoutRestart
from recommenders.startup import Startup
from users.bayesian_user import BayesianUser
from users.user_state import UserState


RECOMMENDERS = {
    "kracer": KRACER,
    "analytical_certificate": AnalyticalCertificate,
    "minimum_eigenvector": MinimumEigenvector,
    "without_restart": WithoutRestart,
}


@dataclass
class SimulationResult:
    interactions: list[Interaction]
    startup: Startup
    recommender: KRACER | None
    user: BayesianUser


class SimulationState:
    def __init__(self, sim_config, recommender_config, algorithm="kracer", *, cutoff_after_startup=False):
        if algorithm not in RECOMMENDERS:
            raise ValueError(f"unknown recommender: {algorithm}")
        self._validate_cutoff(recommender_config.cutoff, sim_config.num_steps)
        self.sim_config = sim_config
        self.recommender_config = recommender_config
        self.algorithm = algorithm
        self.cutoff_after_startup = cutoff_after_startup
        self.user = BayesianUser(UserState(sim_config.user_config, sim_config))
        self.startup = Startup(sim_config, recommender_config)
        self.recommender = None
        self.interactions = []
        self.slate_losses = []

    @staticmethod
    def _validate_cutoff(cutoff, horizon):
        if not isinstance(cutoff, int) or not 0 <= cutoff <= horizon:
            raise ValueError("cutoff must be between 0 and num_steps")

    @property
    def t(self):
        """One-based time of the next interaction."""
        return self.startup.state.t if self.recommender is None else self.recommender.t

    @property
    def done(self):
        return self.t > self.sim_config.num_steps

    @property
    def slate_regret(self):
        return sum(self.slate_losses)

    def deadline_for(self, cutoff):
        offset = self.startup.state.t - 1 if self.cutoff_after_startup else 0
        return min(self.sim_config.num_steps, offset + cutoff)

    def _start_recommender(self):
        config = replace(self.recommender_config,
                         cutoff=self.deadline_for(self.recommender_config.cutoff))
        state = KRACERState(config, self.sim_config)
        state.c, state.F, state.W = self.startup.cone()
        state.V = self.startup.state.V.copy()
        self.recommender = RECOMMENDERS[self.algorithm](
            state, self.sim_config, config, start_time=self.startup.state.t,
        )

    def step(self):
        if self.done:
            return
        active = self.startup if self.recommender is None else self.recommender
        slate = active.recommend(self.recommender_config.slate_size)
        chosen = self.user.choose(slate)
        winner = next(i for i, item in enumerate(slate) if item is chosen)
        interaction = Interaction(slate, winner, chosen, 0.0, 0.0, active.phase, self.t)
        self.user.update(interaction)
        active.update(interaction)
        self.interactions.append(interaction)
        preference = self.user.user_state.preference
        self.slate_losses.append(1.0 - sum(float(preference @ item.emb) for item in slate) / len(slate))
        if self.recommender is None and self.startup.state.status == "CERTIFIED":
            self._start_recommender()

    def fork(self, cutoff):
        """Copy a prefix for a shorter cutoff, including its private RNG state."""
        self._validate_cutoff(cutoff, self.recommender_config.cutoff)
        branch = deepcopy(self)
        branch.recommender_config = replace(branch.recommender_config, cutoff=cutoff)
        if branch.recommender is not None:
            # An analytical-certificate failure may already have ended learning.
            # A copied budget must preserve that earlier policy stop.
            branch.recommender.cutoff = min(branch.recommender.cutoff, branch.deadline_for(cutoff))
        return branch

    def finish(self):
        while not self.done:
            self.step()
        return SimulationResult(self.interactions, self.startup, self.recommender, self.user)
# END T0 FORKING CHANGE.
