from dataclasses import dataclass, replace

from config import RecommenderConfig, SimConfig, UserConfig
from models.interaction import Interaction
from recommenders.KRACER import KRACER
from recommenders.KRACER_state import KRACERState
from recommenders.analytical_certificate import AnalyticalCertificate
from recommenders.minimum_eigenvector import MinimumEigenvector
from recommenders.without_restart import WithoutRestart
from recommenders.startup import Startup
from users.bayesian_user import BayesianUser
from users.user_state import UserState


# All variants use the same startup, user, interaction loop, and absolute cutoff.
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
    recommender: KRACER
    user: BayesianUser


def run_simulation(
    sim_config: SimConfig,
    recommender_config: RecommenderConfig,
    algorithm: str = "kracer",
    *,
    cutoff_after_startup: bool = False,
) -> SimulationResult:
    """Run startup, then K-RACER if startup certifies a cone.

    By default cutoff is an absolute simulation time. With cutoff_after_startup,
    it is an additional exploration budget after startup. A run whose horizon
    ends during startup returns PENDING.
    """
    if algorithm not in RECOMMENDERS:
        raise ValueError(f"unknown recommender: {algorithm}")
    if not isinstance(recommender_config.cutoff, int) or not 0 <= recommender_config.cutoff <= sim_config.num_steps:
        raise ValueError("cutoff must be between 0 and num_steps")
    user = BayesianUser(UserState(sim_config.user_config, sim_config))
    startup = Startup(sim_config, recommender_config)
    interactions = []

    def interact(recommender, t: int) -> None:
        slate = recommender.recommend(recommender_config.slate_size)
        chosen = user.choose(slate)
        winner = next(i for i, item in enumerate(slate) if item is chosen)
        interaction = Interaction(slate, winner, chosen, 0.0, 0.0, recommender.phase, t)
        user.update(interaction)
        recommender.update(interaction)
        interactions.append(interaction)

    while startup.state.status == "PENDING" and not startup.exhausted:
        interact(startup, startup.state.t)

    if startup.state.status != "CERTIFIED":
        return SimulationResult(interactions, startup, None, user)

    state = KRACERState(recommender_config, sim_config)
    state.c, state.F, state.W = startup.cone()
    state.V = startup.state.V.copy()
    if cutoff_after_startup:
        deadline = min(sim_config.num_steps, startup.state.t - 1 + recommender_config.cutoff)
        recommender_config = replace(recommender_config, cutoff=deadline)
    recommender = RECOMMENDERS[algorithm](state, sim_config, recommender_config, start_time=startup.state.t)
    while recommender.t <= sim_config.num_steps:
        interact(recommender, recommender.t)
    return SimulationResult(interactions, startup, recommender, user)
