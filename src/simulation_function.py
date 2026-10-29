from config import RecommenderConfig, SimConfig, UserConfig
# BEGIN T0 FORKING CHANGE: retain the public API; use the shared step engine.
from simulation_state import RECOMMENDERS, SimulationResult, SimulationState
# END T0 FORKING CHANGE.


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
    # BEGIN T0 FORKING CHANGE: independent runs use exactly the same engine.
    return SimulationState(sim_config, recommender_config, algorithm,
                           cutoff_after_startup=cutoff_after_startup).finish()
    # END T0 FORKING CHANGE.
