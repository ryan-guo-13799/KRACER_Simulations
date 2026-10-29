"""Evaluate T0 branches and select policies using validation seeds only."""

# BEGIN T0 FORKING CHANGE: keep prefix sharing and validation scoring separate
# from the simulation engine and the manually written algorithm classes.
from copy import deepcopy
from dataclasses import replace
import math

from simulation_state import RECOMMENDERS, SimulationState


# These policies use T0 to limit their next localization depth. Unknown
# policies conservatively fork immediately after startup and run normally.
_SHAREABLE = tuple(RECOMMENDERS[name] for name in (
    "kracer", "analytical_certificate", "minimum_eigenvector", "without_restart",
))


def _can_share_next_step(state, cutoff):
    agent = state.recommender
    if agent is None:
        return True  # Startup is independent of the post-startup T0 budget.
    remaining = state.deadline_for(cutoff) - state.t + 1
    if remaining <= 0 or type(agent) not in _SHAREABLE:
        return False
    if agent._remaining_depth:
        return remaining >= agent._remaining_depth
    # Prepare only a copy: probing the next axis must not mutate the live run.
    probe = deepcopy(agent)
    probe._prepare_interval()
    return remaining >= probe._max_depth()


def _exploiting(state):
    agent = state.recommender
    return agent is not None and type(agent) in _SHAREABLE and state.t > agent.cutoff


def _finish_score(state):
    while not state.done:
        if _exploiting(state):
            # All remaining positions repeat one fixed estimate. No later
            # private experience can change regret, so only add their losses.
            utility = float(state.user.user_state.preference @ state.recommender.estimate())
            K = state.recommender_config.slate_size
            loss = 1.0 - sum([utility] * K) / K
            remaining = state.sim_config.num_steps - state.t + 1
            # Sum the same ordered loss sequence as the independent runner,
            # including Python versions that use compensated float summation.
            return sum(state.slate_losses + [loss] * remaining)
        state.step()
    return state.slate_regret


def evaluate_cutoffs(sim_config, recommender_config, cutoffs, algorithm="kracer"):
    """Return {post_startup_T0: final_slate_regret} for one seed/parameter.

    The largest budget owns the live run. Fork smaller budgets before their
    next action might differ; never share user histories after that point.
    """
    grid = sorted(set(cutoffs))
    if not grid:
        raise ValueError("at least one cutoff is required")
    for cutoff in grid:
        SimulationState._validate_cutoff(cutoff, sim_config.num_steps)
    state = SimulationState(sim_config, replace(recommender_config, cutoff=grid[-1]),
                            algorithm, cutoff_after_startup=True)
    pending = set(grid[:-1])
    scores = {}
    while not state.done and not _exploiting(state):
        for cutoff in sorted(pending):
            if not _can_share_next_step(state, cutoff):
                scores[cutoff] = _finish_score(state.fork(cutoff))
                pending.remove(cutoff)
        state.step()
    scores[grid[-1]] = _finish_score(state)
    # Early policy stops and incomplete startup leave a common suffix.
    for cutoff in sorted(pending):
        scores[cutoff] = _finish_score(state.fork(cutoff))
    return {cutoff: scores[cutoff] for cutoff in grid}
# END T0 FORKING CHANGE.


# BEGIN RUNNER CLEANUP: selection belongs beside cutoff evaluation.
def validate_cutoffs(cutoffs, seeds, evaluate):
    """Choose a cutoff by mean validation slate regret; ties prefer smaller T0."""
    scores = []
    for cutoff in sorted(set(cutoffs)):
        values = [float(evaluate(seed, cutoff)) for seed in seeds]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("validation scores must be finite and nonempty")
        scores.append({"cutoff": cutoff, "seed_scores": values,
                       "mean_slate_regret": math.fsum(values) / len(values)})
    if not scores:
        raise ValueError("at least one cutoff is required")
    best = min(scores, key=lambda score: (score["mean_slate_regret"], score["cutoff"]))
    return best["cutoff"], scores


def validate_grid(cutoffs, parameters, seeds, evaluate):
    """Choose a parameter/T0 pair; ties prefer smaller T0, then parameter."""
    scores = []
    for parameter in sorted(set(parameters)):
        _, cutoff_scores = validate_cutoffs(
            cutoffs, seeds, lambda seed, cutoff: evaluate(seed, cutoff, parameter)
        )
        for score in cutoff_scores:
            scores.append({"parameter": parameter, **score})
    if not scores:
        raise ValueError("at least one parameter is required")
    best = min(scores, key=lambda score: (score["mean_slate_regret"],
                                          score["cutoff"], score["parameter"]))
    return best["cutoff"], best["parameter"], scores
# END RUNNER CLEANUP.
