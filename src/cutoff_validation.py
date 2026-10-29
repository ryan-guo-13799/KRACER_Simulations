"""Choose a post-startup cutoff using validation seeds only."""

import math


def validate_cutoffs(cutoffs, seeds, evaluate):
    """Evaluate each cutoff and return the best and its validation scores.

    evaluate(seed, cutoff) returns the run's total slate-average regret.
    Exact ties choose the smaller cutoff.
    """
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
    """Select a cutoff and parameter using mean validation slate regret.

    evaluate(seed, cutoff, parameter) returns one run's slate regret. Ties
    prefer the smaller cutoff, then the smaller parameter, as in the study.
    """
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
