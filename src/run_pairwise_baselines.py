"""Run the pairwise baselines on the manual experiment's K=2 cases.

RAES validates an absolute cutoff. DBGD, Doubler and Sparring learn for the
full horizon. No baseline uses K-RACER's coordinate-sign startup.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from config import RecommenderConfig, SimConfig, UserConfig
from cutoff_validation import validate_cutoffs
from models.interaction import Interaction
from recommenders.dbgd import DBGD
from recommenders.doubler import Doubler
from recommenders.raes_reworked import RAESReworked
from recommenders.sparring import Sparring
from simulation_main import DEFAULT_CUTOFFS, study_cases
from users.bayesian_user import BayesianUser
from users.user_state import UserState


BASELINES = {"raes": RAESReworked, "dbgd": DBGD, "doubler": Doubler, "sparring": Sparring}


def run_case(case, seed, horizon, algorithm, cutoff=None, *, save_rounds=True):
    sim = SimConfig(
        ambient_dimension=case["dimension"], num_steps=horizon, seed=seed,
        user_config=UserConfig(sigma_u=case["noise"] / math.sqrt(case["dimension"]), risk=1.0),
        delta_conf=0.05,
    )
    rec = RecommenderConfig(slate_size=2, cutoff=cutoff if cutoff is not None else horizon)
    user = BayesianUser(UserState(sim.user_config, sim))
    recommender = BASELINES[algorithm](sim, rec)
    preference = user.user_state.preference
    rounds = []
    total_slate_regret = 0.0

    for t in range(1, horizon + 1):
        slate = recommender.recommend(2)
        chosen = user.choose(slate)
        winner = next(i for i, item in enumerate(slate) if item is chosen)
        interaction = Interaction(slate, winner, chosen, 0.0, 0.0, recommender.phase, t)
        user.update(interaction)
        recommender.update(interaction)
        slate_regret = 1.0 - sum(float(preference @ item.emb) for item in slate) / 2
        total_slate_regret += slate_regret
        if save_rounds:
            phase = recommender.phase.name if algorithm == "raes" else algorithm.upper()
            rounds.append({
                "t": t, "phase": phase, "slate": [item.emb.tolist() for item in slate],
                "winner": winner, "utility": float(interaction.utility),
                "experience": float(interaction.experience),
                "consumed_regret": 1.0 - float(interaction.utility),
                "slate_average_regret": slate_regret,
            })

    if not save_rounds:
        return total_slate_regret

    # The angle identifies the corresponding K-RACER case for plotting; these
    # baselines do not run startup or use a cone.
    reference_angle = (math.acos(1 / math.sqrt(case["dimension"])) + math.pi / 2) / 2
    return {
        "case": case["id"], "algorithm": algorithm, "label": case["label"], "seed": seed,
        "settings": {
            "dimension": case["dimension"], "slate_size": 2, "horizon": horizon,
            "post_startup_cutoff": cutoff, "absolute_cutoff": cutoff,
            "cutoff_reference": "absolute" if cutoff is not None else "not applicable",
            "sigma_u": sim.user_config.sigma_u, "normalized_noise": case["noise"],
            "risk": sim.user_config.risk, "delta_conf": sim.delta_conf,
            "startup_half_angle_radians": reference_angle, "startup_applies": False,
            "policy_variant": "fixed",
        },
        "preference": preference.tolist(),
        "startup": {"status": "NOT_APPLICABLE", "rounds": 0},
        "summary": {
            "phase_counts": dict(Counter(row["phase"] for row in rounds)),
            "consumed_regret": sum(row["consumed_regret"] for row in rounds),
            "slate_average_regret": sum(row["slate_average_regret"] for row in rounds),
        },
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=BASELINES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=6000)
    parser.add_argument("--validation-seed-count", type=int, default=4)
    parser.add_argument("--validation-start-seed", type=int, default=120000)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    args = parser.parse_args()
    if args.horizon < 1 or args.seed_count < 1 or args.validation_seed_count < 1:
        parser.error("horizon and seed counts must be positive")
    if args.start_seed < 0 or args.validation_start_seed < 0 or any(x < 0 for x in args.cutoffs):
        parser.error("seeds and cutoffs must be nonnegative")
    test_seeds = list(range(args.start_seed, args.start_seed + args.seed_count))
    validation_seeds = list(range(args.validation_start_seed, args.validation_start_seed + args.validation_seed_count))
    if set(test_seeds) & set(validation_seeds):
        parser.error("validation and test seeds must not overlap")
    cutoffs = sorted({min(value, args.horizon) for value in args.cutoffs})
    cases = [case for case in study_cases() if case["K"] == 2]
    total = len(cases) * (len(test_seeds) + (len(cutoffs) * len(validation_seeds) if args.algorithm == "raes" else 0))
    completed = 0
    failures = 0
    skipped = 0
    args.output.mkdir(parents=True, exist_ok=False)

    def show_progress():
        width = len(str(total))
        print(f"Runs: {completed:{width}} completed, {total-completed-failures-skipped:{width}} remaining, "
              f"{failures:{width}} failed, {skipped:{width}} skipped (total {total})", flush=True)

    def run(case, seed, cutoff, *, save_rounds):
        nonlocal completed, failures
        try:
            result = run_case(case, seed, args.horizon, args.algorithm, cutoff, save_rounds=save_rounds)
        except Exception:
            failures += 1
            raise
        else:
            completed += 1
            return result
        finally:
            show_progress()

    show_progress()
    for case in cases:
        directory = args.output / case["id"] / args.algorithm
        directory.mkdir(parents=True)
        cutoff = None
        if args.algorithm == "raes":
            attempts_before = completed + failures
            def validation_score(seed, value):
                print(f"Validating {case['id']}, raes, cutoff {value}, seed {seed}...", flush=True)
                return run(case, seed, value, save_rounds=False)
            try:
                cutoff, scores = validate_cutoffs(cutoffs, validation_seeds, validation_score)
                validation = {"selected_cutoff": cutoff, "parameter_name": "policy_variant",
                    "selected_parameter": "fixed", "validation_seeds": validation_seeds,
                    "test_seeds": test_seeds,
                    "scores": [{"parameter": "fixed", **score} for score in scores]}
            except Exception as error:
                print(f"Validation failed for {case['id']}: {error}", flush=True)
                skipped += len(cutoffs) * len(validation_seeds) + len(test_seeds) - (completed + failures - attempts_before)
                show_progress()
                continue
            with (directory / "validation.json").open("x", encoding="utf-8") as file:
                json.dump(validation, file, indent=2, allow_nan=False)
        for seed in test_seeds:
            print(f"Testing {case['id']}, {args.algorithm}, seed {seed}...", flush=True)
            try:
                data = run(case, seed, cutoff, save_rounds=True)
            except Exception as error:
                data = {"case": case["id"], "algorithm": args.algorithm, "seed": seed,
                        "error": f"{type(error).__name__}: {error}"}
                print(f"Run failed: {data['error']}", flush=True)
            with (directory / f"seed_{seed}.json").open("x", encoding="utf-8") as file:
                json.dump(data, file, indent=2, allow_nan=False)
    print(f"Results saved in {args.output}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
