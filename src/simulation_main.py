"""Run sphere cases and the K-by-noise grid with the manual K-RACER implementation.

Each method selects a post-startup cutoff and its tuned parameter using
separate validation seeds, then saves a full run for each test seed.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parent.parent

from config import RecommenderConfig, SimConfig, UserConfig
from simulation_function import RECOMMENDERS, run_simulation
from cutoff_validation import validate_grid
# BEGIN T0 FORKING CHANGE: the search evaluator only returns validation scores.
from cutoff_search import evaluate_cutoffs
# END T0 FORKING CHANGE.

DEFAULT_CUTOFFS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
GAMMA_VALUES = (0.8, 0.9, 0.95)
GATE_SIZES = (32.0, 64.0, 128.0)


def study_cases():
    """Define the cases locally; noise is sigma_u * sqrt(dimension)."""
    return [
        {"id": "baseline", "label": "Baseline", "dimension": 5, "K": 2, "noise": 0.10},
        {"id": "dimension_3", "label": "D=3", "dimension": 3, "K": 2, "noise": 0.10},
        {"id": "dimension_9", "label": "D=9", "dimension": 9, "K": 2, "noise": 0.10},
        {"id": "speed_0.05", "label": "Faster-learning user", "dimension": 5, "K": 2, "noise": 0.05},
        {"id": "speed_0.15", "label": "Slower-learning user", "dimension": 5, "K": 2, "noise": 0.15},
        {"id": "K_3", "label": "K=3", "dimension": 5, "K": 3, "noise": 0.10},
        {"id": "K_5", "label": "K=5", "dimension": 5, "K": 5, "noise": 0.10},
        {"id": "K_9", "label": "K=9", "dimension": 5, "K": 9, "noise": 0.10},
        {"id": "grid_K3_noise_0.05", "label": "K=3, noise=0.05", "dimension": 5, "K": 3, "noise": 0.05},
        {"id": "grid_K5_noise_0.05", "label": "K=5, noise=0.05", "dimension": 5, "K": 5, "noise": 0.05},
        {"id": "grid_K9_noise_0.05", "label": "K=9, noise=0.05", "dimension": 5, "K": 9, "noise": 0.05},
        {"id": "grid_K3_noise_0.15", "label": "K=3, noise=0.15", "dimension": 5, "K": 3, "noise": 0.15},
        {"id": "grid_K5_noise_0.15", "label": "K=5, noise=0.15", "dimension": 5, "K": 5, "noise": 0.15},
        {"id": "grid_K9_noise_0.15", "label": "K=9, noise=0.15", "dimension": 5, "K": 9, "noise": 0.15},
    ]


# BEGIN T0 FORKING CHANGE: independent runs and searches share config creation.
def case_configs(case, seed, horizon, cutoff, *, gamma_ctr=0.5, gate_size=4.0):
    sim = SimConfig(
        ambient_dimension=case["dimension"],
        num_steps=horizon,
        seed=seed,
        user_config=UserConfig(sigma_u=case["noise"] / math.sqrt(case["dimension"]), risk=1.0),
        delta_conf=0.05,
    )
    rec = RecommenderConfig(
        slate_size=case["K"],
        cutoff=cutoff,
        gamma_ctr=gamma_ctr,
        analytical_gate_size=gate_size,
    )
    return sim, rec


def run_cutoff_search(case, seed, horizon, cutoffs, algorithm="kracer", *, gamma_ctr=0.5, gate_size=4.0):
    sim, rec = case_configs(case, seed, horizon, max(cutoffs), gamma_ctr=gamma_ctr, gate_size=gate_size)
    return evaluate_cutoffs(sim, rec, cutoffs, algorithm)
# END T0 FORKING CHANGE.


def run_case(case, seed, horizon, cutoff, algorithm="kracer", *, gamma_ctr=0.5, gate_size=4.0):
    """Run one scenario and return its data in JSON-friendly form."""
    # BEGIN T0 FORKING CHANGE: serialization and held-out runs remain independent.
    sim, rec = case_configs(case, seed, horizon, cutoff, gamma_ctr=gamma_ctr, gate_size=gate_size)
    # END T0 FORKING CHANGE.
    result = run_simulation(sim, rec, algorithm, cutoff_after_startup=True)
    preference = result.user.user_state.preference

    rounds = []
    for interaction in result.interactions:
        items = [item.emb for item in interaction.slate]
        mean_utility = sum(float(preference @ item) for item in items) / len(items)
        rounds.append({
            "t": interaction.timestep,
            "phase": interaction.phase.name,
            "slate": [item.tolist() for item in items],
            "winner": interaction.chosen_id,
            "utility": float(interaction.utility),
            "experience": float(interaction.experience),
            "consumed_regret": 1.0 - float(interaction.utility),
            "slate_average_regret": 1.0 - mean_utility,
        })

    startup = result.startup.state
    return {
        "case": case["id"],
        "algorithm": algorithm,
        "label": case["label"],
        "seed": seed,
        "settings": {
            "dimension": sim.ambient_dimension,
            "slate_size": rec.slate_size,
            "horizon": horizon,
            "post_startup_cutoff": rec.cutoff,
            "absolute_cutoff": min(horizon, result.startup.state.t - 1 + rec.cutoff)
            if result.startup.state.status == "CERTIFIED" else None,
            "sigma_u": sim.user_config.sigma_u,
            "normalized_noise": case["noise"],
            "risk": sim.user_config.risk,
            "delta_conf": sim.delta_conf,
            "gamma_ctr": rec.gamma_ctr,
            "analytical_gate_size": rec.analytical_gate_size,
            "startup_half_angle_radians": result.startup.half_angle,
        },
        "preference": preference.tolist(),
        "startup": {
            "status": startup.status,
            "rounds": startup.t - 1,
            "counts": startup.counts.tolist(),
            "signs": startup.signs.tolist(),
            "margins": [float(x) if math.isfinite(x) else None for x in startup.margins],
            "constraints": [
                {"t": c.t, "coordinate": c.coordinate, "sign": c.sign, "margin": c.margin}
                for c in startup.constraints
            ],
        },
        "summary": {
            "phase_counts": dict(Counter(x["phase"] for x in rounds)),
            "consumed_regret": sum(x["consumed_regret"] for x in rounds),
            "slate_average_regret": sum(x["slate_average_regret"] for x in rounds),
        },
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS,
                        help="candidate post-startup exploration budgets")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=GAMMA_VALUES)
    parser.add_argument("--gate-sizes", type=float, nargs="+", default=GATE_SIZES)
    parser.add_argument("--validation-seed-count", type=int, default=4)
    parser.add_argument("--validation-start-seed", type=int, default=120000)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=6000)
    parser.add_argument("--output", type=Path, help="new directory for saved JSON files")
    parser.add_argument("--algorithms", nargs="+", choices=RECOMMENDERS, default=list(RECOMMENDERS))
    args = parser.parse_args()
    if args.horizon < 1 or args.seed_count < 1 or args.validation_seed_count < 1:
        parser.error("horizon and seed counts must be positive")
    if args.start_seed < 0 or args.validation_start_seed < 0 or any(c < 0 for c in args.cutoffs):
        parser.error("seeds and cutoffs must be nonnegative")
    if any(not math.isfinite(x) or not 0 < x < 1 for x in args.gamma_values):
        parser.error("gamma values must be finite and between 0 and 1")
    if any(not math.isfinite(x) or x <= 0 for x in args.gate_sizes):
        parser.error("gate sizes must be positive and finite")
    cutoffs = sorted({min(c, args.horizon) for c in args.cutoffs})
    validation_seeds = list(range(args.validation_start_seed, args.validation_start_seed + args.validation_seed_count))
    test_seeds = list(range(args.start_seed, args.start_seed + args.seed_count))
    if set(validation_seeds) & set(test_seeds):
        parser.error("validation and test seeds must not overlap")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output or PROJECT / "output" / f"manual_cases_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)

    cases = study_cases()
    algorithms = list(dict.fromkeys(args.algorithms))
    total_runs = len(cases) * sum(
        len(cutoffs) * len(set(args.gate_sizes if algorithm == "analytical_certificate" else args.gamma_values))
        * len(validation_seeds) + len(test_seeds)
        for algorithm in algorithms
    )
    completed = failed_runs = skipped = 0

    def show_progress():
        remaining = total_runs - completed - failed_runs - skipped
        width = len(str(total_runs))
        print(f"Runs: {completed:{width}} completed, {remaining:{width}} remaining, "
              f"{failed_runs:{width}} failed, {skipped:{width}} skipped (total {total_runs})", flush=True)

    # BEGIN T0 FORKING CHANGE: count logical candidates, including cached scores.
    def run_with_progress(*run_args, evaluate=run_case, **run_kwargs):
        nonlocal completed, failed_runs
        try:
            result = evaluate(*run_args, **run_kwargs)
        except Exception:
            failed_runs += 1
            raise
        else:
            completed += 1
            return result
        finally:
            show_progress()
    # END T0 FORKING CHANGE.

    failures = 0
    show_progress()
    for case in cases:
        case_dir = output / case["id"]
        case_dir.mkdir()
        for algorithm in algorithms:
            algorithm_dir = case_dir / algorithm
            algorithm_dir.mkdir()
            parameter_name = "analytical_gate_size" if algorithm == "analytical_certificate" else "gamma_ctr"
            parameters = args.gate_sizes if algorithm == "analytical_certificate" else args.gamma_values
            attempts_before = completed + failed_runs

            def parameters_for(value):
                if algorithm == "analytical_certificate":
                    return {"gamma_ctr": 0.5, "gate_size": value}
                return {"gamma_ctr": value, "gate_size": 4.0}

            # BEGIN T0 FORKING CHANGE: each seed/parameter shares one T0 search.
            validation_cache = {}

            def cached_score(seed, cutoff, parameter):
                key = (seed, parameter)
                if key not in validation_cache:
                    validation_cache[key] = run_cutoff_search(
                        case, seed, args.horizon, cutoffs, algorithm, **parameters_for(parameter)
                    )
                return validation_cache[key][cutoff]

            def validation_score(seed, cutoff, parameter):
                print(f"Validating {case['id']}, {algorithm}, cutoff {cutoff}, {parameter_name} {parameter}, seed {seed}...", flush=True)
                return run_with_progress(seed, cutoff, parameter, evaluate=cached_score)
            # END T0 FORKING CHANGE.

            try:
                selected, selected_parameter, scores = validate_grid(cutoffs, parameters, validation_seeds, validation_score)
                validation = {"selected_cutoff": selected, "parameter_name": parameter_name,
                              "selected_parameter": selected_parameter, "validation_seeds": validation_seeds,
                              "test_seeds": test_seeds, "scores": scores}
            except Exception as error:
                validation = {"error": f"{type(error).__name__}: {error}",
                              "validation_seeds": validation_seeds, "cutoffs": cutoffs,
                              "parameter_name": parameter_name, "parameters": sorted(set(parameters))}
                print(f"  Validation failed: {validation['error']}", flush=True)
                failures += 1
            with (algorithm_dir / "validation.json").open("x", encoding="utf-8") as file:
                json.dump(validation, file, indent=2, allow_nan=False)
            if "error" in validation:
                planned = len(cutoffs) * len(set(parameters)) * len(validation_seeds) + len(test_seeds)
                skipped += planned - (completed + failed_runs - attempts_before)
                show_progress()
                continue

            for seed in test_seeds:
                print(f"Testing {case['id']}, {algorithm}, cutoff {selected}, {parameter_name} {selected_parameter}, seed {seed}...", flush=True)
                try:
                    data = run_with_progress(case, seed, args.horizon, selected, algorithm, **parameters_for(selected_parameter))
                except Exception as error:
                    data = {"case": case["id"], "algorithm": algorithm, "seed": seed,
                            "error": f"{type(error).__name__}: {error}"}
                    print(f"  Failed: {data['error']}", flush=True)
                    failures += 1
                with (algorithm_dir / f"seed_{seed}.json").open("x", encoding="utf-8") as file:
                    json.dump(data, file, indent=2, allow_nan=False)

    print(f"Results saved in {output}")
    if failures:
        print(f"{failures} run(s) failed; see their JSON files for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
