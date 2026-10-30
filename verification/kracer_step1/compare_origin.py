"""Compare the actual rewrite with the origin; no runtime substitutions.

Run with one numerical thread per process, for example:
    python -B verification/kracer_step1/compare_origin.py

The default check uses all 20 saved baseline seeds and seeds 6000/6001 for
each other origin case, with the origin's saved selected gamma and T0.
This checks matched policies; it does not rerun parameter selection.
"""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import platform
import sys

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin', type=Path, default=PROJECT.parent / 'simulations')
    parser.add_argument('--output', type=Path, default=HERE / 'results.json')
    parser.add_argument('--baseline-seeds', type=int, default=20)
    parser.add_argument('--other-seeds', type=int, default=2)
    args = parser.parse_args()
    sys.path[:0] = [str(PROJECT / 'src'), str(args.origin)]
    from config import RecommenderConfig, SimConfig, UserConfig
    from simulation_state import run_simulation
    from simulation.design import study_cases
    from simulation.episode import EpisodeState

    selected_path = args.origin / 'output/sphere_slow_baseline_20/selected_T0.csv'
    with selected_path.open(newline='', encoding='utf-8') as file:
        selected = {row['case']: row for row in csv.DictReader(file)
                    if row['variant'] == 'adaptive_tree'}

    def fingerprints(directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.rglob('*.py'))}

    document = {
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__},
        'origin': str(args.origin.resolve()),
        'selection_reference': str(selected_path.resolve()),
        'source_hashes': {'rewrite': fingerprints(PROJECT / 'src'),
                          'origin': fingerprints(args.origin / 'simulation')},
        'runs': [],
    }
    from dataclasses import replace
    for case in study_cases():
        choice = selected[case['id']]
        cfg = replace(case['config'], T0=int(choice['T0']),
                      gamma_ctr=float(choice['parameter_value']))
        count = args.baseline_seeds if case['id'] == 'sphere_baseline' else args.other_seeds
        for seed in range(6000, 6000 + count):
            original = EpisodeState(cfg, 'adaptive_tree', seed, audit=False, fast_forward=True)
            origin_rows = []
            choose = original.user.choose

            def record(slate, V, t):
                winner = choose(slate, V, t)
                origin_rows.append((slate.copy(), winner))
                return winner

            original.user.choose = record
            origin_result = original.finish()
            origin_rows.extend([origin_rows[-1]] * (cfg.T - len(origin_rows)))

            sim = SimConfig(ambient_dimension=cfg.D, num_steps=cfg.T, seed=seed,
                            user_config=UserConfig(sigma_u=cfg.sigma_u, risk=cfg.gamma_cara),
                            delta_conf=cfg.delta_conf)
            rec = RecommenderConfig(slate_size=cfg.K, cutoff=cfg.T0,
                                    gamma_ctr=cfg.gamma_ctr,
                                    startup_half_angle=math.radians(cfg.startup_half_angle_deg))
            rewritten = run_simulation(sim, rec, 'kracer', cutoff_after_startup=True)
            rewrite_rows = [(np.array([item.emb for item in x.slate]), x.chosen_id)
                            for x in rewritten.interactions]
            preference = rewritten.user.user_state.preference
            origin_losses = np.array([origin_result.consumed_item_regret,
                                      origin_result.slate_average_regret]).T
            rewrite_losses = np.array([[1 - float(preference @ slate[winner]),
                                        1 - float(np.mean(slate @ preference))]
                                       for slate, winner in rewrite_rows])
            max_slate_error = 0.0
            first_slate_difference = first_winner_difference = first_phase_difference = None
            for t, ((oq, ow), (rq, rw)) in enumerate(zip(origin_rows, rewrite_rows), 1):
                error = float(np.max(abs(oq - rq))) if oq.shape == rq.shape else 2.0
                max_slate_error = max(max_slate_error, error)
                if error > 1e-7 and first_slate_difference is None:
                    first_slate_difference = t
                if ow != rw and first_winner_difference is None:
                    first_winner_difference = t
                op = origin_result.phases[t-1]
                rp = rewritten.interactions[t-1].phase.name.lower()
                if op.startswith('startup'):
                    op = 'startup'
                if op != rp and first_phase_difference is None:
                    first_phase_difference = t
            curve_error = np.max(abs(np.cumsum(origin_losses, axis=0)
                                     - np.cumsum(rewrite_losses, axis=0)), axis=0)
            row = {'case': case['id'], 'seed': seed, 'settings': cfg.to_dict(),
                   'rounds': len(rewrite_rows), 'maximum_slate_entry_error': max_slate_error,
                   'first_slate_difference_over_1e-7': first_slate_difference,
                   'first_winner_difference': first_winner_difference,
                   'first_phase_difference': first_phase_difference,
                   'origin_final_regrets': origin_losses.sum(axis=0).tolist(),
                   'rewrite_final_regrets': rewrite_losses.sum(axis=0).tolist(),
                   'maximum_cumulative_regret_error': curve_error.tolist(),
                   'origin_phase_counts': dict(Counter(origin_result.phases.tolist())),
                   'rewrite_phase_counts': dict(Counter(x.phase.name for x in rewritten.interactions))}
            document['runs'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')
            print(f"{case['id']} seed {seed}: slate error {max_slate_error:.3g}, "
                  f"regret curve errors {curve_error}, winner divergence {first_winner_difference}",
                  flush=True)

    worst = max(max(row['maximum_cumulative_regret_error']) for row in document['runs'])
    all_close = all(row['rounds'] == row['settings']['T']
                    and row['first_slate_difference_over_1e-7'] is None
                    and row['first_winner_difference'] is None
                    and row['first_phase_difference'] is None
                    and max(row['maximum_cumulative_regret_error']) < 1e-4
                    for row in document['runs'])
    document['summary'] = {'runs': len(document['runs']),
                           'largest_cumulative_regret_error': worst,
                           'all_close': all_close,
                           'tolerances': {'slate_entry': 1e-7, 'cumulative_regret': 1e-4}}
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(document['summary']), flush=True)
    return 0 if all_close else 1


if __name__ == '__main__':
    raise SystemExit(main())
