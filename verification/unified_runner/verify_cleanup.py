"""Compare unified records with the two pre-cleanup runners in Git."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'src'))
from cutoff_search import evaluate_cutoffs
from simulation_main import case_configs, run_case
from simulation_state import BASELINES, CONE_RECOMMENDERS


def old_module(filename, name):
    source = subprocess.check_output(['git', '-C', str(PROJECT), 'show',
        f'unified-before-cleanup:src/{filename}'], text=True)
    # Only retarget imports of modules intentionally removed by the cleanup.
    # The old interaction, scoring, and serialization functions stay intact.
    source = source.replace('from simulation_function import ', 'from legacy_simulation_function import ')
    source = source.replace('from simulation_state import ', 'from legacy_simulation_state import ')
    source = source.replace('from cutoff_validation import ', 'from cutoff_search import ')
    module = types.ModuleType(name)
    module.__file__ = str(PROJECT / 'src' / filename)
    sys.modules[name] = module
    exec(compile(source, f'git:unified-before-cleanup:{filename}', 'exec'), module.__dict__)
    return module


def main():
    old_module('simulation_state.py', 'legacy_simulation_state')
    old_module('simulation_function.py', 'legacy_simulation_function')
    old_baselines = old_module('run_pairwise_baselines.py', 'legacy_pairwise_runner')
    old_cones = old_module('simulation_main.py', 'legacy_cone_runner')
    evidence = {'reference': 'unified-before-cleanup', 'baseline_records': [],
                'cone_records': [], 'raes_validation': [],
                'source_hashes': {str(p.relative_to(PROJECT / 'src')): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((PROJECT / 'src').rglob('*.py'))}}
    for D, seed, horizon in [(3, 6000, 40), (5, 120000, 80)]:
        case = {'id': f'dimension_{D}', 'label': f'D={D}', 'dimension': D, 'K': 2, 'noise': .1}
        for algorithm in BASELINES:
            cutoff = 16 if algorithm == 'raes' else None
            old = old_baselines.run_case(case, seed, horizon, algorithm, cutoff)
            new = run_case(case, seed, horizon, cutoff, algorithm)
            old_angle = old['settings']['startup_half_angle_radians']
            new_angle = new['settings']['startup_half_angle_radians']
            np.testing.assert_allclose(old_angle, new_angle, rtol=0, atol=5e-16)
            # Baselines do not use this angle. The unified metadata uses the
            # cone case's rounding so mixed-method plots see the same case.
            old['settings']['startup_half_angle_radians'] = new_angle
            assert old == new, (D, seed, algorithm)
            evidence['baseline_records'].append({'dimension': D, 'seed': seed,
                'algorithm': algorithm, 'all_trajectory_and_regret_fields_exact': True,
                'old_reference_angle': old_angle, 'new_reference_angle': new_angle})
        for algorithm in CONE_RECOMMENDERS:
            gamma = .5 if algorithm == 'analytical_certificate' else .95
            options = {'gamma_ctr': gamma, 'gate_size': 128.}
            assert old_cones.run_case(case, seed, horizon, 16, algorithm, **options) == run_case(
                case, seed, horizon, 16, algorithm, **options)
            evidence['cone_records'].append({'dimension': D, 'seed': seed,
                'algorithm': algorithm, 'complete_record_exact': True})
        grid = [0, 1, 2, 4, 16, horizon]
        sim, rec = case_configs(case, seed, horizon, horizon)
        forked = evaluate_cutoffs(deepcopy(sim), rec, grid, 'raes')
        independent = {cutoff: old_baselines.run_case(case, seed, horizon, 'raes', cutoff,
                                                     save_rounds=False) for cutoff in grid}
        assert forked == independent, (forked, independent)
        evidence['raes_validation'].append({'dimension': D, 'seed': seed, 'cutoffs': grid,
                                            'all_scores_exact': True, 'scores': forked})
    (HERE / 'results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('8 baseline records and 8 cone records verified; 12 RAES cutoff scores match exactly.')


if __name__ == '__main__':
    main()
