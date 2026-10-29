"""Check the old independent runner and time one full-horizon T0 grid."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'src'))
from config import SimConfig, UserConfig, RecommenderConfig
from cutoff_search import evaluate_cutoffs
from simulation_function import RECOMMENDERS, run_simulation


def legacy_runner():
    source = subprocess.check_output(['git', '-C', str(PROJECT), 'show',
        't0-before-forking:src/simulation_function.py'], text=True)
    module = types.ModuleType('legacy_independent_simulation')
    sys.modules[module.__name__] = module
    exec(compile(source, 'git:t0-before-forking:simulation_function.py', 'exec'), module.__dict__)
    return module.run_simulation


def same_result(left, right):
    assert len(left.interactions) == len(right.interactions)
    for a, b in zip(left.interactions, right.interactions):
        assert (a.timestep, a.phase, a.chosen_id, a.experience, a.utility) == (
            b.timestep, b.phase, b.chosen_id, b.experience, b.utility)
        np.testing.assert_array_equal([q.emb for q in a.slate], [q.emb for q in b.slate])
    for name in ('mean', 'cov', 'V', 'preference'):
        np.testing.assert_array_equal(getattr(left.user.user_state, name),
                                      getattr(right.user.user_state, name))
    assert left.user.user_state.rng.bit_generator.state == right.user.user_state.rng.bit_generator.state


def score(result):
    preference = result.user.user_state.preference
    return sum(1. - sum(float(preference @ item.emb) for item in x.slate) / len(x.slate)
               for x in result.interactions)


def main():
    old_run = legacy_runner()
    evidence = {'environment': {'python': platform.python_version(), 'numpy': np.__version__},
                'legacy_reference': 't0-before-forking', 'legacy_compatibility': [], 'benchmarks': [],
                'source_hashes': {str(p.relative_to(PROJECT / 'src')): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((PROJECT / 'src').rglob('*.py'))}}
    for algorithm in RECOMMENDERS:
        for horizon in (2, 180):
            sim = SimConfig(ambient_dimension=5, num_steps=horizon, seed=120000,
                            user_config=UserConfig(sigma_u=.1 / np.sqrt(5)))
            rec = RecommenderConfig(slate_size=2, cutoff=min(64, horizon), gamma_ctr=.95,
                                   analytical_gate_size=128.)
            old = old_run(deepcopy(sim), rec, algorithm, cutoff_after_startup=True)
            new = run_simulation(deepcopy(sim), rec, algorithm, cutoff_after_startup=True)
            same_result(old, new)
            evidence['legacy_compatibility'].append({'algorithm': algorithm, 'horizon': horizon,
                                                      'all_interactions_and_private_state_exact': True})

    grid = [0, 1, 2, 4, 8, 16, 32, 64, 128]
    for algorithm in RECOMMENDERS:
        sim = SimConfig(ambient_dimension=5, num_steps=10000, seed=120000,
                        user_config=UserConfig(sigma_u=.1 / np.sqrt(5)))
        rec = RecommenderConfig(slate_size=2, cutoff=128,
                               gamma_ctr=.5 if algorithm == 'analytical_certificate' else .95,
                               analytical_gate_size=128.)
        started = time.perf_counter()
        independent = {cutoff: score(old_run(deepcopy(sim), replace(rec, cutoff=cutoff), algorithm,
                                             cutoff_after_startup=True)) for cutoff in grid}
        independent_seconds = time.perf_counter() - started
        started = time.perf_counter()
        forked = evaluate_cutoffs(deepcopy(sim), rec, grid, algorithm)
        forked_seconds = time.perf_counter() - started
        assert independent == forked, (algorithm, independent, forked)
        row = {'algorithm': algorithm, 'horizon': sim.num_steps, 'seed': sim.seed,
               'cutoffs': grid, 'all_scores_exact': True, 'scores': forked,
               'independent_seconds': independent_seconds, 'forked_seconds': forked_seconds,
               'speedup': independent_seconds / forked_seconds}
        evidence['benchmarks'].append(row)
        (HERE / 'results.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    print('Legacy compatibility and all forked T0 scores match exactly.', flush=True)


if __name__ == '__main__':
    main()
