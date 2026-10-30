"""Read-only DBGD comparison; change the policy seed only in memory.

Run from manual_simulations with python -B verification/dbgd_comparison/compare.py.
All generated evidence stays next to this standalone verification script.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ORIGIN = PROJECT.parent / 'simulations'
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT / 'src'), str(ORIGIN)]
from simulation_main import case_configs
from simulation_state import SimulationState
from recommenders.baseline_support.dbgd import DBGD
from simulation.config import Config
from simulation.episode import EpisodeState


def fingerprints():
    return {str(p.relative_to(PROJECT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in (PROJECT / 'src', ORIGIN / 'simulation')
            for p in sorted(directory.rglob('*.py'))}


def main():
    before = fingerprints()
    policy = PROJECT / 'src/recommenders/baseline_support/dbgd.py'
    original_policy = ORIGIN / 'simulation/agents/dbgd.py'
    assert policy.read_text().replace('from .dueling import', 'from ..baselines.base import') == original_policy.read_text()
    protocol = PROJECT / 'src/recommenders/baseline_support/dueling.py'
    original_protocol = ORIGIN / 'simulation/baselines/base.py'
    assert protocol.read_text().replace('from .agent import', 'from ..agents.base import') == original_protocol.read_text()

    saved_root = PROJECT / 'output/parallel_TiOHP9zY/results'
    study_path = ORIGIN / 'output/sphere_slow_baseline_20/study.json'
    study = json.loads(study_path.read_text())
    raw = study['cases']['sphere_baseline']['raw_selected']['dbgd']
    origin_saved = dict(zip(raw['seeds'], raw['finals']['cumulative_slate_average_regret']))
    document = {'policy_and_protocol_identical_except_imports': True,
                'manual_saved_run': str(saved_root), 'origin_saved_run': str(study_path),
                'saved_cases': {}, 'seed_only_counterfactual': [], 'source_hashes': before}
    for case_name in ('baseline', 'dimension_3', 'dimension_9', 'speed_0.05', 'speed_0.15'):
        records = [json.loads(p.read_text()) for p in sorted((saved_root / case_name / 'dbgd').glob('seed_*.json'))]
        assert len(records) == 20
        assert all(np.array_equal(r['preference'], r['rounds'][0]['slate'][0]) for r in records)
        document['saved_cases'][case_name] = {
            'seeds': len(records), 'initial_incumbent_equals_true_preference_in_every_run': True,
            'mean_slate_regret': float(np.mean([r['summary']['slate_average_regret'] for r in records])),
            'mean_consumed_regret': float(np.mean([r['summary']['consumed_regret'] for r in records]))}

    case = dict(id='baseline', label='Baseline', dimension=5, K=2, noise=.1)
    for seed in range(6000, 6020):
        start = time.perf_counter()
        sim, rec = case_configs(case, seed, 10000, 10000)
        state = SimulationState(sim, rec, 'dbgd')
        preference = state.user.user_state.preference
        assert np.array_equal(state.recommender.policy.w, preference)
        cfg = Config(D=5, K=2, T=10000, T0=10000, sigma_u=sim.user_config.sigma_u,
                     gamma_cara=sim.user_config.risk, startup_regime='sphere')
        origin = EpisodeState(cfg, 'dbgd', seed, audit=False, fast_forward=False)
        assert np.array_equal(origin.u, preference)
        rng = deepcopy(state.user.user_state.rng)
        assert np.array_equal(rng.normal(0, sim.user_config.sigma_u, sim.num_steps), origin.noise)

        # The only counterfactual change: replace this in-memory policy instance.
        policy_seed = np.random.SeedSequence([seed, 0xDB6D, 17])
        state.recommender.policy = DBGD(state.recommender.policy.cfg, seed=policy_seed)
        assert np.array_equal(state.recommender.policy.w, origin.agent.w)
        assert state.recommender.policy.rng.bit_generator.state == origin.agent.rng.bit_generator.state
        initial_alignment = float(preference @ state.recommender.policy.w)
        result = state.finish()
        consumed = sum(1.0 - float(x.utility) for x in result.interactions)
        row = {'seed': seed, 'initial_alignment_with_separate_policy_seed': initial_alignment,
               'manual_seed_only_slate_regret': state.slate_regret,
               'manual_seed_only_consumed_regret': consumed,
               'manual_seed_only_winner_updates': state.recommender.policy.winner_updates,
               'saved_origin_slate_regret': origin_saved[seed]}

        # Replay three full origin episodes to check remaining implementation differences.
        if seed < 6003:
            compare = {'max_slate_error': 0.0, 'different_winners': 0, 'first_different_winner': None}
            choose = origin.user.choose

            def record_choice(slate, V, t):
                winner = choose(slate, V, t)
                manual = result.interactions[t - 1]
                error = float(np.max(np.abs(slate - np.array([q.emb for q in manual.slate]))))
                compare['max_slate_error'] = max(compare['max_slate_error'], error)
                if winner != manual.chosen_id:
                    compare['different_winners'] += 1
                    if compare['first_different_winner'] is None:
                        compare['first_different_winner'] = t
                return winner

            origin.user.choose = record_choice
            origin.finish()
            row['live_origin_comparison'] = {
                **compare, 'slate_regret': float(origin.losses[1].sum()),
                'consumed_regret': float(origin.losses[0].sum()),
                'winner_updates': origin.agent.winner_updates}
        document['seed_only_counterfactual'].append(row)
        print(f"Seed {seed}: separate-seed manual slate regret={state.slate_regret:.3f}; "
              f"saved origin={origin_saved[seed]:.3f}; {time.perf_counter()-start:.1f}s", flush=True)
        del result, state, origin

    rows = document['seed_only_counterfactual']
    document['baseline_mean_slate_regret'] = {
        'manual_as_saved': document['saved_cases']['baseline']['mean_slate_regret'],
        'manual_policy_seed_changed_only_in_memory': float(np.mean([r['manual_seed_only_slate_regret'] for r in rows])),
        'origin_as_saved': float(np.mean(list(origin_saved.values())))}
    document['source_files_unchanged'] = fingerprints() == before
    assert document['source_files_unchanged']
    (HERE / 'results.json').write_text(json.dumps(document, indent=2), encoding='utf-8')
    print(json.dumps(document['baseline_mean_slate_regret'], indent=2), flush=True)


if __name__ == '__main__':
    main()
