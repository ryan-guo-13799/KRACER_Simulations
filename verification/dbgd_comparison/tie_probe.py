"""Locate remaining DBGD divergence after matching only the policy RNG seed."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT.parent / 'simulations')]
from simulation_main import case_configs
from simulation_state import SimulationState
from recommenders.baseline_support.dbgd import DBGD
from simulation.config import Config
from simulation.episode import EpisodeState


def manual_state(seed):
    case = dict(id='baseline', label='Baseline', dimension=5, K=2, noise=.1)
    sim, rec = case_configs(case, seed, 10000, 10000)
    state = SimulationState(sim, rec, 'dbgd')
    state.recommender.policy = DBGD(state.recommender.policy.cfg,
        seed=np.random.SeedSequence([seed, 0xDB6D, 17]))
    return state


def main():
    evidence = []
    previous = json.loads((HERE / 'results.json').read_text())
    rows = [row for row in previous['seed_only_counterfactual']
            if abs(row['manual_seed_only_slate_regret'] - row['saved_origin_slate_regret']) > 1e-6]
    for row in rows:
        seed = row['seed']
        state = manual_state(seed)
        origin = EpisodeState(Config(D=5, K=2, T=10000, T0=10000, startup_regime='sphere',
                                     sigma_u=state.sim_config.user_config.sigma_u),
                              'dbgd', seed, audit=False, fast_forward=False)
        captured = {}
        manual_choose, origin_choose = state.user.choose, origin.user.choose

        def capture_manual(slate):
            user = state.user.user_state
            scores = [float(user.mean @ q.emb - user.risk / 2 * (q.emb @ user.cov @ q.emb)) for q in slate]
            chosen = manual_choose(slate)
            captured['manual'] = (np.array([q.emb for q in slate]), scores,
                                  next(i for i, q in enumerate(slate) if q is chosen))
            return chosen

        def capture_origin(slate, V, t):
            inv_q = np.linalg.solve(V, slate.T)
            scores = (slate @ origin.user.posterior_mean(V)
                      - .5 * origin.user.gamma_cara * origin.user.sigma_u**2
                      * np.sum(slate.T * inv_q, axis=0))
            winner = origin_choose(slate, V, t)
            captured['origin'] = (slate.copy(), scores.tolist(), winner)
            return winner

        state.user.choose, origin.user.choose = capture_manual, capture_origin
        for t in range(1, 10001):
            state.step()
            origin.step()
            mq, ms, mw = captured['manual']
            oq, os, ow = captured['origin']
            assert np.array_equal(mq, oq), 'Unexpected slate difference before a winner difference'
            if mw != ow:
                difference = dict(seed=seed, round=t, slates_identical=True,
                    manual_scores=ms, origin_scores=os,
                    manual_score_difference=ms[1] - ms[0],
                    origin_score_difference=os[1] - os[0],
                    manual_winner=mw, origin_winner=ow,
                    saved_origin_slate_regret=row['saved_origin_slate_regret'])
                break
        else:
            raise AssertionError('Expected a differing winner')

        # Diagnostic intervention only: force that one winner in memory.
        replay = manual_state(seed)
        choose = replay.user.choose

        def force_one_choice(slate):
            return slate[ow] if replay.t == t else choose(slate)

        replay.user.choose = force_one_choice
        replay.finish()
        difference['manual_regret_after_matching_one_choice_only'] = replay.slate_regret
        difference['absolute_regret_error_after_one_choice_override'] = abs(
            replay.slate_regret - row['saved_origin_slate_regret'])
        evidence.append(difference)
        print(json.dumps(difference), flush=True)
    (HERE / 'tie_probe.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    corrections = {row['seed']: row['manual_regret_after_matching_one_choice_only'] for row in evidence}
    rows = previous['seed_only_counterfactual']
    combined = [corrections.get(row['seed'], row['manual_seed_only_slate_regret']) for row in rows]
    previous['baseline_mean_slate_regret'][
        'manual_separate_seed_plus_matching_first_choice_in_memory_only'] = float(np.mean(combined))
    previous['round_one_tie_probe_summary'] = {
        'affected_seeds': sorted(corrections),
        'max_final_regret_error_after_diagnostic_choice_match': max(
            abs(value - row['saved_origin_slate_regret']) for value, row in zip(combined, rows)),
        'note': 'Choice overrides are diagnostic only; no production code was changed.'}
    (HERE / 'results.json').write_text(json.dumps(previous, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
