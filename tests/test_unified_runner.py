"""The single runner must preserve baseline lifecycles and saved data."""

# BEGIN RUNNER CLEANUP: coverage for all eight methods in one workflow.
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from cutoff_search import evaluate_cutoffs
from models.phase import Phase
from simulation_state import BASELINES, CONE_RECOMMENDERS, ONLINE_BASELINES, RECOMMENDERS, SimulationState, run_simulation


def configs(cutoff=4):
    return (SimConfig(ambient_dimension=3, num_steps=18, seed=6000,
                      user_config=UserConfig(sigma_u=.02)),
            RecommenderConfig(slate_size=2, cutoff=cutoff))


# BEGIN DBGD SEED FIX: match origin's policy stream without sharing the user's draw.
@pytest.mark.parametrize('seed', range(6000, 6020))
@pytest.mark.parametrize('dimension', [3, 5, 9])
def test_dbgd_uses_independent_origin_seed(seed, dimension):
    sim, rec = configs()
    state = SimulationState(replace(sim, seed=seed, ambient_dimension=dimension), rec, 'dbgd')
    policy = state.recommender.policy
    expected_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xDB6D, 17]))
    expected_initial = expected_rng.normal(size=dimension)
    expected_initial /= np.linalg.norm(expected_initial)
    np.testing.assert_array_equal(policy.w, expected_initial)
    assert not np.array_equal(policy.w, state.user.user_state.preference)
    np.testing.assert_array_equal(policy.rng.normal(size=dimension), expected_rng.normal(size=dimension))
# END DBGD SEED FIX.


@pytest.mark.parametrize('algorithm', BASELINES)
def test_baselines_do_not_run_cone_startup(algorithm):
    sim, rec = configs()
    result = run_simulation(sim, rec, algorithm, cutoff_after_startup=True)
    assert result.startup is None
    assert len(result.interactions) == sim.num_steps
    assert all(x.phase != Phase.STARTUP for x in result.interactions)
    assert result.recommender.t == sim.num_steps + 1
    np.testing.assert_array_equal(result.user.user_state.V, result.recommender.V)
    if algorithm == 'raes':
        assert result.recommender.policy.cfg.T0 == rec.cutoff
        assert all(x.phase == Phase.EXPLOIT for x in result.interactions[rec.cutoff:])
    else:
        assert all(x.phase != Phase.EXPLOIT for x in result.interactions)


@pytest.mark.parametrize('algorithm', BASELINES)
def test_pairwise_baselines_reject_larger_slates(algorithm):
    sim, rec = configs()
    with pytest.raises(ValueError, match='slate size of 2'):
        run_simulation(sim, replace(rec, slate_size=3), algorithm)


def test_raes_forks_match_absolute_cutoffs_and_legacy_summation():
    sim, rec = configs(cutoff=18)
    grid = [0, 1, 2, 4, 9, 18]
    scores = evaluate_cutoffs(sim, rec, grid, 'raes')
    for cutoff in grid:
        result = run_simulation(deepcopy(sim), replace(rec, cutoff=cutoff), 'raes')
        preference = result.user.user_state.preference
        expected = 0.0
        for interaction in result.interactions:
            expected += 1. - sum(float(preference @ q.emb) for q in interaction.slate) / 2
        assert scores[cutoff] == expected


def test_raes_fork_copies_private_state_and_policy_cutoff():
    sim, rec = configs(cutoff=18)
    state = SimulationState(sim, rec, 'raes')
    state.step()
    state.step()
    branch = state.fork(4)
    assert state.recommender.policy.cfg.T0 == 18
    assert branch.recommender.policy.cfg.T0 == 4
    result = branch.finish()
    expected = run_simulation(deepcopy(sim), replace(rec, cutoff=4), 'raes')
    for actual, reference in zip(result.interactions, expected.interactions):
        assert (actual.chosen_id, actual.experience, actual.phase) == (
            reference.chosen_id, reference.experience, reference.phase)
        np.testing.assert_array_equal([q.emb for q in actual.slate], [q.emb for q in reference.slate])
    np.testing.assert_array_equal(result.user.user_state.mean, expected.user.user_state.mean)
    assert result.user.user_state.rng.bit_generator.state == expected.user.user_state.rng.bit_generator.state


def test_default_cli_runs_all_eight_and_skips_ineligible_pairs(tmp_path, monkeypatch, capsys):
    import simulation_main as main
    from plotting.data import load_run

    assert len(RECOMMENDERS) == 8
    cases = [dict(id='baseline', label='Baseline', dimension=5, K=2, noise=.1),
             dict(id='K_3', label='K=3', dimension=5, K=3, noise=.1)]
    monkeypatch.setattr(main, 'study_cases', lambda: cases)
    output = tmp_path / 'all_methods'
    monkeypatch.setattr('sys.argv', ['simulation_main.py', '--horizon', '12',
        '--cutoffs', '0', '3', '--gamma-values', '.9', '--gate-sizes', '4',
        '--validation-seed-count', '1', '--seed-count', '1', '--output', str(output)])
    main.main()
    assert {p.name for p in (output / 'baseline').iterdir()} == set(RECOMMENDERS)
    assert {p.name for p in (output / 'K_3').iterdir()} == set(CONE_RECOMMENDERS)
    for algorithm in RECOMMENDERS:
        directory = output / 'baseline' / algorithm
        payload = json.loads((directory / 'seed_6000.json').read_text())
        assert len(payload['rounds']) == 12
        assert (directory / 'validation.json').exists() == (algorithm not in ONLINE_BASELINES)
        if algorithm in BASELINES:
            assert payload['startup'] == {'status': 'NOT_APPLICABLE', 'rounds': 0}
            assert payload['settings']['startup_applies'] is False
        if algorithm in ONLINE_BASELINES:
            assert payload['settings']['post_startup_cutoff'] is None
            assert set(row['phase'] for row in payload['rounds']) == {algorithm.upper()}
        if algorithm == 'raes':
            assert payload['settings']['cutoff_reference'] == 'absolute'
            validation = json.loads((directory / 'validation.json').read_text())
            assert validation['selected_parameter'] == 'fixed'
            assert validation['parameter_name'] == 'policy_variant'
    study = load_run(output)
    assert set(study['cases']['baseline']['keys']) == set(RECOMMENDERS)
    assert not study['issues']
    assert '30 completed,  0 remaining,  0 failed,  0 skipped (total 30)' in capsys.readouterr().out
# END RUNNER CLEANUP.
