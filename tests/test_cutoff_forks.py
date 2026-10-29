"""A copied search prefix must equal a completely independent run."""

# BEGIN T0 FORKING CHANGE: equivalence, private-state isolation and reuse tests.
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from cutoff_search import _can_share_next_step, evaluate_cutoffs
from models.phase import Phase
from recommenders.analytical_certificate import AnalyticalCertificate
from recommenders.startup import Startup
from simulation_function import RECOMMENDERS, run_simulation
from simulation_state import SimulationState


def configs(*, K=2, horizon=35, sigma=.01):
    sim = SimConfig(ambient_dimension=3, num_steps=horizon, seed=42,
                    user_config=UserConfig(sigma_u=sigma))
    rec = RecommenderConfig(slate_size=K, cutoff=horizon, gamma_ctr=.5,
                           startup_half_angle=1.2, analytical_gate_size=128.)
    return sim, rec


def independent_score(sim, rec, algorithm):
    result = run_simulation(deepcopy(sim), rec, algorithm, cutoff_after_startup=True)
    u = result.user.user_state.preference
    return sum(1.0 - sum(float(u @ item.emb) for item in x.slate) / len(x.slate)
               for x in result.interactions)


def assert_same_result(left, right):
    assert len(left.interactions) == len(right.interactions)
    for a, b in zip(left.interactions, right.interactions):
        assert (a.timestep, a.phase, a.chosen_id, a.utility, a.experience) == (
            b.timestep, b.phase, b.chosen_id, b.utility, b.experience,
        )
        np.testing.assert_array_equal([item.emb for item in a.slate],
                                      [item.emb for item in b.slate])
    for field in ("mean", "cov", "V", "preference"):
        np.testing.assert_array_equal(getattr(left.user.user_state, field),
                                      getattr(right.user.user_state, field))
    assert left.user.user_state.rng.bit_generator.state == right.user.user_state.rng.bit_generator.state
    if left.recommender is not None:
        for field in ("c", "F", "W", "V"):
            np.testing.assert_array_equal(getattr(left.recommender.state, field),
                                          getattr(right.recommender.state, field))


@pytest.mark.parametrize('algorithm', list(RECOMMENDERS))
@pytest.mark.parametrize('K', [2, 3])
def test_forked_scores_exactly_equal_independent_runs(algorithm, K):
    sim, rec = configs(K=K)
    grid = [0, 1, 2, 4, 9, 20, 35]
    scores = evaluate_cutoffs(sim, rec, grid, algorithm)
    expected = {cutoff: independent_score(sim, replace(rec, cutoff=cutoff), algorithm)
                for cutoff in grid}
    assert scores == expected
    assert rec.cutoff == 35


@pytest.mark.parametrize('algorithm', list(RECOMMENDERS))
def test_failed_startup_is_shared_once(algorithm, monkeypatch):
    sim, rec = configs(horizon=2, sigma=.2)
    calls = []
    recommend = Startup.recommend

    def record(self, k):
        calls.append(self.state.t)
        return recommend(self, k)

    monkeypatch.setattr(Startup, 'recommend', record)
    scores = evaluate_cutoffs(sim, rec, [0, 1, 2], algorithm)
    assert calls == [1, 2]
    assert len(set(scores.values())) == 1
    assert scores[0] == independent_score(sim, replace(rec, cutoff=0), algorithm)


def certified_state():
    sim, rec = configs(horizon=35, sigma=.001)
    state = SimulationState(sim, rec, cutoff_after_startup=True)
    while state.recommender is None:
        state.step()
    return state


def test_fork_happens_before_a_short_budget_changes_the_first_query():
    state = certified_state()
    agent = state.recommender
    assert agent.state.i is None
    assert not _can_share_next_step(state, 1)
    assert agent.state.i is None  # Probing did not prepare the live algorithm.
    assert _can_share_next_step(state, 20)

    short = state.fork(1)
    short.step()
    state.step()
    assert short.interactions[-1].phase == Phase.EXCITE
    assert state.interactions[-1].phase == Phase.LOCALIZE
    sim, rec = configs(horizon=35, sigma=.001)
    independent = run_simulation(sim, replace(rec, cutoff=1), cutoff_after_startup=True)
    assert_same_result(short.finish(), independent)


def test_active_epoch_can_be_copied_without_losing_private_history():
    state = certified_state()
    state.step()
    assert state.recommender._remaining_depth > 0
    assert _can_share_next_step(state, 20)
    result = state.fork(20).finish()
    sim, rec = configs(horizon=35, sigma=.001)
    expected = run_simulation(sim, replace(rec, cutoff=20), cutoff_after_startup=True)
    assert_same_result(result, expected)


def test_fork_does_not_share_arrays_history_or_randomness():
    state = certified_state()
    branch = state.fork(20)
    original_mean = state.user.user_state.mean.copy()
    original_V = state.recommender.state.V.copy()
    original_item = state.interactions[0].slate[0].emb.copy()
    original_rng = deepcopy(state.user.user_state.rng.bit_generator.state)
    branch.user.user_state.mean[:] = 123.
    branch.recommender.state.V[:] = 456.
    branch.interactions[0].slate[0].emb[:] = 789.
    branch.user.user_state.rng.normal(size=10)
    np.testing.assert_array_equal(state.user.user_state.mean, original_mean)
    np.testing.assert_array_equal(state.recommender.state.V, original_V)
    np.testing.assert_array_equal(state.interactions[0].slate[0].emb, original_item)
    assert state.user.user_state.rng.bit_generator.state == original_rng


def test_analytical_early_stop_is_preserved_by_larger_pending_budgets(monkeypatch):
    monkeypatch.setattr(AnalyticalCertificate, '_finish_round',
                        lambda self: setattr(self, 'cutoff', self.t))
    sim, rec = configs(horizon=35, sigma=.001)
    grid = [12, 20, 35]
    scores = evaluate_cutoffs(sim, rec, grid, 'analytical_certificate')
    assert scores == {cutoff: independent_score(sim, replace(rec, cutoff=cutoff),
                                               'analytical_certificate') for cutoff in grid}
    assert len(set(scores.values())) == 1


def test_grid_is_sorted_deduplicated_and_validated():
    sim, rec = configs(horizon=2)
    assert list(evaluate_cutoffs(sim, rec, [2, 0, 2, 1])) == [0, 1, 2]
    for grid in ([], [-1], [3], [1.5]):
        with pytest.raises(ValueError):
            evaluate_cutoffs(sim, rec, grid)


def test_startup_can_certify_on_the_last_horizon_round():
    sim, rec = configs(horizon=6, sigma=.001)
    result = run_simulation(sim, rec, cutoff_after_startup=True)
    assert result.startup.state.status == "CERTIFIED"
    assert result.recommender.t == 7
    scores = evaluate_cutoffs(sim, rec, [0, 1, 6])
    assert scores == {cutoff: independent_score(sim, replace(rec, cutoff=cutoff), 'kracer')
                      for cutoff in (0, 1, 6)}


def test_main_runner_selects_the_same_grid_as_independent_runs(tmp_path, monkeypatch, capsys):
    import json
    import simulation_main as main
    from cutoff_validation import validate_grid

    case = {'id': 'small', 'label': 'small', 'dimension': 3, 'K': 2, 'noise': .02}
    cutoffs = [0, 1, 4, 9]
    seeds = [120000, 120001]
    monkeypatch.setattr(main, 'study_cases', lambda: [case])
    monkeypatch.setattr('sys.argv', ['simulation_main.py', '--horizon', '35',
        '--cutoffs', '0', '1', '4', '9', '--gamma-values', '.5', '.9',
        '--gate-sizes', '4', '16', '--validation-seed-count', '2', '--seed-count', '1',
        '--output', str(tmp_path / 'run')])
    main.main()

    for algorithm in RECOMMENDERS:
        def parameters(value):
            return ({'gamma_ctr': .5, 'gate_size': value} if algorithm == 'analytical_certificate'
                    else {'gamma_ctr': value, 'gate_size': 4.})

        def independent(seed, cutoff, value):
            return main.run_case(case, seed, 35, cutoff, algorithm,
                                 **parameters(value))['summary']['slate_average_regret']

        values = [4., 16.] if algorithm == 'analytical_certificate' else [.5, .9]
        cutoff, value, scores = validate_grid(cutoffs, values, seeds, independent)
        directory = tmp_path / 'run' / 'small' / algorithm
        validation = json.loads((directory / 'validation.json').read_text())
        assert validation['scores'] == scores
        assert validation['selected_cutoff'] == cutoff
        assert validation['selected_parameter'] == value
        assert json.loads((directory / 'seed_6000.json').read_text()) == main.run_case(
            case, 6000, 35, cutoff, algorithm, **parameters(value),
        )
    assert '68 completed,  0 remaining,  0 failed,  0 skipped (total 68)' in capsys.readouterr().out
# END T0 FORKING CHANGE.
