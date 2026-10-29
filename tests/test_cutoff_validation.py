import json

import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from cutoff_search import validate_cutoffs, validate_grid
from models.phase import Phase
from simulation_function import run_simulation
import simulation_main


def test_validation_uses_mean_slate_regret_and_smaller_cutoff_on_ties():
    data = {(10, 0): 8, (11, 0): 2, (10, 1): 3, (11, 1): 3,
            (10, 2): 4, (11, 2): 2}
    best, scores = validate_cutoffs([2, 0, 1], [10, 11], lambda seed, cutoff: data[seed, cutoff])
    assert best == 1
    assert [row["mean_slate_regret"] for row in scores] == [5, 3, 3]


def test_nonfinite_validation_does_not_select_a_cutoff():
    with pytest.raises(ValueError, match="finite"):
        validate_cutoffs([0, 1], [10], lambda seed, cutoff: np.nan)


def test_grid_selects_best_pair_and_breaks_ties_by_cutoff_then_parameter():
    def score(seed, cutoff, parameter):
        return 1.0 if (cutoff, parameter) in [(2, 0.3), (0, 0.7), (0, 0.5)] else 2.0

    cutoff, parameter, scores = validate_grid([2, 0], [0.7, 0.3, 0.5], [10, 11], score)
    assert (cutoff, parameter) == (0, 0.5)
    assert len(scores) == 6
    assert all(len(row["seed_scores"]) == 2 for row in scores)


def test_cutoff_is_additional_to_startup_and_keeps_absolute_confidence_time():
    sim = SimConfig(ambient_dimension=3, num_steps=15, seed=42,
                    user_config=UserConfig(sigma_u=0.01))
    rec = RecommenderConfig(slate_size=2, cutoff=2, startup_half_angle=1.5)
    result = run_simulation(sim, rec, cutoff_after_startup=True)
    assert result.startup.state.t == 7
    assert result.recommender.cutoff == 8
    assert [x.phase for x in result.interactions[6:8]] == [Phase.LOCALIZE] * 2
    assert result.interactions[8].phase == Phase.EXPLOIT
    assert rec.cutoff == 2  # the caller's configuration is unchanged


def test_runner_selects_using_validation_seeds_only(tmp_path, monkeypatch, capsys):
    calls = []
    searches = []

    def fake_run(case, seed, horizon, cutoff, algorithm, *, gamma_ctr, gate_size):
        calls.append((seed, cutoff, gamma_ctr))
        return {"seed": seed, "settings": {"post_startup_cutoff": cutoff, "gamma_ctr": gamma_ctr},
                "summary": {"slate_average_regret": abs(cutoff - 2) + abs(gamma_ctr - 0.7) + seed / 1e6}}

    # BEGIN T0 FORKING CHANGE: validation uses one batch per seed/parameter;
    # held-out seeds still take the ordinary independent run path.
    def fake_search(case, seed, horizon, cutoffs, algorithm, *, gamma_ctr, gate_size):
        searches.append((seed, tuple(cutoffs), gamma_ctr))
        return {cutoff: abs(cutoff - 2) + abs(gamma_ctr - 0.7) + seed / 1e6
                for cutoff in cutoffs}

    monkeypatch.setattr(simulation_main, "study_cases", lambda: [{"id": "one"}])
    monkeypatch.setattr(simulation_main, "run_case", fake_run)
    monkeypatch.setattr(simulation_main, "run_cutoff_search", fake_search)
    monkeypatch.setattr("sys.argv", ["simulation_main.py", "--horizon", "4", "--cutoffs", "0", "2",
                                     "--gamma-values", "0.3", "0.7", "--validation-seed-count", "2", "--seed-count", "1",
                                     "--algorithms", "kracer", "--output", str(tmp_path / "result")])
    simulation_main.main()
    folder = tmp_path / "result" / "one" / "kracer"
    validation = json.loads((folder / "validation.json").read_text())
    test = json.loads((folder / "seed_6000.json").read_text())
    assert validation["selected_cutoff"] == 2
    assert validation["parameter_name"] == "gamma_ctr"
    assert validation["selected_parameter"] == 0.7
    assert validation["validation_seeds"] == [120000, 120001]
    assert searches == [(120000, (0, 2), 0.3), (120001, (0, 2), 0.3),
                        (120000, (0, 2), 0.7), (120001, (0, 2), 0.7)]
    assert calls == [(6000, 2, 0.7)]
    assert test["settings"]["post_startup_cutoff"] == 2
    assert test["settings"]["gamma_ctr"] == 0.7
    assert "Runs: 9 completed, 0 remaining, 0 failed, 0 skipped (total 9)" in capsys.readouterr().out
    # END T0 FORKING CHANGE.


def test_progress_counts_failed_and_skipped_runs(tmp_path, monkeypatch, capsys):
    def failed_run(*args, **kwargs):
        raise ValueError("cannot run")

    monkeypatch.setattr(simulation_main, "study_cases", lambda: [{"id": "one"}])
    monkeypatch.setattr(simulation_main, "run_case", failed_run)
    # BEGIN T0 FORKING CHANGE: fail the validation batch, not the held-out path.
    monkeypatch.setattr(simulation_main, "run_cutoff_search", failed_run)
    # END T0 FORKING CHANGE.
    monkeypatch.setattr("sys.argv", ["simulation_main.py", "--horizon", "4", "--cutoffs", "0", "2",
                                     "--gamma-values", "0.3", "0.3", "0.7", "--validation-seed-count", "2",
                                     "--seed-count", "1", "--algorithms", "kracer",
                                     "--output", str(tmp_path / "failed")])
    with pytest.raises(SystemExit):
        simulation_main.main()
    assert "Runs: 0 completed, 0 remaining, 1 failed, 8 skipped (total 9)" in capsys.readouterr().out


def test_runner_tunes_analytical_gate_with_fixed_gamma(tmp_path, monkeypatch):
    calls = []
    searches = []

    def fake_run(case, seed, horizon, cutoff, algorithm, *, gamma_ctr, gate_size):
        calls.append((seed, cutoff, gamma_ctr, gate_size))
        return {"settings": {"gate_size": gate_size, "gamma_ctr": gamma_ctr},
                "summary": {"slate_average_regret": abs(gate_size - 16)}}

    # BEGIN T0 FORKING CHANGE: keep gate tuning separate from budget sharing.
    def fake_search(case, seed, horizon, cutoffs, algorithm, *, gamma_ctr, gate_size):
        searches.append((seed, tuple(cutoffs), gamma_ctr, gate_size))
        return {cutoff: abs(gate_size - 16) for cutoff in cutoffs}

    monkeypatch.setattr(simulation_main, "study_cases", lambda: [{"id": "one"}])
    monkeypatch.setattr(simulation_main, "run_case", fake_run)
    monkeypatch.setattr(simulation_main, "run_cutoff_search", fake_search)
    monkeypatch.setattr("sys.argv", ["simulation_main.py", "--horizon", "4", "--cutoffs", "1",
                                     "--gate-sizes", "4", "16", "--validation-seed-count", "1",
                                     "--seed-count", "1", "--algorithms", "analytical_certificate",
                                     "--output", str(tmp_path / "result")])
    simulation_main.main()
    folder = tmp_path / "result" / "one" / "analytical_certificate"
    validation = json.loads((folder / "validation.json").read_text())
    test = json.loads((folder / "seed_6000.json").read_text())
    assert validation["parameter_name"] == "analytical_gate_size"
    assert validation["selected_parameter"] == 16
    assert searches == [(120000, (1,), 0.5, 4), (120000, (1,), 0.5, 16)]
    assert calls == [(6000, 1, 0.5, 16)]
    assert test["settings"] == {"gate_size": 16, "gamma_ctr": 0.5}
    # END T0 FORKING CHANGE.
