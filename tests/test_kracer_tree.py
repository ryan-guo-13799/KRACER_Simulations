from itertools import product

import numpy as np
import pytest

from config import SimConfig, UserConfig
from recommenders import KRACER_functions
from recommenders.KRACER_functions import Cut, Restart, TreeCertificate


def cone():
    c = np.array([1.0, 0.0, 0.0])
    F = np.eye(3)[:, 1:]
    W = np.array([1.0, 0.64])
    return c, F, W


def test_perfect_user_one_step_matches_hand_calculation():
    c, F, W = cone()
    I = (-np.pi / 4, np.pi / 4)
    # For either winner, the restarted active squared width is (3 - sqrt(5))/2.
    value = np.sqrt(3 - np.sqrt(5))
    assert TreeCertificate(I, np.eye(3), 1, 1, c, F, W, 0, 2, lambda t: 0.0, value + 1e-10) == 1
    assert TreeCertificate(I, np.eye(3), 1, 1, c, F, W, 0, 2, lambda t: 0.0, value - 1e-10) is None


def test_tree_matches_independent_winner_paths_without_changing_inputs():
    c, F, W = cone()
    V = 2.0 * np.eye(3)
    I = (-np.pi / 4, np.pi / 4)
    originals = [x.copy() for x in (c, F, W, V)]
    schedule = lambda t: 0.02 * t
    scores = []
    for depth in range(1, 4):
        values = []
        for path in product(range(3), repeat=depth):
            branch_I, branch_V = I, V.copy()
            for step, winner in enumerate(path):
                branch_I, item = Cut(branch_I, branch_V, schedule(4 + step), winner,
                                     c, F[:, 0], 3, W, 0)
                branch_V = branch_V + np.outer(item.emb, item.emb)
            _, _, new_W = Restart(c, F, W, 0, branch_I)
            values.append(np.sqrt(2) * np.sqrt(new_W[0] / W[0]))
        scores.append(max(values))

    for threshold in (0.0, 0.5, 0.7, 0.9):
        expected = next((d for d, score in enumerate(scores, 1) if score <= threshold), None)
        assert TreeCertificate(I, V, 4, 3, c, F, W, 0, 3, schedule, threshold) == expected
    for actual, original in zip((c, F, W, V), originals):
        np.testing.assert_array_equal(actual, original)


def test_tree_uses_absolute_rounds_and_accepts_the_config_schedule():
    c, F, W = cone()
    config = SimConfig(ambient_dimension=3, user_config=UserConfig(sigma_u=0.1))
    calls = []

    def schedule(t):
        calls.append(t)
        return config.h_t(t)

    result = TreeCertificate((-np.pi / 4, np.pi / 4), np.eye(3), 7, 2,
                             c, F, W, 0, 2, schedule, 0.0)
    assert calls == [7, 8]
    assert result is None


def test_tree_builds_each_branch_once_and_stops_on_first_passing_depth(monkeypatch):
    c, F, W = cone()
    original = KRACER_functions.Cut
    calls = []

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(KRACER_functions, "Cut", record)
    interval = (-np.pi / 4, np.pi / 4)
    result = TreeCertificate(interval, np.eye(3), 1, 3, c, F, W, 0, 2, lambda t: 0.0, 0.0)
    assert result is None
    assert len(calls) == 2 + 4 + 8
    calls.clear()
    assert TreeCertificate(interval, np.eye(3), 1, 3, c, F, W, 0, 2, lambda t: 0.0, 0.9) == 1
    assert len(calls) == 2


def test_tree_rejects_zero_depth():
    c, F, W = cone()
    with pytest.raises(ValueError):
        TreeCertificate((-np.pi / 4, np.pi / 4), np.eye(3), 1, 0,
                        c, F, W, 0, 2, lambda t: 0.0, 0.5)
