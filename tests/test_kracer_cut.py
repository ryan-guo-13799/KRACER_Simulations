import numpy as np
import pytest

from recommenders import KRACER_functions
from recommenders.KRACER_functions import Cut, EQ, SafeAngularBounds


def test_safe_angular_bounds_computes_margin_from_public_information():
    c, g = np.eye(3)[:, :2].T
    margin = 0.1  # V = I and the comparison normal has unit length.
    bound = np.arctan(margin * np.sqrt(1.25 / (1 - margin**2 * 0.75)))
    result = SafeAngularBounds((-0.6, 0.6), np.eye(3), margin, 0.0, c, g,
                               np.array([1.0, 0.25]), 0)
    np.testing.assert_allclose(result, [-bound, bound], atol=1e-12)


@pytest.mark.parametrize("boundary", [-0.8, 0.8])
def test_degenerate_quadratic_uses_the_linear_root(boundary):
    c, g = np.eye(3)[:, :2].T
    W = np.array([100.0, 1.0])
    interval = (-np.arctan(10.0), np.arctan(10.0))
    b = 1 - W[1] / W[0]
    h = np.cos(boundary) / np.sqrt(b)
    margin = h * np.linalg.norm(-np.sin(boundary) * c + np.cos(boundary) * g)
    assert np.cos(boundary)**2 - margin**2 * b == 0
    linear = -2 * np.sin(boundary) * np.cos(boundary)
    constant = np.sin(boundary)**2 - margin**2 * (1 + W[1])
    root_angle = np.arctan(-constant / linear)

    left, right = SafeAngularBounds(interval, np.eye(3), h, boundary, c, g, W, 0)
    expected = (root_angle, interval[1]) if boundary > 0 else (interval[0], root_angle)
    np.testing.assert_allclose([left, right], expected, atol=1e-12)


@pytest.mark.parametrize("relative_change", [-1e-14, 1e-14])
def test_nearly_degenerate_quadratic_keeps_the_finite_root(relative_change):
    c, g = np.eye(3)[:, :2].T
    W = np.array([100.0, 1.0])
    interval = (-np.arctan(10.0), np.arctan(10.0))
    boundary = 0.8
    h = np.cos(boundary) / np.sqrt(1 - W[1] / W[0]) * (1 + relative_change)
    left, right = SafeAngularBounds(interval, np.eye(3), h, boundary, c, g, W, 0)
    assert left == pytest.approx(-0.43625, abs=1e-4)
    assert right == interval[1]


@pytest.mark.parametrize("winner,expected_boundaries", [(0, [-0.2]), (1, [-0.2, 0.2]), (2, [0.2])])
def test_cut_delegates_adjacent_boundaries_to_safe_angular_bounds(
    monkeypatch, winner, expected_boundaries
):
    calls = []
    original = KRACER_functions.SafeAngularBounds

    def record(I, V, h, boundary, c, g_i, W, i):
        calls.append(boundary)
        return original(I, V, h, boundary, c, g_i, W, i)

    monkeypatch.setattr(KRACER_functions, "SafeAngularBounds", record)
    c, g = np.eye(3)[:, :2].T
    Cut((-0.6, 0.6), np.eye(3), 0.1, winner, c, g, 3, np.array([1.0, 0.25]), 0)
    np.testing.assert_allclose(calls, expected_boundaries)


@pytest.mark.parametrize("K", [2, 3, 5])
def test_zero_margin_retains_the_winning_angular_bin(K):
    c, g = np.eye(3)[:, :2].T
    W = np.array([1.0, 0.25])
    interval = (-0.6, 0.6)
    expected_slate = EQ(interval, c, g, K)
    width = (interval[1] - interval[0]) / K
    for winner in range(K):
        result, chosen = Cut(interval, np.eye(3), 0.0, winner, c, g, K, W, 0)
        np.testing.assert_allclose(result, [interval[0] + winner * width,
                                           interval[0] + (winner + 1) * width])
        np.testing.assert_allclose(chosen.emb, expected_slate[winner].emb)


def test_positive_margin_expands_the_boundary_by_the_paper_formula():
    c, g = np.eye(3)[:, :2].T
    W = np.array([1.0, 0.25])
    margin = 0.1  # V = I, so the unit normal has variance 1.
    bound = np.arctan(margin * np.sqrt(1.25 / (1 - margin**2 * 0.75)))
    left, _ = Cut((-0.6, 0.6), np.eye(3), margin, 0, c, g, 2, W, 0)
    right, _ = Cut((-0.6, 0.6), np.eye(3), margin, 1, c, g, 2, W, 0)
    np.testing.assert_allclose(left, [-0.6, bound], atol=1e-12)
    np.testing.assert_allclose(right, [-bound, 0.6], atol=1e-12)


def test_large_margin_keeps_the_full_interval():
    c, g = np.eye(3)[:, :2].T
    for winner in range(3):
        result, _ = Cut((-0.6, 0.6), np.eye(3), 10.0, winner, c, g, 3,
                        np.array([1.0, 0.25]), 0)
        assert result == (-0.6, 0.6)


def test_small_positive_margin_does_not_collapse_to_the_boundary():
    c, g = np.eye(3)[:, :2].T
    interval = (-0.5, 0.7)
    result, _ = Cut(interval, np.eye(3), 1e-12, 0, c, g, 2, np.array([1.0, 0.25]), 0)
    boundary = (interval[0] + interval[1]) / 2
    assert result[1] > boundary + 1e-13


def test_cut_keeps_an_outer_feasible_angle_when_feasibility_is_disconnected():
    c, g = np.eye(3)[:, :2].T
    W = np.array([820005.9019069502, 1.0418470992315414])
    interval = (-1.5205611467051903, 1.484549059229012)
    margin = 0.9127613940980384
    boundary = interval[0] + (interval[1] - interval[0]) / 3
    angles = np.linspace(boundary, interval[1], 5001)
    x = np.tan(angles)
    profile = (x * np.cos(boundary) - np.sin(boundary)) / np.sqrt(
        1 + W[1] + (1 - W[1] / W[0]) * x**2
    )
    assert profile.max() > margin and profile[-1] < margin
    result, _ = Cut(interval, np.eye(3), margin, 0, c, g, 3, W, 0)
    assert result[1] == interval[1]


def test_negative_discriminant_keeps_the_interval_without_taking_its_square_root():
    c, g = np.eye(3)[:, :2].T
    interval = (1.0, 1.5)
    with np.errstate(invalid="raise"):
        result = SafeAngularBounds(interval, np.eye(3), 0.9, 1.4, c, g,
                                   np.array([10000.0, 1.0]), 0)
    assert result == interval


def test_cut_rejects_bad_winner_axis_and_matrix():
    c, g = np.eye(3)[:, :2].T
    W = np.array([1.0, 0.25])
    with pytest.raises(ValueError):
        Cut((-0.6, 0.6), np.eye(3), 0.1, 2, c, g, 2, W, 0)
    with pytest.raises(ValueError):
        Cut((-0.6, 0.6), np.eye(3), 0.1, 0, c, g, 2, W, 1)
    with pytest.raises(ValueError):
        Cut((-0.6, 0.6), np.zeros((3, 3)), 0.1, 0, c, g, 2, W, 0)


def test_cut_does_not_change_the_public_inputs():
    c, g = np.eye(3)[:, :2].T
    W = np.array([1.0, 0.25])
    V = np.eye(3)
    old_c, old_g, old_W, old_V = c.copy(), g.copy(), W.copy(), V.copy()
    Cut((-0.6, 0.6), V, 0.1, 1, c, g, 3, W, 0)
    for actual, original in ((c, old_c), (g, old_g), (W, old_W), (V, old_V)):
        np.testing.assert_array_equal(actual, original)
