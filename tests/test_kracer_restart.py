import numpy as np
import pytest

from recommenders.KRACER_functions import Restart


def cone():
    c = np.array([1.0, 0.0, 0.0])
    F = np.eye(3)[:, 1:]
    W = np.array([1.0, 0.64])
    return c, F, W


def test_full_interval_keeps_the_cone():
    c, F, W = cone()
    new_c, new_F, new_W = Restart(c, F, W, 0, (-np.pi / 4, np.pi / 4))
    np.testing.assert_allclose(new_c, c)
    np.testing.assert_allclose(new_F, F)
    np.testing.assert_allclose(new_W, W)


def test_symmetric_slab_has_hand_calculated_widths():
    c, F, W = cone()
    angle = np.arctan(0.25)
    new_c, new_F, new_W = Restart(c, F, W, 0, (-angle, angle))
    np.testing.assert_allclose(new_c, c)
    np.testing.assert_allclose(new_F, F)
    np.testing.assert_allclose(new_W, [0.125, 1.2])


def test_asymmetric_slab_rotates_the_axis_without_mutating_inputs():
    c, F, W = cone()
    old_c, old_F, old_W = c.copy(), F.copy(), W.copy()
    new_c, new_F, new_W = Restart(c, F, W, 0, (np.arctan(-0.5), np.arctan(0.25)))
    np.testing.assert_allclose(new_c, [0.9966828945138263, -0.08138309273762757, 0.0])
    np.testing.assert_allclose(new_W, [0.27567946344609745, 1.0774252230734584])
    frame = np.column_stack((new_c, new_F))
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-14)
    np.testing.assert_array_equal(c, old_c)
    np.testing.assert_array_equal(F, old_F)
    np.testing.assert_array_equal(W, old_W)


@pytest.mark.parametrize("left,right", [(-1.0, 1.0), (-0.25, 0.25), (-0.5, 0.25), (0.1, 0.8)])
def test_restart_contains_sampled_retained_directions(left, right):
    c, F, W = cone()
    new_c, new_F, new_W = Restart(c, F, W, 0, (np.arctan(left), np.arctan(right)))
    for x in np.linspace(left, right, 17):
        for transverse in np.linspace(-1.0, 1.0, 13):
            y = np.array([x, 0.8 * transverse * np.sqrt(max(0.0, 1 - x * x))])
            u = c + F @ y
            u /= np.linalg.norm(u)
            assert new_c @ u > 0
            new_y = (new_F.T @ u) / (new_c @ u)
            assert np.sum(new_y**2 / new_W) <= 1 + 1e-10


def test_restart_rejects_invalid_cones_and_intervals():
    c, F, W = cone()
    with pytest.raises(ValueError):
        Restart(c, F, np.array([[1.0, 0.1], [0.1, 0.64]]), 0, (-0.1, 0.1))
    with pytest.raises(ValueError):
        Restart(c, F, W, 2, (-0.1, 0.1))
    with pytest.raises(ValueError):
        Restart(c, F, W, 0, (-1.0, 1.0))


def test_wide_cone_restart_avoids_cancellation():
    c, F, _ = cone()
    W = np.array([1e16, 1.0])
    interval = (np.arctan(0.5e8), np.arctan(0.9e8))
    new_c, new_F, new_W = Restart(c, F, W, 0, interval)
    for angle in interval:
        x = np.tan(angle) / 1e8
        y = np.array([1e8 * x, np.sqrt(1 - x * x)])
        u = c + F @ y
        u /= np.linalg.norm(u)
        new_y = new_F.T @ u / (new_c @ u)
        assert np.sum(new_y**2 / new_W) <= 1 + 1e-10
