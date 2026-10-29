"""Regression checks for retained step-one geometry and paper excitation."""

# BEGIN K-RACER PARITY CHANGE (step 1): protect basis/excitation conventions
# while preserving the paper's signed excitation center.
import math

import numpy as np
import pytest

from recommenders.KRACER_functions import ExcitationSlate
from recommenders.startup_functions import make_cone


def test_startup_uses_the_origin_canonical_basis_order():
    _, frame, _ = make_cone(np.ones(3), 1.2)
    expected = np.array([[math.sqrt(2 / 3), 0],
                         [-1 / math.sqrt(6), 1 / math.sqrt(2)],
                         [-1 / math.sqrt(6), -1 / math.sqrt(2)]])
    np.testing.assert_allclose(frame, expected, atol=1e-14)


def test_excitation_ring_uses_the_public_axis_as_its_first_tangent():
    axis = np.array([1., 0., 1.]) / math.sqrt(2)
    direction = np.array([0., 1., 0.])
    slate = ExcitationSlate((-.4, .4), np.diag([2., 1., 2.]),
                            axis, direction, 2, .5)
    vectors = np.array([item.emb for item in slate])
    expected = math.sqrt(1 - .25**2) * direction + .25 * np.array([axis, -axis])
    np.testing.assert_allclose(vectors, expected, atol=1e-14)


# BEGIN PAPER ALIGNMENT CHANGE: replace orientation-dependent origin totals
# with Eq. (44)'s signed-cap conditions, including backward-facing centers.
def test_excitation_keeps_a_backward_cap_at_its_signed_center():
    axis, direction = np.eye(3)[:2]
    slate = ExcitationSlate((.1, .5), np.diag([1., 4., 3.]),
                            axis, direction, 3, .5)
    center = np.array([item.emb for item in slate]).mean(axis=0)
    assert center @ axis < 0
    assert center @ direction > 0


@pytest.mark.parametrize('K', [2, 3, 5, 9])
def test_excitation_items_follow_the_paper_signed_cap_formula(K):
    axis, direction = np.eye(3)[:2]
    V = np.array([[2., .7, 0.], [.7, 3., 0.], [0., 0., 2.]])
    # On [-.5,.5] this variance profile is maximized at the right endpoint.
    normal = -math.sin(.5) * axis + math.cos(.5) * direction
    center = np.linalg.solve(V + np.eye(3), normal)
    center /= np.linalg.norm(center)
    vectors = np.array([item.emb for item in ExcitationSlate(
        (-.5, .5), V, axis, direction, K, 1.,
    )])
    radius = .25
    np.testing.assert_allclose(vectors @ center, math.sqrt(1 - radius**2), atol=1e-14)
    perturbations = (vectors - math.sqrt(1 - radius**2) * center) / radius
    np.testing.assert_allclose(perturbations @ center, 0., atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(perturbations, axis=1), 1., atol=1e-14)
    assert vectors.mean(axis=0) @ axis < 0
# END PAPER ALIGNMENT CHANGE.
# END K-RACER PARITY CHANGE (step 1).
