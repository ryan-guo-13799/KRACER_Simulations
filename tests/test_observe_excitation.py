import numpy as np
import pytest

from models.item import Item
from recommenders.KRACER_functions import ExcitationSlate, Observe


def test_observe_updates_information_for_the_chosen_item():
    slate = [Item(np.array([1., 0., 0.])), Item(np.array([0., 1., 0.])),
             Item(np.array([0., 0., 1.]))]
    V = np.diag([2., 4., 8.])
    original = V.copy()

    chosen, updated, next_t = Observe(slate, 1, V, 7)

    assert chosen is slate[1]
    np.testing.assert_allclose(updated, np.diag([2., 5., 8.]))
    np.testing.assert_array_equal(V, original)
    assert next_t == 8


@pytest.mark.parametrize("V", [
    np.diag([2., 1., 3.]),
    np.array([[2., 0.7, 0.2], [0.7, 3., 0.1], [0.2, 0.1, 2.]]),
    np.diag([1., 3., 2.]),
])
@pytest.mark.parametrize("K", [2, 3, 7])
def test_excitation_targets_worst_normal_and_returns_unit_distinct_items(V, K):
    c, g = np.eye(3)[:2]
    A, B = -0.6, 0.4
    items = ExcitationSlate((A, B), V, c, g, K, 0.5)
    vectors = np.array([item.emb for item in items])
    radius = min(0.25, 1 / np.sqrt(2 * (1 + 1 / 0.5)))
    center = vectors.mean(axis=0) / np.sqrt(1 - radius**2)

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(vectors @ center, np.sqrt(1 - radius**2), atol=1e-12)
    assert all(np.linalg.norm(vectors[i] - vectors[j]) > 1e-12
               for i in range(K) for j in range(i))

    # Recover the chosen normal from the center and compare its uncertainty
    # with a dense, independent search over the entire angular interval.
    direction = (V + np.eye(3)) @ center
    direction /= np.linalg.norm(direction)
    grid = np.linspace(A, B, 20001)
    normals = -np.sin(grid)[:, None] * c + np.cos(grid)[:, None] * g
    uncertainties = np.einsum("ij,ij->i", normals, np.linalg.solve(V, normals.T).T)
    assert direction @ np.linalg.solve(V, direction) >= max(uncertainties) - 1e-10


def test_excitation_checks_information_lower_bound():
    with pytest.raises(ValueError, match="lambda0"):
        ExcitationSlate((-0.5, 0.5), np.eye(3), np.eye(3)[0], np.eye(3)[1], 3, 2.)


def test_excitation_uses_smaller_radius_for_low_initial_information():
    lambda0 = 0.001
    c, g = np.eye(3)[:2]
    vectors = np.array([item.emb for item in ExcitationSlate(
        (-0.5, 0.5), np.eye(3), c, g, 2, lambda0
    )])
    radius = 1 / np.sqrt(2 * (1 + 1 / lambda0))
    assert np.linalg.norm(vectors[0] - vectors[1]) == pytest.approx(2 * radius)
