import numpy as np
import pytest

from models.item import Item
from recommenders.KRACER_functions import EQ


def test_eq_places_items_at_the_angular_bin_midpoints():
    c = np.array([1.0, 0.0, 0.0])
    g = np.array([0.0, 1.0, 0.0])
    slate = EQ((-np.pi / 3, np.pi / 3), c, g, 2)

    assert len(slate) == 2
    assert all(isinstance(item, Item) for item in slate)
    np.testing.assert_allclose(slate[0].emb, [np.sqrt(3) / 2, -0.5, 0.0], atol=1e-15)
    np.testing.assert_allclose(slate[1].emb, [np.sqrt(3) / 2, 0.5, 0.0], atol=1e-15)


@pytest.mark.parametrize("K", [2, 3, 5])
def test_eq_returns_ordered_unit_items_in_the_query_plane(K):
    c = np.array([1.0, 0.0, 0.0])
    g = np.array([0.0, 1.0, 0.0])
    A, B = -0.7, 0.4
    slate = EQ((A, B), c, g, K)
    vectors = np.array([item.emb for item in slate])

    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(vectors[:, 2], 0.0)
    angles = np.arctan2(vectors @ g, vectors @ c)
    np.testing.assert_allclose(angles, A + (np.arange(K) + 0.5) * (B - A) / K)


def test_eq_rejects_invalid_inputs():
    c = np.array([1.0, 0.0, 0.0])
    g = np.array([0.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        EQ((-0.5, 0.5), c, g, 1)
    with pytest.raises(ValueError):
        EQ((0.5, -0.5), c, g, 2)
    with pytest.raises(ValueError):
        EQ((-0.5, 0.5), c, c, 2)


def test_eq_normalizes_items_from_nearly_orthonormal_inputs():
    c = np.array([1.0 + 9e-9, 0.0, 0.0])
    g = np.array([9e-9, 1.0 + 9e-9, 0.0])
    vectors = [item.emb for item in EQ((0.7, 0.9), c, g, 2)]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=0, atol=1e-15)
