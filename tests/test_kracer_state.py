import numpy as np

from config import RecommenderConfig, SimConfig
from recommenders.KRACER_functions import EQ
from recommenders.KRACER_state import KRACERState


def test_new_state_starts_without_a_cone():
    state = KRACERState(RecommenderConfig(slate_size=2), SimConfig(ambient_dimension=3))
    assert state.c is None
    assert state.F is None
    assert state.W is None
    assert state.I is None


def test_state_can_hold_a_cone():
    c = np.array([1.0, 0.0, 0.0])
    F = np.eye(3)[:, 1:]
    W = np.ones(2)
    recommender_config = RecommenderConfig()
    state = KRACERState(recommender_config, SimConfig(ambient_dimension=3))
    state.c = c
    state.F = F
    state.W = W
    state.I = (-0.5, 0.5)
    state.i = 0
    np.testing.assert_array_equal(state.c, c)
    np.testing.assert_array_equal(state.F, F)
    np.testing.assert_array_equal(state.W, W)
    assert state.I == (-0.5, 0.5)
    slate = EQ(state.I, state.c, state.F[:, state.i], recommender_config.slate_size)
    assert len(slate) == recommender_config.slate_size
