import numpy as np

from config import SimConfig, UserConfig
from models.interaction import Interaction
from models.item import Item
from models.phase import Phase
from users.bayesian_user import BayesianUser
from users.user_state import UserState


def test_choice_uses_mean_covariance_and_signed_risk():
    state = UserState(UserConfig(), SimConfig(ambient_dimension=3, seed=42))
    state.mean = np.array([1.0, 0.8, 0.0])
    state.cov = np.diag([2.0, 0.1, 1.0])
    user = BayesianUser(state)
    slate = [Item(np.array([1.0, 0.0, 0.0])), Item(np.array([0.0, 1.0, 0.0]))]

    state.risk = 0.0
    assert user.choose(slate) is slate[0]
    state.risk = 1.0
    assert user.choose(slate) is slate[1]
    state.risk = -1.0
    assert user.choose(slate) is slate[0]


def test_choice_breaks_exact_ties_by_first_position_without_drawing_noise():
    state = UserState(UserConfig(), SimConfig(ambient_dimension=3, seed=42))
    user = BayesianUser(state)
    slate = [Item(np.array([1.0, 0.0, 0.0])), Item(np.array([0.0, 1.0, 0.0]))]
    rng_before = state.rng.bit_generator.state

    assert user.choose(slate) is slate[0]
    assert user.choose([slate[1], slate[0]]) is slate[1]
    assert state.rng.bit_generator.state == rng_before


def test_private_experience_and_posterior_are_reproducible():
    state = UserState(UserConfig(sigma_u=0.1), SimConfig(ambient_dimension=3, seed=42))
    user = BayesianUser(state)
    rng = np.random.default_rng(42)
    preference = rng.normal(size=3)
    preference /= np.linalg.norm(preference)
    np.testing.assert_array_equal(state.preference, preference)

    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    V = 0.03 * np.eye(3)
    total = np.zeros(3)
    for vector in vectors:
        item = Item(vector)
        interaction = Interaction([item], 0, item, 0.0, 0.0, Phase.EXCITE)
        user.update(interaction)
        utility = preference @ vector
        experience = utility + rng.normal(0.0, 0.1)
        assert interaction.utility == utility
        assert interaction.experience == experience
        V += np.outer(vector, vector)
        total += vector * experience
        np.testing.assert_allclose(state.V, V)
        np.testing.assert_allclose(state.mean, np.linalg.solve(V, total), atol=1e-12)
        np.testing.assert_allclose(state.cov, 0.01 * np.linalg.inv(V), atol=1e-12)


def test_equal_seeds_produce_the_same_private_experience():
    first = BayesianUser(UserState(UserConfig(), SimConfig(ambient_dimension=3, seed=6000)))
    second = BayesianUser(UserState(UserConfig(), SimConfig(ambient_dimension=3, seed=6000)))
    item = Item(np.array([1.0, 0.0, 0.0]))
    for _ in range(3):
        left = Interaction([item], 0, item, 0.0, 0.0, Phase.EXCITE)
        right = Interaction([item], 0, item, 0.0, 0.0, Phase.EXCITE)
        first.update(left)
        second.update(right)
        assert left.experience == right.experience
