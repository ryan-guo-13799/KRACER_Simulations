import numpy as np
import pytest

from config import SimConfig, UserConfig
from users.user_state import UserState


def test_lambda0_uses_dimension_and_noise_for_initial_information():
    user = UserConfig(sigma_u=0.2, dimension=3)
    assert user.lambda0 == pytest.approx(0.12)
    config = SimConfig(ambient_dimension=5, user_config=user)
    assert config.user_config.dimension == 5
    assert config.user_config.lambda0 == pytest.approx(0.2)
    state = UserState(user, SimConfig(ambient_dimension=3))
    np.testing.assert_allclose(state.V, 0.12 * np.eye(3))
