import pytest

from config import SimConfig, UserConfig
from users.user_state import UserState


def test_bayesian_confidence_schedule_matches_hand_calculation():
    user = UserConfig(sigma_u=0.1, risk=2.0)
    config = SimConfig(ambient_dimension=4, user_config=user, delta_conf=0.05)
    assert config.h_t(1) == pytest.approx(0.5447746830680817)
    assert config.h_t(100) > config.h_t(1)

    # The decision uncertainty uses the magnitude of gamma.
    user.risk = -2.0
    assert config.h_t(1) == pytest.approx(0.5447746830680817)


def test_confidence_schedule_checks_its_inputs():
    config = SimConfig()
    for t in (0, -1, 1.5):
        with pytest.raises(ValueError):
            config.h_t(t)
    config.delta_conf = 1.0
    with pytest.raises(ValueError):
        config.h_t(1)


def test_confidence_and_user_state_can_share_settings():
    user = UserConfig(sigma_u=0.2, risk=1.5)
    config = SimConfig(ambient_dimension=3, user_config=user)
    state = UserState(config.user_config, config)
    assert state.sigma_u == config.user_config.sigma_u
    assert state.risk == config.user_config.risk
    assert config.h_t(1) > 0
