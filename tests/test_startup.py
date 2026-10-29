import math

import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from models.interaction import Interaction
from models.phase import Phase
from recommenders.startup import Startup
from recommenders.startup_functions import make_cone, sign_threshold
from simulation_state import run_simulation


def response(slate, winner, t):
    return Interaction(slate, winner, slate[winner], 0.0, 0.0, Phase.STARTUP, t)


def test_cone_matches_paper_geometry():
    D = 3
    angle = 1.2
    signs = np.array([1, -1, 1])
    c, F, W = make_cone(signs, angle)
    np.testing.assert_allclose(c, signs / np.sqrt(D))
    np.testing.assert_allclose(F.T @ F, np.eye(D - 1), atol=1e-14)
    np.testing.assert_allclose(F.T @ c, 0, atol=1e-14)
    np.testing.assert_allclose(W, np.tan(angle)**2)
    assert sign_threshold(D, angle) > 0
    with pytest.raises(ValueError):
        make_cone(np.array([1, 0, 1]), angle)


def test_uses_pre_response_margin_and_only_certifies_a_fresh_query():
    sim = SimConfig(ambient_dimension=3, num_steps=10)
    rec = RecommenderConfig(slate_size=2, cutoff=4, startup_half_angle=1.2)
    startup = Startup(sim, rec)
    threshold = startup.threshold
    # First query is just above threshold; second is below because V increased.
    h = 1.1 * threshold * np.sqrt(sim.user_config.lambda0)
    sim.h_t = lambda t: h
    slate = startup.recommend(2)
    assert startup._coordinate == 0
    startup.update(response(slate, 1, 1))
    assert startup.state.signs[0] == 0
    assert startup.state.counts[0] == 1
    constraint = startup.state.constraints[0]
    assert (constraint.t, constraint.coordinate, constraint.sign) == (1, 0, -1)
    assert constraint.margin == pytest.approx(1.1 * threshold)
    # Other coordinates are less informed, so they are selected before 0 again.
    for coordinate in (1, 2):
        slate = startup.recommend(2)
        assert startup._coordinate == coordinate
        startup.update(response(slate, 0, startup.state.t))
    slate = startup.recommend(2)
    assert startup._coordinate == 0
    startup.update(response(slate, 1, 4))
    assert startup.state.signs[0] == -1
    assert startup.state.margins[0] == pytest.approx(h / np.sqrt(sim.user_config.lambda0 + 1))
    np.testing.assert_allclose(startup.state.V, np.diag(sim.user_config.lambda0 + startup.state.counts))


def test_horizon_ending_does_not_return_a_cone():
    sim = SimConfig(ambient_dimension=3, num_steps=1)
    startup = Startup(sim, RecommenderConfig(slate_size=2, cutoff=1))
    slate = startup.recommend(2)
    startup.update(response(slate, 0, 1))
    assert startup.exhausted
    assert startup.state.status == "PENDING"
    with pytest.raises(RuntimeError, match="not certified"):
        startup.cone()
    with pytest.raises(RuntimeError, match="horizon"):
        startup.recommend(2)


def test_simulation_hands_off_time_and_user_information(monkeypatch):
    # Zero margins exercise the successful handoff without a long preparation.
    monkeypatch.setattr(SimConfig, "h_t", lambda self, t: 0.0)
    sim = SimConfig(ambient_dimension=3, num_steps=5, seed=42)
    rec = RecommenderConfig(slate_size=2, cutoff=3)
    result = run_simulation(sim, rec)
    assert result.startup.state.status == "CERTIFIED"
    assert result.startup.state.t == 4
    assert result.recommender.t == 6
    assert [x.timestep for x in result.interactions] == [1, 2, 3, 4, 5]
    assert [x.phase for x in result.interactions] == [Phase.STARTUP] * 3 + [Phase.EXPLOIT] * 2
    np.testing.assert_allclose(result.user.user_state.V, result.recommender.state.V)
    np.testing.assert_allclose(result.user.user_state.cov,
                               sim.user_config.sigma_u**2 * np.linalg.inv(result.recommender.state.V))
    repeat = run_simulation(SimConfig(ambient_dimension=3, num_steps=5, seed=42), rec)
    np.testing.assert_allclose([x.experience for x in result.interactions],
                               [x.experience for x in repeat.interactions])


def test_pending_simulation_returns_partial_run():
    sim = SimConfig(ambient_dimension=3, num_steps=1)
    rec = RecommenderConfig(slate_size=2, cutoff=1)
    result = run_simulation(sim, rec)
    assert result.startup.state.status == "PENDING"
    assert result.recommender is None
    assert len(result.interactions) == 1
    np.testing.assert_allclose(result.user.user_state.V, result.startup.state.V)


def test_full_run_with_actual_confidence_schedule():
    sim = SimConfig(ambient_dimension=3, num_steps=15, seed=42,
                    user_config=UserConfig(sigma_u=0.01))
    rec = RecommenderConfig(slate_size=2, cutoff=8, startup_half_angle=1.5)
    result = run_simulation(sim, rec)
    assert result.startup.state.status == "CERTIFIED"
    assert result.startup.state.t == 7
    assert len(result.interactions) == 15
    assert result.interactions[6].phase == Phase.LOCALIZE
    assert result.interactions[8].phase == Phase.EXPLOIT
    np.testing.assert_allclose(result.user.user_state.V, result.recommender.state.V)
    np.testing.assert_allclose(result.user.user_state.cov,
                               sim.user_config.sigma_u**2 * np.linalg.inv(result.recommender.state.V))


def test_startup_uses_binary_slate_with_larger_maximum(monkeypatch):
    monkeypatch.setattr(SimConfig, "h_t", lambda self, t: 0.0)
    sim = SimConfig(ambient_dimension=3, num_steps=4, seed=42)
    rec = RecommenderConfig(slate_size=5, cutoff=0)
    result = run_simulation(sim, rec)
    assert result.startup.state.status == "CERTIFIED"
    assert [len(x.slate) for x in result.interactions] == [2, 2, 2, 5]
    assert result.interactions[-1].phase == Phase.EXPLOIT
