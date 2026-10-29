import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from models.interaction import Interaction
from models.phase import Phase
from recommenders import KRACER as kracer_module
from recommenders.KRACER_state import KRACERState


def make_agent(*, exploration_steps=5):
    config = SimConfig(ambient_dimension=3, num_steps=6,
                       user_config=UserConfig(sigma_u=0.1))
    recommender_config = RecommenderConfig(slate_size=2, cutoff=exploration_steps, gamma_ctr=0.5)
    state = KRACERState(recommender_config, config)
    state.c = np.array([1.0, 0.0, 0.0])
    state.F = np.eye(3)[:, 1:]
    state.W = np.array([0.5, 0.3])
    state.V = 0.5 * np.eye(3)
    return kracer_module.KRACER(state, config, recommender_config, a_geo=0.001), state, config


def interaction(slate, winner, phase, timestep):
    return Interaction(slate, winner, slate[winner], 123.0, -456.0, phase, timestep)


def test_two_step_localization_uses_pre_response_information_and_restarts(monkeypatch):
    agent, state, config = make_agent()
    starting_V = state.V.copy()
    cuts = []
    restarts = []

    # The first depth fails; the second succeeds, requiring two real responses.
    def tree(I0, V, t, max_depth, c, F, W, i, K, h_t, gamma_ctr):
        return 2 if max_depth >= 2 else None

    def cut(I, V, h, winner, c, g, K, W, i):
        cuts.append((I, V.copy(), h, winner))
        chosen = kracer_module.EQ(I, c, g, K)[winner]
        return (I[0] + 0.01, I[1] - 0.01), chosen

    def restart(c, F, W, i, I):
        restarts.append(I)
        return c.copy(), F.copy(), W.copy()

    monkeypatch.setattr(kracer_module, 'TreeCertificate', tree)
    monkeypatch.setattr(kracer_module, 'Cut', cut)
    monkeypatch.setattr(kracer_module, 'Restart', restart)

    slate1 = agent.recommend(2)
    assert agent.phase == Phase.LOCALIZE
    agent.update(interaction(slate1, 1, agent.phase, 1))
    expected_V = starting_V + np.outer(slate1[1].emb, slate1[1].emb)
    np.testing.assert_allclose(cuts[0][1], starting_V)
    assert cuts[0][2] == pytest.approx(config.h_t(1))
    np.testing.assert_allclose(state.V, expected_V)
    assert restarts == []

    slate2 = agent.recommend(2)
    assert agent.phase == Phase.LOCALIZE
    agent.update(interaction(slate2, 0, agent.phase, 2))
    np.testing.assert_allclose(cuts[1][1], expected_V)
    assert cuts[1][2] == pytest.approx(config.h_t(2))
    expected_V += np.outer(slate2[0].emb, slate2[0].emb)
    np.testing.assert_allclose(state.V, expected_V)
    assert len(restarts) == 1
    assert agent.t == 3


def test_failed_tree_explores_then_exploits_after_cutoff(monkeypatch):
    agent, state, _ = make_agent(exploration_steps=1)
    initial_V = state.V.copy()
    initial_c, initial_F, initial_W = state.c.copy(), state.F.copy(), state.W.copy()
    monkeypatch.setattr(kracer_module, 'TreeCertificate', lambda *args: None)

    slate = agent.recommend(2)
    assert agent.phase == Phase.EXCITE
    agent.update(interaction(slate, 0, agent.phase, 1))
    np.testing.assert_allclose(state.V, initial_V + np.outer(slate[0].emb, slate[0].emb))
    np.testing.assert_array_equal(state.c, initial_c)
    np.testing.assert_array_equal(state.F, initial_F)
    np.testing.assert_array_equal(state.W, initial_W)

    slate = agent.recommend(2)
    assert agent.phase == Phase.EXPLOIT
    assert len(slate) == 2
    for item in slate:
        np.testing.assert_allclose(item.emb, initial_c)
    previous_V = state.V.copy()
    agent.update(interaction(slate, 1, agent.phase, 2))
    np.testing.assert_allclose(state.V, previous_V + np.outer(initial_c, initial_c))


def test_recommend_and_update_are_paired():
    agent, _, _ = make_agent(exploration_steps=0)
    with pytest.raises((ValueError, RuntimeError)):
        agent.update(None)
    slate = agent.recommend(2)
    with pytest.raises((ValueError, RuntimeError)):
        agent.recommend(2)
    agent.update(interaction(slate, 0, agent.phase, 1))


def test_real_helpers_run_reproducibly_through_exploration_and_exploitation():
    def run():
        agent, state, _ = make_agent(exploration_steps=3)
        slates = []
        phases = []
        for t in range(1, 7):
            slate = agent.recommend(2)
            phases.append(agent.phase)
            slates.append(np.array([item.emb.copy() for item in slate]))
            agent.update(interaction(slate, t % 2, agent.phase, t))
        return np.array(slates), phases, state.V

    first, phases, V = run()
    second, second_phases, second_V = run()
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(V, second_V)
    np.testing.assert_allclose(np.linalg.norm(first, axis=2), 1.0)
    assert phases == second_phases
    assert phases[3:] == [Phase.EXPLOIT] * 3
