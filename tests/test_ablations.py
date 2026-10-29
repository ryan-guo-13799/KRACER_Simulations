import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from models.interaction import Interaction
from models.phase import Phase
from recommenders import KRACER as kracer_module
from recommenders import analytical_certificate as analytical_module
from recommenders.analytical_certificate import AnalyticalCertificate
from recommenders.minimum_eigenvector import MinimumEigenvector
from recommenders.without_restart import WithoutRestart
from simulation_state import CONE_RECOMMENDERS, run_simulation


def make_agent(algorithm, *, cutoff=8, W=(0.5, 0.3), V=None):
    sim = SimConfig(ambient_dimension=3, num_steps=10,
                    user_config=UserConfig(sigma_u=0.1))
    config = RecommenderConfig(slate_size=2, cutoff=cutoff, gamma_ctr=0.5)
    state = kracer_module.KRACERState(config, sim)
    state.c = np.array([1.0, 0.0, 0.0])
    state.F = np.eye(3)[:, 1:]
    state.W = np.array(W)
    state.V = np.eye(3) if V is None else V.copy()
    return algorithm(state, sim, config), state, sim


def respond(agent, slate, winner=0):
    agent.update(Interaction(slate, winner, slate[winner], 0.0, 0.0,
                              agent.phase, agent.t))


def test_minimum_eigenvector_changes_excitation_center_but_keeps_cap_radius(monkeypatch):
    agent, _, _ = make_agent(MinimumEigenvector, V=np.diag([4.0, 1.0, 2.0]))
    monkeypatch.setattr(agent, "_find_depth", lambda max_depth: None)
    slate = agent.recommend(2)
    assert agent.phase == Phase.EXCITE
    vectors = np.array([item.emb for item in slate])
    center = vectors.mean(axis=0)
    center /= np.linalg.norm(center)
    np.testing.assert_allclose(np.abs(center), [0, 1, 0], atol=1e-12)
    radius = min(0.25, 1 / np.sqrt(2 * (1 + 1 / agent.user_lambda0)))
    assert np.linalg.norm(vectors[0] - vectors[1]) == pytest.approx(2 * radius)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_analytical_admission_uses_geometric_condition_without_tree(monkeypatch):
    agent, state, sim = make_agent(AnalyticalCertificate)
    sim.h_t = lambda t: 0.0
    monkeypatch.setattr(kracer_module, "TreeCertificate",
                        lambda *args: pytest.fail("analytical gate must not run the winner tree"))
    slate = agent.recommend(2)
    assert agent.phase == Phase.LOCALIZE
    alpha = np.arctan(np.sqrt(state.W[state.i]))
    C_d = 2.0  # D=3: sqrt(d) * (d/(d-1))**((d-1)/2)
    C_i = C_d * 2 * alpha / np.sin(2 * alpha)
    assert C_i / 2**agent._remaining_depth <= agent.gamma_ctr
    if agent._remaining_depth > 1:
        assert C_i / 2**(agent._remaining_depth - 1) > agent.gamma_ctr


def test_analytical_gate_includes_future_confidence_in_its_window():
    agent, _, sim = make_agent(AnalyticalCertificate)
    agent._prepare_interval()
    sim.h_t = lambda t: 0.0
    assert agent._certificate(3) <= agent.gamma_ctr
    sim.h_t = lambda t: 100.0 if t == agent.t + 1 else 0.0
    assert agent._certificate(3) > agent.gamma_ctr


def test_analytical_failed_actual_contraction_preserves_cone(monkeypatch):
    agent, state, sim = make_agent(AnalyticalCertificate)
    sim.h_t = lambda t: 0.0
    original = [x.copy() for x in (state.c, state.F, state.W)]

    def no_cut(I, V, h, winner):
        slate = kracer_module.EQ(I, state.c, state.F[:, state.i], agent.K)
        return I, slate[winner]

    monkeypatch.setattr(agent, "_cut", no_cut)
    slate = agent.recommend(2)
    depth = agent._remaining_depth
    for j in range(depth):
        if j:
            slate = agent.recommend(2)
        respond(agent, slate)
    for value, before in zip((state.c, state.F, state.W), original):
        np.testing.assert_array_equal(value, before)
    agent.recommend(2)
    assert agent.phase == Phase.EXPLOIT


def test_analytical_numerical_restart_failure_preserves_cone(monkeypatch):
    agent, state, _ = make_agent(AnalyticalCertificate)
    agent._prepare_interval()
    before = state.W.copy()

    def failed_restart(*args):
        raise FloatingPointError("unresolved restart")

    monkeypatch.setattr(analytical_module, "Restart", failed_restart)
    agent._finish_round()
    np.testing.assert_array_equal(state.W, before)
    assert agent.cutoff == agent.t


def test_fixed_chart_selects_remaining_width_even_for_smaller_initial_axis():
    agent, state, sim = make_agent(WithoutRestart, W=(1.0, 0.25))
    sim.h_t = lambda t: 0.0
    agent.intervals[0] = [-0.01, 0.01]
    slate = agent.recommend(2)
    assert state.i == 1
    assert agent.phase == Phase.LOCALIZE
    respond(agent, slate)


def test_fixed_chart_preserves_cone_and_exploits_tangent_box_midpoint():
    agent, state, sim = make_agent(WithoutRestart, cutoff=3)
    sim.h_t = lambda t: 0.0
    before = [x.copy() for x in (state.c, state.F, state.W)]
    for _ in range(3):
        slate = agent.recommend(2)
        respond(agent, slate, 0)
    for actual, original in zip((state.c, state.F, state.W), before):
        np.testing.assert_array_equal(actual, original)
    midpoint = np.tan(agent.intervals).mean(axis=1)
    raw = state.c + state.F @ midpoint
    expected = raw / np.linalg.norm(raw)
    slate = agent.recommend(2)
    assert agent.phase == Phase.EXPLOIT
    for item in slate:
        np.testing.assert_allclose(item.emb, expected)


def test_fixed_chart_tree_matches_perfect_user_tangent_bin_widths():
    agent, state, sim = make_agent(WithoutRestart)
    sim.h_t = lambda t: 0.0
    agent._prepare_interval()
    A, B = state.I
    boundaries = np.linspace(A, B, 5)  # K=2, depth=2 gives four exact bins.
    worst = max(np.diff(np.tan(boundaries))) / (np.tan(B) - np.tan(A))
    agent.gamma_ctr = (0.5 + worst) / 2
    assert agent._find_depth(2) == 2
    agent.gamma_ctr = worst - 1e-10
    assert agent._find_depth(2) is None


def test_fixed_chart_tree_builds_each_branch_once(monkeypatch):
    agent, _, sim = make_agent(WithoutRestart)
    sim.h_t = lambda t: 0.0
    agent.gamma_ctr = 0.0
    agent._prepare_interval()
    original = agent._cut
    calls = []

    def record(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(agent, "_cut", record)
    assert agent._find_depth(3) is None
    assert len(calls) == 2 + 4 + 8


@pytest.mark.parametrize("algorithm", list(CONE_RECOMMENDERS))
def test_each_variant_runs_the_shared_user_loop_reproducibly(algorithm):
    def run():
        sim = SimConfig(ambient_dimension=3, num_steps=15, seed=42,
                        user_config=UserConfig(sigma_u=0.01))
        config = RecommenderConfig(slate_size=2, cutoff=10, startup_half_angle=1.5)
        return run_simulation(sim, config, algorithm)

    result, repeat = run(), run()
    assert len(result.interactions) == 15
    np.testing.assert_array_equal([x.experience for x in result.interactions],
                                   [x.experience for x in repeat.interactions])
    np.testing.assert_allclose(result.user.user_state.V, result.recommender.state.V)
    np.testing.assert_allclose(result.user.user_state.cov,
                               0.01**2 * np.linalg.inv(result.recommender.state.V))
