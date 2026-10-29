"""Regression checks for the first K-RACER parity step."""

# BEGIN K-RACER PARITY CHANGE (step 1): protect basis/excitation conventions
# and representative origin regret values without importing the origin.
import math

import numpy as np
import pytest

from config import RecommenderConfig, SimConfig, UserConfig
from recommenders.KRACER_functions import ExcitationSlate
from recommenders.startup_functions import make_cone
from simulation_function import run_simulation


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


def test_excitation_globally_flips_a_backward_cap():
    axis, direction = np.eye(3)[:2]
    slate = ExcitationSlate((.1, .5), np.diag([1., 4., 3.]),
                            axis, direction, 3, .5)
    center = np.array([item.emb for item in slate]).mean(axis=0)
    assert center @ axis > 0
    assert center @ direction < 0


@pytest.mark.parametrize('dimension,K,seed,gamma,cutoff,origin_regrets', [
    (5, 2, 6000, .95, 64, (73.72320051911986, 99.505766426168)),
    (5, 2, 6001, .95, 64, (62.066284401649405, 87.61984574768029)),
    (3, 2, 6000, .9, 32, (31.957157486567315, 40.21345279523891)),
    (5, 3, 6000, .95, 32, (83.2710601443518, 108.9221743969934)),
])
def test_representative_kracer_runs_remain_close_to_origin(
    dimension, K, seed, gamma, cutoff, origin_regrets,
):
    sim = SimConfig(ambient_dimension=dimension, num_steps=10000, seed=seed,
                    user_config=UserConfig(sigma_u=.1 / math.sqrt(dimension)))
    rec = RecommenderConfig(slate_size=K, cutoff=cutoff, gamma_ctr=gamma)
    result = run_simulation(sim, rec, 'kracer', cutoff_after_startup=True)
    preference = result.user.user_state.preference
    consumed = math.fsum(1 - float(x.utility) for x in result.interactions)
    slate = math.fsum(1 - float(np.mean([preference @ item.emb for item in x.slate]))
                      for x in result.interactions)
    assert len(result.interactions) == sim.num_steps
    np.testing.assert_allclose((consumed, slate), origin_regrets, rtol=0, atol=1e-6)
# END K-RACER PARITY CHANGE (step 1).
