"""Connect the original pairwise policies to the manual recommender protocol."""

from types import SimpleNamespace
import numpy as np

from models.item import Item
from models.phase import Phase
from protocols.recommender import Recommender
from .dbgd import DBGD


class PairwiseAdapter(Recommender):
    def __init__(self, sim_config, recommender_config, policy_class, *, raes=False, **options):
        if recommender_config.slate_size != 2:
            raise ValueError("This baseline requires a slate size of 2")
        self.config = sim_config
        self.t = 1
        self.phase = None
        self._slate = None
        dimension = sim_config.ambient_dimension
        self.V = sim_config.user_config.lambda0 * np.eye(dimension)
        # These names are those used by the original baseline implementations.
        policy_config = SimpleNamespace(
            D=dimension, K=2, T=sim_config.num_steps,
            T0=recommender_config.cutoff, kappa_cut=1.05,
        )
        # BEGIN DBGD SEED FIX: use origin's separate stream, not the user's seed.
        policy_seed = (np.random.SeedSequence([sim_config.seed, 0xDB6D, 17])
                       if policy_class is DBGD else sim_config.seed)
        self.policy = (policy_class(policy_config, None) if raes else
                       policy_class(policy_config, seed=policy_seed, **options))
        # END DBGD SEED FIX.

    def recommend(self, k):
        if k != 2:
            raise ValueError("This baseline requires a slate size of 2")
        if self._slate is not None:
            raise RuntimeError("Update the previous recommendation first")
        if self.t > self.config.num_steps:
            raise RuntimeError("The simulation has finished")
        slate, phase = self.policy.propose(self.t, self.V, self.config.h_t(self.t), 0.0)
        self.phase = {"localize": Phase.LOCALIZE, "exploit": Phase.EXPLOIT}.get(phase, Phase.EXCITE)
        self._slate = [Item(vector.copy()) for vector in slate]
        return self._slate

    def update(self, interaction):
        if self._slate is None:
            raise RuntimeError("Recommend a slate before updating")
        winner = interaction.chosen_id
        if not isinstance(winner, (int, np.integer)) or isinstance(winner, (bool, np.bool_)) or winner not in (0, 1):
            raise ValueError("chosen_id must be 0 or 1")
        if len(interaction.slate) != 2 or any(
            not np.array_equal(actual.emb, expected.emb)
            for actual, expected in zip(interaction.slate, self._slate)
        ) or not np.array_equal(interaction.chosen_item.emb, self._slate[winner].emb):
            raise ValueError("Interaction must match the recommended slate and winner")
        slate = np.array([item.emb for item in self._slate])
        self.policy.observe(self.t, slate, winner, self.V, self.config.h_t(self.t), 0.0)
        chosen = self._slate[winner].emb
        self.V += np.outer(chosen, chosen)
        self.t += 1
        self._slate = None

    def estimate(self):
        return self.policy.estimate()
