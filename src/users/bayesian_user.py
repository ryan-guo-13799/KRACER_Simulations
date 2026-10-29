import numpy as np

from models.interaction import Interaction
from models.item import Item
from .user_state import UserState
from protocols.user import User


class BayesianUser(User):
    def __init__(self, user_state: UserState) -> None:
        self.user_state = user_state

    def choose(self, slate: list[Item]) -> Item:
        state = self.user_state
        scores = []
        for item in slate:
            q = item.emb
            expected_utility = state.mean @ q
            uncertainty = q @ state.cov @ q
            scores.append(expected_utility - (state.risk / 2) * uncertainty)

        return slate[int(np.argmax(scores))]

    def update(self, interaction: Interaction) -> None:
        item_emb = interaction.chosen_item.emb
        state = self.user_state

        interaction.utility = state.preference @ item_emb
        interaction.experience = float(interaction.utility + state.rng.normal(0.0, state.sigma_u))
        cov_item = state.cov @ item_emb
        denominator = state.sigma_u**2 + item_emb @ cov_item
        prediction_error = interaction.experience - state.mean @ item_emb

        state.mean = state.mean + cov_item * (prediction_error / denominator)
        state.cov = state.cov - np.outer(cov_item, cov_item) / denominator
        state.V += np.outer(item_emb, item_emb)
