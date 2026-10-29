from typing import Protocol

from models.item import Item
from models.interaction import Interaction


class Recommender(Protocol):

    def recommend(self, k: int) -> list[Item]:
        ...

    def update(self, interaction: Interaction) -> None:
        ...