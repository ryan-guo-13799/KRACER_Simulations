from typing import Protocol

from models.item import Item
from models.interaction import Interaction


class User(Protocol):

    def choose(self, slate: list[Item]) -> Item:
        ...

    def update(self, interaction: Interaction) -> None:
        ...