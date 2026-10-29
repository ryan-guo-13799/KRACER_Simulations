from dataclasses import dataclass

from .item import Item
from .phase import Phase

@dataclass
class Interaction:
    slate: list[Item]
    chosen_id: int
    chosen_item: Item
    experience: float
    utility: float
    phase: Phase
    timestep: int = 0
