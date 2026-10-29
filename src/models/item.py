from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Item:
    emb: np.array