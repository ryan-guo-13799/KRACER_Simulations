from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray

from config import RecommenderConfig
from config import SimConfig
from models.phase import Phase
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
AngularInterval = tuple[float, float]

@dataclass
class KRACERState:
    recommender_config: InitVar[RecommenderConfig]
    sim_config: InitVar[SimConfig]

    c: Vector | None = field(init=False, default=None) # axis of the cone
    F: Matrix | None = field(init=False, default=None) # eigenvectors
    W: Vector | None = field(init=False, default=None) # diagonal of the shape matrix
    I: AngularInterval | None = field(init=False, default=None) # angular inteval w.r.t. the largest axis
    i: int = field(init=False, default=None) # largest axis index

    V: Matrix | None = field(init=False, default=None)
    
