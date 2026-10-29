from __future__ import annotations
from dataclasses import dataclass, field, InitVar
import numpy as np
from numpy.typing import NDArray
from config import UserConfig
from config import SimConfig
Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]

@dataclass
class UserState:
    config: InitVar[UserConfig]
    sim_config: InitVar[SimConfig]

    mean: Vector = field(init=False)
    cov: Matrix = field(init=False)
    V: Matrix = field(init=False)

    preference: Vector = field(init=False)
    sigma_u: float = field(init=False)
    risk: float = field(init=False)
    rng: np.random.Generator = field(init=False, repr=False)


    def __post_init__(
        self,
        user_config: UserConfig,
        sim_config: SimConfig,
    ) -> None:
        D = sim_config.ambient_dimension

        self.rng = np.random.default_rng(sim_config.seed)
        self.preference = self.rng.normal(size=D)
        self.preference /= np.linalg.norm(self.preference)
        self.risk = user_config.risk
        self.sigma_u = user_config.sigma_u
        self.mean = np.zeros(D, dtype=np.float64)
        self.cov = np.eye(D, dtype=np.float64) / D
        user_config.set_dimension(D)
        self.V = user_config.lambda0 * np.eye(D, dtype=np.float64)
