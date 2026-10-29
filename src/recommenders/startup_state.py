from dataclasses import dataclass, field

import numpy as np

from config import SimConfig


@dataclass(frozen=True)
class StartupConstraint:
    """A public comparison: sign * preference[coordinate] >= -margin."""

    t: int
    coordinate: int
    sign: int
    margin: float


@dataclass
class StartupState:
    """Public information collected by coordinate-sign startup."""

    sim_config: SimConfig
    t: int = field(init=False, default=1)
    V: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    signs: np.ndarray = field(init=False)
    margins: np.ndarray = field(init=False)
    constraints: list[StartupConstraint] = field(init=False, default_factory=list)
    status: str = field(init=False, default="PENDING")

    def __post_init__(self) -> None:
        D = self.sim_config.ambient_dimension
        self.V = self.sim_config.user_config.lambda0 * np.eye(D)
        self.counts = np.zeros(D, dtype=int)
        self.signs = np.zeros(D, dtype=int)  # zero means  the axis is not yet certified
        self.margins = np.full(D, np.nan)
