from dataclasses import dataclass, field
import math
from numbers import Integral


@dataclass
class RecommenderConfig:
    slate_size: int = 5
    cutoff: int = 100 # T_0
    gamma_ctr: float = 0.9
    startup_half_angle: float | None = None  # radians; None chooses an angle for D
    analytical_gate_size: float = 4.0  # used only by the analytical certificate ablation


@dataclass
class UserConfig:
    sigma_u: float = 0.1
    risk: float = 1.0
    dimension: int = 5
    lambda0: float = field(init=False)

    def __post_init__(self) -> None:
        self.set_dimension(self.dimension)

    def set_dimension(self, dimension: int) -> None:
        self.dimension = dimension
        self.lambda0 = dimension * self.sigma_u**2


@dataclass
class SimConfig:
    ambient_dimension: int = 5
    num_steps: int = 1000
    seed: int = 42
    user_config: UserConfig = field(default_factory=UserConfig)
    delta_conf: float = 0.05

    def __post_init__(self) -> None:
        self.user_config.set_dimension(self.ambient_dimension)

    def h_t(self, t: int) -> float:
        """Return h_t = beta_t + zeta_t for the Bayesian-CARA user."""
        D = self.ambient_dimension
        sigma = self.user_config.sigma_u
        risk = self.user_config.risk
        if not isinstance(t, Integral) or isinstance(t, bool) or t < 1:
            raise ValueError("t must be a positive integer")
        if not isinstance(D, Integral) or isinstance(D, bool) or D < 1:
            raise ValueError("ambient_dimension must be a positive integer")
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(risk):
            raise ValueError("sigma_u must be positive and risk must be finite")
        if not 0 < self.delta_conf < 1:
            raise ValueError("delta_conf must be between 0 and 1")

        estimation = sigma * (
            math.sqrt(D)
            + math.sqrt(D * math.log1p((t - 1) / (D**2 * sigma**2))
                        + 2 * math.log(1 / self.delta_conf))
        )
        decision = abs(risk) * sigma / math.sqrt(D)
        return estimation + decision
