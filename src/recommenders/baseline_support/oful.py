"""OFUL confidence-set optimization for continuous spherical actions.

Features are phi(q)=(q,1). Rewards are public binary wins, not user rewards.
The confidence radius is Abbasi-Yadkori et al. (2011), Theorem 2. The reference
oracle uses multistart SLSQP and sphere Newton refinement. The explicitly
approximate fast oracle uses a bounded projected ascent on the same confidence
objective. Neither oracle is a certified global maximizer.
"""

from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from .dueling import unit


@dataclass(frozen=True)
class OFULConfig:
    regularization: float = 1.0
    confidence_delta: float = 0.05
    noise_scale: float = 0.5
    parameter_bound: float = 1.0
    random_starts: int = 8
    max_iterations: int = 200
    tolerance: float = 1e-10
    feasibility_tolerance: float = 1e-8
    stationarity_tolerance: float = 1e-6
    polish_max_iterations: int = 50
    intrinsic_max_iterations: int = 1000
    optimizer: str = "reference"
    optimizer_version: str = "oful-oracle-v2"
    fast_random_starts: int = 2
    fast_max_iterations: int = 16
    fast_step_size: float = 0.5
    fast_line_search_steps: int = 6

    def __post_init__(self):
        for name in (
            "regularization",
            "tolerance",
            "feasibility_tolerance",
            "stationarity_tolerance",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.confidence_delta < 1:
            raise ValueError("Require 0<confidence_delta<1")
        for name in ("noise_scale", "parameter_bound"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.random_starts, int) or self.random_starts < 1:
            raise ValueError("At least one seeded random start is required")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(self.polish_max_iterations, int) or self.polish_max_iterations < 1:
            raise ValueError("polish_max_iterations must be a positive integer")
        if not isinstance(self.intrinsic_max_iterations, int) or self.intrinsic_max_iterations < 1:
            raise ValueError("intrinsic_max_iterations must be a positive integer")
        if self.optimizer not in ("reference", "fast"):
            raise ValueError("optimizer must be reference or fast")
        if self.optimizer_version != "oful-oracle-v2":
            raise ValueError("Unsupported OFUL optimizer version")
        for name in ("fast_random_starts", "fast_max_iterations", "fast_line_search_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.fast_step_size) or not 0 < self.fast_step_size <= 1:
            raise ValueError("fast_step_size must be finite and in (0,1]")


class OFUL:
    def __init__(self, dimension, *, seed=0, config=None):
        if not isinstance(dimension, int) or dimension < 2:
            raise ValueError("OFUL requires action dimension >=2")
        self.D, self.p = dimension, dimension + 1
        self.config = OFULConfig() if config is None else config
        self.rng = np.random.default_rng(seed)
        self.total_updates = self.selections = self.resets = 0
        self.solver_failures = self.solver_attempts = 0
        self.polished_starts = self.polishing_iterations = 0
        self.intrinsic_attempts = self.intrinsic_iterations = 0
        self.fast_iterations = self.fast_objective_evaluations = 0
        self.last_optimization = {}
        self.reset_statistics()

    def reset_statistics(self):
        """Clear epoch statistics, preserving the independently seeded RNG stream."""
        self.A = self.config.regularization * np.eye(self.p)
        self.b = np.zeros(self.p)
        self.n = 0
        self._previous = None
        self.resets += 1

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.total_updates = self.selections = self.resets = 0
        self.solver_failures = self.solver_attempts = 0
        self.polished_starts = self.polishing_iterations = 0
        self.intrinsic_attempts = self.intrinsic_iterations = 0
        self.fast_iterations = self.fast_objective_evaluations = 0
        self.last_optimization = {}
        self.reset_statistics()

    def features(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.D,) or not np.all(np.isfinite(q)):
            raise ValueError("Invalid action")
        if not np.isclose(np.linalg.norm(q), 1.0, rtol=0, atol=1e-8):
            raise ValueError("OFUL actions must be unit vectors")
        return np.r_[q, 1.0]

    def statistics(self):
        factor = cho_factor(self.A, lower=True, check_finite=True)
        theta = cho_solve(factor, self.b)
        inverse = cho_solve(factor, np.eye(self.p))
        inverse = (inverse + inverse.T) / 2
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        logratio = max(0.0, float(logdet - self.p * np.log(self.config.regularization)))
        rho = (
            self.config.noise_scale
            * np.sqrt(logratio + 2 * np.log(1 / self.config.confidence_delta))
            + np.sqrt(self.config.regularization) * self.config.parameter_bound
        )
        return theta, inverse, float(rho)

    @property
    def theta(self):
        return self.statistics()[0]

    @property
    def confidence_radius(self):
        return self.statistics()[2]

    def ucb(self, q):
        phi = self.features(q)
        theta, inverse, rho = self.statistics()
        return float(phi @ theta + rho * np.sqrt(phi @ inverse @ phi))

    def optimistic_parameter(self, q):
        phi = self.features(q)
        theta, inverse, rho = self.statistics()
        return theta + rho * (inverse @ phi) / np.sqrt(phi @ inverse @ phi)

    def _objective(self, q, theta, inverse, rho, *, hessian=False):
        """Unchanged confidence objective, with analytic ambient derivatives."""
        phi = np.r_[q, 1.0]
        product = inverse @ phi
        variance = float(phi @ product)
        if variance <= 0 or not np.isfinite(variance):
            raise FloatingPointError("Invalid positive-definite confidence geometry")
        norm = np.sqrt(variance)
        gradient = theta[: self.D] + rho * product[: self.D] / norm
        value = float(theta @ phi + rho * norm)
        if not hessian:
            return value, gradient
        curvature = rho * (
            inverse[: self.D, : self.D] / norm
            - np.outer(product[: self.D], product[: self.D]) / norm**3
        )
        return value, gradient, curvature

    def _polish(self, q, theta, inverse, rho, *, intrinsic=False):
        """Refine a feasible candidate without relaxing the stationarity test.

        Solve the Newton system on q's tangent space for the same maximization
        objective; if its curvature is not negative definite, use a projected
        ascent direction. Retraction and backtracking keep actions on the sphere
        and require objective ascent (up to floating-point roundoff). No policy
        RNG draws or alternative rewards enter this refinement.
        """
        cfg = self.config
        iterations = 0
        limit = cfg.intrinsic_max_iterations if intrinsic else cfg.polish_max_iterations
        for _ in range(limit):
            value, gradient, curvature = self._objective(q, theta, inverse, rho, hessian=True)
            tangent_gradient = gradient - q * (q @ gradient)
            residual = float(np.linalg.norm(tangent_gradient))
            if residual <= cfg.stationarity_tolerance:
                break
            # Orthonormal basis of the tangent space, with no chart singularity.
            frame = np.linalg.svd(q[None, :], full_matrices=True)[2][1:].T
            h = frame.T @ gradient
            B = frame.T @ (curvature - float(q @ gradient) * np.eye(self.D)) @ frame
            B = (B + B.T) / 2
            if intrinsic:
                # Positive definite ascent preconditioner even at indefinite
                # curvature or flat directions. This optimizes the same UCB,
                # with every iterate retracted onto the sphere.
                eigenvalues, eigenvectors = np.linalg.eigh(B)
                floor = 1e-6 * max(1.0, float(np.max(np.abs(eigenvalues))))
                inverse_curvature = 1 / np.maximum(-eigenvalues, floor)
                direction = eigenvectors @ (inverse_curvature * (eigenvectors.T @ h))
            elif np.linalg.eigvalsh(B)[-1] < -1e-12:
                direction = np.linalg.solve(-B, h)
            else:
                direction = h / max(1.0, float(np.linalg.norm(B, ord=2)))
            if float(h @ direction) <= 0 or not np.all(np.isfinite(direction)):
                direction = h.copy()
            length = np.linalg.norm(direction)
            if length > 0.5:
                direction *= 0.5 / length
            slope = float(h @ direction)
            rounding = 16 * np.finfo(float).eps * max(1.0, abs(value))
            accepted = False
            for exponent in range(32):
                step = 2.0 ** (-exponent)
                trial = unit(q + step * (frame @ direction))
                trial_value, trial_gradient = self._objective(trial, theta, inverse, rho)
                trial_residual = float(
                    np.linalg.norm(trial_gradient - trial * (trial @ trial_gradient))
                )
                armijo = 1e-4 * step * slope
                # At machine-precision objective differences, also require a
                # smaller independently computed stationarity residual.
                if trial_value - value >= max(armijo, rounding) or (
                    armijo <= rounding
                    and trial_value >= value - rounding
                    and trial_residual < residual
                ):
                    q = trial
                    accepted = True
                    iterations += 1
                    break
            if not accepted:
                break
        return q, iterations

    def _select_fast(self, theta, inverse, rho, *, starts=None):
        """Bounded ascent of the actual UCB, accepting approximate stationarity.

        Score a small public-state/seeded candidate pool, then improve its best
        member. Every accepted retraction increases the same confidence-set
        objective. Hitting the work budget is an advertised approximation, not
        an optimizer failure or a substitution of a greedy/random policy.
        """
        cfg = self.config
        if starts is None:
            starts = [unit(self.rng.normal(size=self.D)) for _ in range(cfg.fast_random_starts)]
            if self._previous is not None:
                starts.append(self._previous.copy())
            for vector in (theta[: self.D], inverse[: self.D, -1]):
                if np.linalg.norm(vector) > 1e-14:
                    direction = unit(vector)
                    starts.extend((direction, -direction))
            # A cheap uncertainty candidate; no eigenframe or tangent SVD.
            axis = np.eye(self.D)[int(np.argmax(np.diag(inverse)[: self.D]))]
            starts.extend((axis, -axis))
        else:
            starts = [np.asarray(start, dtype=float).copy() for start in starts]
            if not starts:
                raise ValueError("Optimizer replay requires at least one start")
        for start in starts:
            self.features(start)
        starts = [unit(start) for start in starts]

        evaluations = 0
        self.solver_attempts += 1

        def objective(q):
            nonlocal evaluations
            evaluations += 1
            value, gradient = self._objective(q, theta, inverse, rho)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError("Nonfinite fast OFUL confidence objective")
            return value, gradient

        try:
            best = None
            for start in starts:
                value, gradient = objective(start)
                if best is None or value > best[0] + 1e-12:
                    best = value, start.copy(), gradient
            value, q, gradient = best
            initial_value = value
            accepted = 0
            termination = "iteration_limit"
            for _ in range(cfg.fast_max_iterations):
                tangent = gradient - q * (q @ gradient)
                residual = float(np.linalg.norm(tangent))
                if residual <= cfg.stationarity_tolerance:
                    termination = "stationary"
                    break
                direction = tangent / residual
                step = cfg.fast_step_size
                for _ in range(cfg.fast_line_search_steps):
                    trial = unit(q + step * direction)
                    trial_value, trial_gradient = objective(trial)
                    if trial_value >= value + 1e-4 * step * residual:
                        q, value, gradient = trial, trial_value, trial_gradient
                        accepted += 1
                        break
                    step *= 0.5
                else:
                    termination = "line_search_limit"
                    break
            residual = float(np.linalg.norm(gradient - q * (q @ gradient)))
            if not np.isfinite(residual) or value < initial_value:
                raise FloatingPointError("Invalid fast OFUL ascent result")
            self.features(q)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as cause:
            self.solver_failures += 1
            self.fast_objective_evaluations += evaluations
            error = RuntimeError(f"Fast OFUL objective optimization failed: {cause}")
            error.oful_state = {
                "dimension": self.D,
                "config": asdict(cfg),
                "A": self.A.tolist(),
                "b": self.b.tolist(),
                "n": self.n,
                "starts": [start.tolist() for start in starts],
            }
            raise error from cause

        upper = float(
            theta[-1]
            + np.linalg.norm(theta[: self.D])
            + rho * np.sqrt(2 * np.linalg.eigvalsh(inverse)[-1])
        )
        self.last_optimization = {
            "optimizer": cfg.optimizer,
            "optimizer_version": cfg.optimizer_version,
            "solver": "bounded_projected_sphere_ascent",
            "global_optimality_certified": False,
            "bounded_approximation": True,
            "objective": value,
            "initial_objective": initial_value,
            "objective_improvement": value - initial_value,
            "stationarity_residual": residual,
            "stationarity_required": False,
            "stationarity_satisfied": bool(residual <= cfg.stationarity_tolerance),
            "feasibility_residual": float(abs(q @ q - 1)),
            "starts": len(starts),
            "optimized_starts": 1,
            "converged_starts": int(residual <= cfg.stationarity_tolerance),
            "slsqp_successes": 0,
            "polished_starts": 0,
            "intrinsic_used": False,
            "intrinsic_iterations": 0,
            "polishing_iterations": 0,
            "iterations": accepted,
            "objective_evaluations": evaluations,
            "termination": termination,
            "analytic_upper_bound": upper,
            "analytic_gap_upper_bound": max(0.0, upper - value),
        }
        self._previous = q.copy()
        self.selections += 1
        self.fast_iterations += accepted
        self.fast_objective_evaluations += evaluations
        return q

    def select_action(self, *, starts=None):
        theta, inverse, rho = self.statistics()
        cfg = self.config
        if cfg.optimizer == "fast":
            return self._select_fast(theta, inverse, rho, starts=starts)

        def objective(q):
            return self._objective(q, theta, inverse, rho)

        # Random start first: initial isotropic objective has a continuum of
        # maxima. Independent learner streams avoid imposing identical choices.
        if starts is None:
            starts = [unit(self.rng.normal(size=self.D)) for _ in range(cfg.random_starts)]
            if self._previous is not None:
                starts.append(self._previous.copy())
            for vector in (theta[: self.D], inverse[: self.D, -1]):
                if np.linalg.norm(vector) > 1e-14:
                    starts.extend((unit(vector), -unit(vector)))
            _, eigenvectors = np.linalg.eigh(inverse[: self.D, : self.D])
            starts.extend(eigenvectors.T)
            starts.extend(-eigenvectors.T)
        else:
            # Diagnostic replay of stored PUBLIC optimizer starts; production
            # learners use the seeded generation above without passing inputs.
            starts = [np.asarray(start, dtype=float).copy() for start in starts]
            if not starts:
                raise ValueError("Optimizer replay requires at least one start")
            for start in starts:
                self.features(start)
        best = None
        successful = 0
        polished = polish_iterations = feasible = slsqp_successes = 0
        smallest_residual = float("inf")
        slsqp_results = []
        intrinsic_used = False
        intrinsic_steps = 0
        constraint = {"type": "eq", "fun": lambda q: float(q @ q - 1), "jac": lambda q: 2 * q}
        for start in starts:
            self.solver_attempts += 1
            try:
                result = minimize(
                    lambda q: -objective(q)[0],
                    start,
                    jac=lambda q: -objective(q)[1],
                    method="SLSQP",
                    constraints=constraint,
                    options={"ftol": cfg.tolerance, "maxiter": cfg.max_iterations},
                )
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                slsqp_results.append({"exception": str(error)})
                self.solver_failures += 1
                continue
            slsqp_results.append(
                {
                    "success": bool(result.success),
                    "status": getattr(result, "status", None),
                    "message": str(getattr(result, "message", "")),
                    "norm_error": float(abs(result.x @ result.x - 1)),
                }
            )
            slsqp_successes += int(result.success)
            if (
                not np.all(np.isfinite(result.x))
                or abs(result.x @ result.x - 1) > cfg.feasibility_tolerance
            ):
                self.solver_failures += 1
                continue
            q = unit(result.x)
            feasible += 1
            value, gradient = objective(q)
            residual = float(np.linalg.norm(gradient - q * (q @ gradient)))
            if residual > cfg.stationarity_tolerance:
                q, iterations = self._polish(q, theta, inverse, rho)
                polished += 1
                polish_iterations += iterations
                value, gradient = objective(q)
                residual = float(np.linalg.norm(gradient - q * (q @ gradient)))
            smallest_residual = min(smallest_residual, residual)
            # Validate convergence ourselves: a solver status flag alone is
            # neither sufficient nor necessary after independent refinement.
            if residual > cfg.stationarity_tolerance:
                self.solver_failures += 1
                continue
            successful += 1
            if best is None or value > best[0] + 1e-12:
                best = value, q.copy(), residual
        if best is None:
            intrinsic_used = True
            # Do not normalize an invalid SLSQP result into an action. Instead,
            # run an independent optimizer from the original valid start set.
            # No additional policy randomness or reward observations are used.
            for start in starts:
                q, iterations = self._polish(unit(start), theta, inverse, rho, intrinsic=True)
                self.intrinsic_attempts += 1
                self.intrinsic_iterations += iterations
                intrinsic_steps += iterations
                value, gradient = objective(q)
                residual = float(np.linalg.norm(gradient - q * (q @ gradient)))
                smallest_residual = min(smallest_residual, residual)
                if (
                    not np.isfinite(value)
                    or not np.isfinite(residual)
                    or residual > cfg.stationarity_tolerance
                ):
                    continue
                successful += 1
                if best is None or value > best[0] + 1e-12:
                    best = value, q.copy(), residual
        if best is None:
            error = RuntimeError(
                "OFUL optimization failed: no feasible converged stationary candidate; "
                f"D={self.D}, updates={self.n}, feasible_starts={feasible}/{len(starts)}, "
                f"SLSQP_successes={slsqp_successes}, intrinsic_starts={len(starts)}, "
                f"min_tangent_residual={smallest_residual:.3g}, "
                f"required={cfg.stationarity_tolerance:.3g}"
            )
            # Reproducible PUBLIC bandit state: contains no user V, estimate,
            # true preference, private experience, or reward outside binary wins.
            error.oful_state = {
                "dimension": self.D,
                "config": asdict(cfg),
                "A": self.A.tolist(),
                "b": self.b.tolist(),
                "n": self.n,
                "starts": [start.tolist() for start in starts],
                "slsqp_results": slsqp_results,
                "min_tangent_residual": smallest_residual,
            }
            raise error
        value, q, residual = best
        # Cauchy-Schwarz supplies a conservative analytic global upper bound;
        # the gap is logged, never described as a certificate of optimality.
        upper = float(
            theta[-1]
            + np.linalg.norm(theta[: self.D])
            + rho * np.sqrt(2 * np.linalg.eigvalsh(inverse)[-1])
        )
        self.last_optimization = {
            "optimizer": cfg.optimizer,
            "optimizer_version": cfg.optimizer_version,
            "solver": "multistart_slsqp_intrinsic_sphere_rescue",
            "global_optimality_certified": False,
            "bounded_approximation": False,
            "stationarity_required": True,
            "objective": value,
            "stationarity_residual": residual,
            "feasibility_residual": float(abs(q @ q - 1)),
            "starts": len(starts),
            "converged_starts": successful,
            "slsqp_successes": slsqp_successes,
            "polished_starts": polished,
            "intrinsic_used": intrinsic_used,
            "intrinsic_iterations": intrinsic_steps,
            "polishing_iterations": polish_iterations,
            "analytic_upper_bound": upper,
            "analytic_gap_upper_bound": max(0.0, upper - value),
        }
        self._previous = q.copy()
        self.polished_starts += polished
        self.polishing_iterations += polish_iterations
        self.selections += 1
        return q

    def update(self, q, binary_reward):
        if binary_reward not in (0, 1) or not np.isscalar(binary_reward):
            raise ValueError("OFUL accepts only a public binary duel reward")
        phi = self.features(q)
        self.A += np.outer(phi, phi)
        self.b += float(binary_reward) * phi
        self.n += 1
        self.total_updates += 1

    def diagnostics(self):
        return {
            "epoch_updates": self.n,
            "total_updates": self.total_updates,
            "selections": self.selections,
            "statistics_resets": self.resets,
            "solver_attempts": self.solver_attempts,
            "solver_failures": self.solver_failures,
            "polished_starts": self.polished_starts,
            "polishing_iterations": self.polishing_iterations,
            "intrinsic_attempts": self.intrinsic_attempts,
            "intrinsic_iterations": self.intrinsic_iterations,
            "fast_iterations": self.fast_iterations,
            "fast_objective_evaluations": self.fast_objective_evaluations,
            "feature_dimension": self.p,
            "confidence_radius": self.confidence_radius,
            "parameters": asdict(self.config),
            "last_optimization": self.last_optimization,
        }
