"""Public agent contract used by the episode engine and checkpoint evaluator."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """A replaceable policy: no private user state or true preference is supplied.

    Proposals and observations use the information matrix before consumption.
    Membership is an evaluator-only audit, never an input to policy decisions.
    New implementations execute independent cutoffs unless they explicitly
    certify that their next decision is independent of a shorter cutoff.
    """

    @abstractmethod
    def propose(self, t, V_pre, beta_t, zeta_t):
        """Return a unit-action slate and its phase at absolute round t."""

    @abstractmethod
    def observe(self, t, slate, winner, V_pre, beta_t, zeta_t):
        """Update public policy state from the actual displayed slate and winner."""

    @abstractmethod
    def estimate(self):
        """Return the recommendation frozen after preparation."""

    @abstractmethod
    def membership(self, u):
        """Evaluator-only enclosure check for the hidden true preference."""

    @abstractmethod
    def diagnostics(self):
        """Return public phase and numerical diagnostics."""

    def cone_radius(self):
        """Return tangent-cone radius, or None for a non-cone policy."""
        return None

    def prefix_is_safe(self, t, cutoff):
        """Conservatively refuse checkpoint reuse for an unknown implementation."""
        return False

    def set_config(self, cfg):
        """Install a fork's configuration without sharing mutable policy state."""
        self.cfg = cfg

    def retained_constraints(self):
        """Expose real public comparisons to the evaluator."""
        return tuple(getattr(self, "public_constraints", ()))
