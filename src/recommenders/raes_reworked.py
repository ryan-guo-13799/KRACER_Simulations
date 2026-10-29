"""RAESReworked baseline for the manual recommender interface (K=2)."""

from recommenders.baseline_support.adapter import PairwiseAdapter
from recommenders.baseline_support.raes import RAESUpdated as OriginalPolicy


class RAESReworked(PairwiseAdapter):
    def __init__(self, sim_config, recommender_config):
        super().__init__(sim_config, recommender_config, OriginalPolicy, raes=True)
