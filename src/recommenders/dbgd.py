"""DBGD baseline for the manual recommender interface (K=2)."""

from recommenders.baseline_support.adapter import PairwiseAdapter
from recommenders.baseline_support.dbgd import DBGD as OriginalPolicy


class DBGD(PairwiseAdapter):
    def __init__(self, sim_config, recommender_config, **options):
        super().__init__(sim_config, recommender_config, OriginalPolicy, **options)
