"""Sparring baseline for the manual recommender interface (K=2)."""

from recommenders.baseline_support.adapter import PairwiseAdapter
from recommenders.baseline_support.sparring import Sparring as OriginalPolicy
from recommenders.baseline_support.oful import OFULConfig


class Sparring(PairwiseAdapter):
    def __init__(self, sim_config, recommender_config, **options):
        options.setdefault("oful_config", OFULConfig(optimizer="fast"))
        super().__init__(sim_config, recommender_config, OriginalPolicy, **options)
