from typing import List
import logging
from beaker.cache import CacheManager

from apxinfer.core.utils import XIPFeatureVec, XIPRequest, XIPQueryConfig
from apxinfer.core.data import XIPDataLoader

fcache_manager = CacheManager(cache_regions={'feature': {'type': 'memory', 'expire': 3600}})
logging.basicConfig(level=logging.INFO)


class XIPQuery:
    def __init__(self, key: str, data_loader: XIPDataLoader,
                 fnames: List[str], cfg_pools: List[XIPQueryConfig],
                 enable_cache: bool = False) -> None:
        self.key = key
        self.data_loader = data_loader
        self.fnames = fnames
        self.n_features = len(fnames)
        self.cfg_pools = cfg_pools

        self.logger = logging.getLogger(f'XIPQuery-{key}')

        # cache for query running
        if enable_cache:
            self.run = fcache_manager.cache('feature', expire=60)(self.run)

    def run(self, request: XIPRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        raise NotImplementedError

    def estimate_cost(self, request: XIPRequest, qcfg: XIPQueryConfig) -> float:
        """ Estimate the cost of extracting features for a request """
        raise NotImplementedError

    def estimate_inc_cost(self, request: XIPRequest, qcfg: XIPQueryConfig) -> float:
        """ Estimate the incremental cost of extracting features for a request """
        raise NotImplementedError
