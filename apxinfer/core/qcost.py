import numpy as np
from typing import List, Dict
import logging
from sklearn.base import BaseEstimator

from apxinfer.core.utils import XIPRequest, XIPQueryConfig, QueryCostEstimation

logging.basicConfig(level=logging.INFO)


class XIPQCostModel(BaseEstimator):
    """ Cost model for XIP, estimate execution cost of a query given the request and cfg
        This base version estimate according to cfg only.
    """
    def __init__(self) -> None:
        self.logger = logging.getLogger('XIPQCostModel')

    def fit(self, requests: List[XIPRequest], qcfgs: List[XIPQueryConfig], qcosts: List[QueryCostEstimation]) -> None:
        """ Fit the cost model given the model and feature extractors
        """
        mapping = Dict[XIPQueryConfig, List[QueryCostEstimation]]
        for i in range(len(requests)):
            if qcfgs[i] not in mapping:
                mapping[qcfgs[i]] = []
            mapping[qcfgs[i]].append(qcosts[i])

        self.mapping = {}
        for cfg, costs in mapping.items():
            avg_time = np.mean([cost['time'] for cost in costs])
            avg_memory = np.mean([cost['memory'] for cost in costs])
            self.mapping[cfg] = QueryCostEstimation(avg_time, avg_memory)

    def estimate(self, request: XIPRequest, qcfg: XIPQueryConfig) -> QueryCostEstimation:
        return self.mapping[qcfg]


class QueryCostModel(XIPQCostModel):
    """ TODO: Final Version of Cost Model for XIP
        fit a simple model to estimate the cost of a query given the request and cfg
    """
    def __init__(self) -> None:
        pass

    def fit(self, requests: List[XIPRequest], qcfgs: List[XIPQueryConfig], qcosts: List[QueryCostEstimation]) -> None:
        raise NotImplementedError

    def estimate(self, request: XIPRequest, qcfg: XIPQueryConfig) -> QueryCostEstimation:
        raise NotImplementedError
