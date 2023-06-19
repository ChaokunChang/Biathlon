import numpy as np
from typing import List
import logging

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec, XIPPredEstimation
from apxinfer.core.utils import QueryCostEstimation, XIPExecutionProfile
from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import XIPPredictionEstimator
from apxinfer.core.qcost import XIPQCostModel
from apxinfer.core.qinfluence import XIPQInfEstimator

logging.basicConfig(level=logging.INFO)


class XIPScheduler:
    """ Base class for XIP Scheduler
    """
    def __init__(self, fextractor: XIPFeatureExtractor,
                 model: XIPModel,
                 pred_estimator: XIPPredictionEstimator,
                 qcost_estimator: XIPQCostModel,
                 qinf_estimator: XIPQInfEstimator) -> None:
        self.fextractor = fextractor
        self.model = model
        self.pred_estimator = pred_estimator
        self.qcost_estimator = qcost_estimator
        self.qinf_estimator = qinf_estimator
        self.history: List[XIPExecutionProfile] = []
        self.logger = logging.getLogger('XIPScheduler')

    def get_init_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Get the initial set of qcfgs to start the scheduler"""
        return [qry.cfg_pools[0] for qry in self.fextractor.queries]

    def get_final_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Get the final set of qcfgs to finish the scheduler"""
        return [qry.cfg_pools[-1] for qry in self.fextractor]

    def start(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Start the scheduler given a request
        """
        self.history = []
        return self.get_init_qcfgs(request)

    def get_next_qcfgs(self, request: XIPRequest, qcfgs: List[XIPQueryConfig],
                       fvec: XIPFeatureVec, pred: XIPPredEstimation,
                       qcosts: List[QueryCostEstimation]) -> List[XIPQueryConfig]:
        self.history.append(XIPExecutionProfile(request=request, qcfgs=qcfgs,
                                                fvec=fvec, pred=pred, qcosts=qcosts))
        qinf_est = self.qinf_estimator.estimate(self.model, self.fextractor, fvec, pred)
        qinfs = qinf_est['qinfs']
        sorted_qids = np.argsort(qinfs)[::-1]

        next_qcfgs = []
        for qid in sorted_qids:
            qcfg_id = qcfgs[qid]['qcfg_id']
            if qcfg_id == len(self.fextractor.queries[qid].cfg_pools) - 1:
                continue
            next_qcfgs.append(self.fextractor.queries[qid].cfg_pools[qcfg_id + 1])
        return next_qcfgs
