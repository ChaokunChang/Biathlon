import numpy as np
from typing import List
import logging
import copy

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
                 qinf_estimator: XIPQInfEstimator,
                 qcost_estimator: XIPQCostModel,
                 batch_size: int = 1,
                 verbose: bool = False) -> None:
        self.fextractor = fextractor
        self.model = model
        self.pred_estimator = pred_estimator
        self.qinf_estimator = qinf_estimator
        self.qcost_estimator = qcost_estimator
        self.batch_size = batch_size
        self.verbose = verbose

        self.history: List[XIPExecutionProfile] = []

        self.logger = logging.getLogger('XIPScheduler')
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def get_init_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Get the initial set of qcfgs to start the scheduler"""
        return [qry.cfg_pools[0] for qry in self.fextractor.queries]

    def get_final_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Get the final set of qcfgs to finish the scheduler"""
        return [qry.cfg_pools[-1] for qry in self.fextractor.queries]

    def start(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """ Start the scheduler given a request
        """
        self.history = []
        return self.get_init_qcfgs(request)

    def record(self, request: XIPRequest, qcfgs: List[XIPQueryConfig],
               fvec: XIPFeatureVec, pred: XIPPredEstimation,
               qcosts: List[QueryCostEstimation]) -> None:
        self.logger.debug(f'round-{len(self.history)}: qsamples = {[qcfg["qsample"] for qcfg in qcfgs]}')
        self.logger.debug(f'round-{len(self.history)}: qcosts   = {[qcost["time"] for qcost in qcosts]}')
        self.logger.debug(f'round-{len(self.history)}: pred={pred["pred_value"]}, error={pred["pred_error"]}, conf={pred["pred_conf"]}, {[qcost["time"] for qcost in qcosts]}')
        self.history.append(XIPExecutionProfile(request=request, qcfgs=qcfgs,
                                                fvec=fvec, pred=pred, qcosts=qcosts))

    def get_next_qcfgs(self, request: XIPRequest, qcfgs: List[XIPQueryConfig],
                       fvec: XIPFeatureVec, pred: XIPPredEstimation,
                       qcosts: List[QueryCostEstimation]) -> List[XIPQueryConfig]:
        qinf_est = self.qinf_estimator.estimate(self.model, self.fextractor, fvec, pred)
        qinfs = qinf_est['qinfs']
        sorted_qids = np.argsort(qinfs)[::-1]

        next_qcfgs = copy.deepcopy(qcfgs)
        if self.batch_size > 0:
            num = self.batch_size
        elif self.batch_size == 0:
            # adaptive batch size
            num = np.ceil(1.0 / pred['pred_conf'])
        else:
            raise ValueError(f'Invalid batch size {self.batch_size}')

        max_num = np.sum([len(self.fextractor.queries[i].cfg_pools) - 1 - qcfgs[i]["qcfg_id"] for i in range(len(qcfgs))])
        num = min(max_num, num)
        while (num > 0):
            for qid in sorted_qids:
                if num == 0:
                    break
                qcfg_id = qcfgs[qid]['qcfg_id']
                if qcfg_id == len(self.fextractor.queries[qid].cfg_pools) - 1:
                    continue
                next_qcfgs[qid] = self.fextractor.queries[qid].cfg_pools[qcfg_id + 1]
                num -= 1

        # if qcard is too small, just use final qcfgs
        for qid in range(len(qcfgs)):
            if qcosts[qid]['qcard'] is not None and qcosts[qid]['qcard'] < 30:
                next_qcfgs[qid] = self.fextractor.queries[qid].cfg_pools[-1]
        return next_qcfgs
