import numpy as np
from typing import List
import logging
import copy

from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec, XIPPredEstimation
from apxinfer.core.utils import QueryCostEstimation, XIPExecutionProfile
from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import XIPPredictionEstimator
from apxinfer.core.qcost import XIPQCostModel
from apxinfer.core.qinfluence import XIPQInfEstimator

logging.basicConfig(level=logging.INFO)


class XIPScheduler:
    """Base class for XIP Scheduler"""

    def __init__(
        self,
        fextractor: XIPFeatureExtractor,
        model: XIPModel,
        pred_estimator: XIPPredictionEstimator,
        qinf_estimator: XIPQInfEstimator,
        qcost_estimator: XIPQCostModel,
        sample_grans: List[float] = None,
        min_qsamples: List[float] = None,
        max_qsamples: List[float] = None,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> None:
        self.fextractor = fextractor
        self.model = model
        self.pred_estimator = pred_estimator
        self.qinf_estimator = qinf_estimator
        self.qcost_estimator = qcost_estimator
        self.sample_grans = sample_grans
        self.min_qsamples = min_qsamples
        self.max_qsamples = max_qsamples
        self.batch_size = batch_size
        self.verbose = verbose

        self.history: List[XIPExecutionProfile] = []

        if self.sample_grans is None:
            self.sample_grans = [
                0.1 if qry.qtype == XIPQType.AGG else 1.0
                for qry in self.fextractor.queries
            ]

        if self.min_qsamples is None:
            self.min_qsamples = [
                qry.get_qcfg(
                    0, self.sample_grans[i] if qry.qtype == XIPQType.AGG else 1.0, 0.0
                )
                for i, qry in enumerate(self.fextractor.queries)
            ]

        if self.max_qsamples is None:
            return [qry.get_qcfg(100, 1.0, 0.0) for qry in self.fextractor.queries]

        self.logger = logging.getLogger("XIPScheduler")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def get_init_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """Get the initial set of qcfgs to start the scheduler"""
        return [
            qry.get_qcfg(0, qsample, 0.0)
            for qry, qsample in zip(self.fextractor.queries, self.min_qsamples)
        ]

    def get_final_qcfgs(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """Get the final set of qcfgs to finish the scheduler"""
        return [
            qry.get_qcfg(0, qsample, 0.0)
            for qry, qsample in zip(self.fextractor.queries, self.max_qsamples)
        ]

    def start(self, request: XIPRequest) -> List[XIPQueryConfig]:
        """Start the scheduler given a request"""
        self.history = []
        return self.get_init_qcfgs(request)

    def record(
        self,
        request: XIPRequest,
        qcfgs: List[XIPQueryConfig],
        fvec: XIPFeatureVec,
        pred: XIPPredEstimation,
        qcosts: List[QueryCostEstimation],
    ) -> None:
        self.logger.debug(
            f'round-{len(self.history)}: qsmpls = {[qcfg["qsample"] for qcfg in qcfgs]}'
        )
        self.logger.debug(
            f'round-{len(self.history)}: qcosts = {[qcost["time"] for qcost in qcosts]}'
        )
        self.logger.debug(
            f'round-{len(self.history)}: pred={pred["pred_value"]},'
            f'error={pred["pred_error"]}, conf={pred["pred_conf"]},'
            f'{[qcost["time"] for qcost in qcosts]}'
        )
        self.history.append(
            XIPExecutionProfile(
                request=request, qcfgs=qcfgs, fvec=fvec, pred=pred, qcosts=qcosts
            )
        )

    def get_next_qcfgs(
        self,
        request: XIPRequest,
        qcfgs: List[XIPQueryConfig],
        fvec: XIPFeatureVec,
        pred: XIPPredEstimation,
        qcosts: List[QueryCostEstimation],
    ) -> List[XIPQueryConfig]:
        qinf_est = self.qinf_estimator.estimate(self.model, self.fextractor, fvec, pred)
        qinfs = qinf_est["qinfs"]
        sorted_qids = np.argsort(qinfs)[::-1]

        next_qcfgs = copy.deepcopy(qcfgs)
        if self.batch_size > 0:
            nsteps = self.batch_size
        elif self.batch_size == 0:
            # adaptive batch size
            nsteps = np.ceil(1.0 / pred["pred_conf"])
        else:
            raise ValueError(f"Invalid batch size {self.batch_size}")
        valid_nsteps = [
            (self.max_qsamples[qid] - qcfgs[qid]["qsample"]) / self.sample_grans[qid]
            for qid in range(len(qcfgs))
        ]
        nsteps = min(np.sum(valid_nsteps), nsteps)
        while nsteps > 0:
            for qid in sorted_qids:
                if nsteps == 0:
                    break
                if qcfgs[qid]["qsample"] == self.max_qsamples[qid]:
                    continue
                next_qcfgs[qid] = qcfgs[qid]["qsample"] + self.sample_grans[qid]
                nsteps -= 1

        # if qcard is too small, just use final qcfgs
        for qid in range(len(qcfgs)):
            if qcosts[qid]["qcard"] is not None and qcosts[qid]["qcard"] < 30:
                next_qcfgs[qid] = self.max_qsamples[qid]
        return next_qcfgs
