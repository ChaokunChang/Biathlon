import time
from typing import List, Tuple
import logging
import itertools

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec, QueryCostEstimation
from apxinfer.core.utils import merge_fvecs

from apxinfer.core.query import XIPQuery


logging.basicConfig(level=logging.INFO)


class XIPQEngine:
    def __init__(
        self, queries: List[XIPQuery], max_nthreads: int, verbose: bool = False
    ) -> None:
        self.queries = queries
        self.num_queries = len(queries)
        self.qnames = [qry.qname for qry in self.queries]
        assert len(self.qnames) == len(set(self.qnames)), "Query names must be unique"
        self.fnames = list(
            itertools.chain.from_iterable([qry.fnames for qry in self.queries])
        )
        assert len(self.fnames) == len(set(self.fnames)), "Feature names must be unique"

        self.max_nthreads = max_nthreads
        self.verbose = verbose

        self.logger = logging.getLogger("XIPQEngine")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def run(
        self, requets: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[QueryCostEstimation]]:
        qcosts = []
        fvecs = []
        for i in range(self.num_queries):
            st = time.time()
            fvecs.append(self.queries[i].run(requets, qcfgs[i]))
            et = time.time()
            qcost = et - st
            qcard = self.queries[i].estimate_cardinality(requets, qcfgs[i])
            qcosts.append(QueryCostEstimation(time=qcost, memory=None, qcard=qcard))

        return merge_fvecs(fvecs), qcosts

    def run_fs_only(
        self, requets: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> XIPFeatureVec:
        fvecs = []
        for i in range(self.num_queries):
            fvecs.append(self.queries[i].run(requets, qcfgs[i]))
        return merge_fvecs(fvecs)
