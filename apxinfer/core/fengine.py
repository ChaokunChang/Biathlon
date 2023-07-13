import time
from typing import List, Tuple
import logging
import itertools

from multiprocessing import Pool
from functools import partial

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec, QueryCostEstimation
from apxinfer.core.utils import merge_fvecs

from apxinfer.core.query import XIPQuery


logging.basicConfig(level=logging.INFO)


class XIPFEngine:
    def __init__(
        self,
        queries: List[XIPQuery],
        loading_nthreads: int,
        nprocs: int = 1,
        verbose: bool = False,
    ) -> None:
        self.queries = queries
        self.num_queries = len(queries)
        self.qnames = [qry.qname for qry in self.queries]
        assert len(self.qnames) == len(set(self.qnames)), "Query names must be unique"
        self.fnames = list(
            itertools.chain.from_iterable([qry.fnames for qry in self.queries])
        )
        assert len(self.fnames) == len(set(self.fnames)), "Feature names must be unique"

        self.loading_nthreads = loading_nthreads
        self.nprocs = nprocs
        self.verbose = verbose

        if self.nprocs > 1:
            self.mp_pool = Pool(self.nprocs)

        self.logger = logging.getLogger("XIPQEngine")
        if verbose:
            self.logger.setLevel(logging.DEBUG)

    def extract(
        self, request: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[QueryCostEstimation]]:
        qcosts = []
        fvecs = []
        for i in range(self.num_queries):
            st = time.time()
            fvecs.append(self.queries[i].run(request, qcfgs[i], self.loading_nthreads))
            et = time.time()
            qcost = et - st
            qcard = self.queries[i].estimate_cardinality(request, qcfgs[i])
            qcosts.append(QueryCostEstimation(time=qcost, memory=None, qcard=qcard))

        return merge_fvecs(fvecs), qcosts

    def extract_worker(
        request: XIPRequest, ldnthreads: int, qry: XIPQuery, qcfg: XIPQueryConfig
    ) -> Tuple[XIPFeatureVec, QueryCostEstimation]:
        st = time.time()
        fvec = qry.run(request, qcfg, ldnthreads)
        qcard = qry.estimate_cardinality(request, qcfg)
        qcost = time.time() - st
        return fvec, QueryCostEstimation(time=qcost, memory=None, qcard=qcard)

    def extract_parallel(
        self, request: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[QueryCostEstimation]]:
        worker = partial(XIPFEngine.extract_worker, request, self.loading_nthreads)
        results = self.mp_pool.map(
            worker, [(self.queries[i], qcfgs[i]) for i in range(self.num_queries)]
        )
        self.mp_pool.join()

        fvecs = []
        qcosts = []
        for res in results:
            fvecs.append(res[0])
            qcosts.append(res[1])

        return merge_fvecs(fvecs), qcosts

    def extract_fs_only(
        self, request: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> XIPFeatureVec:
        fvecs = []
        for i in range(self.num_queries):
            fvecs.append(self.queries[i].run(request, qcfgs[i], self.loading_nthreads))
        return merge_fvecs(fvecs)
