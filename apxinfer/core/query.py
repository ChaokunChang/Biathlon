from typing import List, TypedDict, Union, Callable
import logging
import numpy as np
import pandas as pd
import time
import asyncio
from aiohttp import ClientSession
from aiochclient import ChClient

from beaker.cache import CacheManager

from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.festimator import XIPFeatureEstimator

fcache_manager = CacheManager(
    cache_regions={"feature": {"type": "memory", "expire": 3600}}
)
logging.basicConfig(level=logging.INFO)


class XIPQuery:
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        fnames: List[str],
        enable_cache: bool = False,
    ) -> None:
        self.qname = qname
        self.qtype = qtype
        self.data_loader = data_loader
        self.fnames = fnames
        self.n_features = len(fnames)

        self.logger = logging.getLogger(f"XIPQuery-{qname}")

        # cache for query running
        self.enable_cache = enable_cache
        if self.enable_cache:
            self.run = fcache_manager.cache("query", expire=60)(self.run)

    def run(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> XIPFeatureVec:
        raise NotImplementedError

    def get_default_fvec(
        self, request: XIPRequest, qcfg: XIPQueryConfig
    ) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        fvals = np.zeros(len(self.fnames))
        if is_same_float(qsample, 1.0):
            self.logger.warning(f"no data for {request} in {self.qname}")
            fests = np.zeros(len(self.fnames))
        else:
            fests = np.ones(len(self.fnames)) * 1e9
        return XIPFeatureVec(
            fnames=self.fnames,
            fvals=fvals,
            fests=fests,
            fdists=["normal"] * len(self.fnames),
        )

    def get_fvec_with_default_est(self, fvals: np.ndarray) -> XIPFeatureVec:
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )

    def estimate_cost(self, request: XIPRequest, qcfg: XIPQueryConfig) -> float:
        """Estimate the cost of extracting features for a request"""
        raise NotImplementedError

    def estimate_inc_cost(self, request: XIPRequest, qcfg: XIPQueryConfig) -> float:
        """Estimate the incremental cost of extracting features for a request"""
        raise NotImplementedError

    def estimate_cardinality(self, request: XIPRequest, qcfg: XIPQueryConfig) -> int:
        """Get the number of data loaded for this query"""
        if self.data_loader is not None:
            return self.data_loader.estimate_cardinality(request, qcfg)
        return None

    def get_qcfg(self, cfg_id: int, sample: float, offset: float = 0.0):
        return XIPQueryConfig(
            qname=self.qname,
            qtype=self.qtype,
            qcfg_id=cfg_id,
            qoffset=max(offset, 0.0),
            qsample=min(sample, 1.0),
        )


class XIPQProfile(TypedDict, total=False):
    loading_nthreads: int
    loading_time: float
    computing_time: float
    rrd_nrows: int
    rrd_ncols: int


class XIPQOperatorDescription:
    dcol: str
    dops: List[Union[str, Callable]]


class XIPQueryProcessor:
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        self.qname = qname
        self.qtype = qtype
        self.fnames = fnames

        self.data_loader = data_loader
        if self.data_loader is not None:
            self.database = self.data_loader.database
            self.table = self.data_loader.table
            self.dbtable = f"{self.database}.{self.table}"
            self.tsize = self.data_loader.statistics["tsize"]
            self.nparts = self.data_loader.statistics["nparts"]
            self.dbclient = self.data_loader.db_client

        self.qops: List[XIPQOperatorDescription] = self.get_query_ops()
        if self.fnames is None:
            self.fnames = []
            for qop in self.qops:
                for opid, op in enumerate(qop["dops"]):
                    if isinstance(op, str):
                        fname = f"{self.qtype._name_}_{qop['dcol']}_{op}_{self.qname}"
                    else:
                        fname = f"{self.qtype._name_}_{qop['dcol']}_{opid}_{self.qname}"
                    self.fnames.append(fname)
        self.n_features = len(fnames)

        self.verbose = verbose
        self.logger = logging.getLogger(f"XIPQuery-{qname}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.set_enable_dcache()
        self.set_enable_qcache()
        self.set_enable_asyncio()
        self.set_estimator()

        self.profiles: List[XIPQProfile] = []

    def set_enable_dcache(self, enable_dcache: bool = True) -> None:
        self.enable_dcache = enable_dcache
        if self.enable_dcache:
            self._dcache = {"cached_req": None, "cached_nparts": 0, "cached_rrd": None}
            self.load_rrdata_agg = self.load_rrdata_agg_with_dcache
        else:
            self._dcache = None

    def set_enable_qcache(self, enable_qcache: bool = False) -> None:
        self.enable_qcache = enable_qcache

    def set_enable_asyncio(self, enable_ayncio: bool = False) -> None:
        self.enable_async_io = enable_ayncio

    def set_estimator(self, festimator: XIPFeatureEstimator = None) -> None:
        if festimator is not None:
            self.festimator = festimator
        else:
            self.festimator = XIPFeatureEstimator(err_module=None)

    def get_query_condition(self) -> str:
        # return where clause without pid constraint
        # should be overwritten by user
        return None

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        pass

    def run(
        self,
        request: XIPRequest,
        qcfg: XIPQueryConfig,
        loading_nthreads: int = 1,
        computing_nthreads: int = 1,
    ) -> XIPFeatureVec:
        st = time.time()
        rrdata = self.load_rrdata(request, qcfg, loading_nthreads)
        loading_time = time.time() - st

        st = time.time()
        if rrdata is None:
            fvec = self.get_default_fvec(request, qcfg)
        else:
            fvec = self.compute_features(rrdata, computing_nthreads)
        computing_time = time.time() - st

        self.profiles.append(
            XIPQProfile(
                loading_nthreads=loading_nthreads,
                loading_time=loading_time,
                computing_time=computing_time,
                rrd_nrows=rrdata.shape[0],
                rrd_ncols=rrdata.shape[1],
            )
        )
        return fvec

    def load_rrdata(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        if self.qtype == XIPQType.AGG:
            return self.load_rrdata_agg(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.FSTORE, XIPQType.KeySearch]:
            return self.load_rrdata_ks(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.NORMAL, XIPQType.TRANSFORM]:
            return self.load_rrdata_tr(request, qcfg, loading_nthreads)
        else:
            raise NotImplementedError(f"{self.qtype} not supported yet.")

    def load_rrdata_agg(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        pass

    def load_rrdata_agg_directly(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        qcond = self.get_query_condition(request)
        from_pid = round(self.nparts * qcfg.get("qoffset", 0))
        to_pid = round(self.nparts * qcfg.get("qsample", 1))
        sql = f"""
                SELECT {', '.join([qop['dcol'] for qop in self.qops])}
                FROM {self.dbtable}
                WHERE {qcond} AND pid >= {from_pid} AND pid < {to_pid}
                SETTINGS max_threads = {loading_nthreads}
            """
        return self.execute_sql(sql)

    def load_rrdata_agg_with_dcache(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        qcond = self.get_query_condition(request)

        from_pid = round(self.nparts * qcfg.get("qoffset", 0))
        to_pid = round(self.nparts * qcfg.get("qsample", 1))
        assert from_pid == 0, "currently we do not support qoffset > 0"

        req_id = request["req_id"]
        if self._dcache.get("cached_req", None) == req_id:
            cached_nparts = self._dcache.get("cached_nparts", 0)
            assert cached_nparts <= to_pid
            from_pid = cached_nparts
        else:
            self._dcache["cached_req"] = req_id
            self._dcache["cached_nparts"] = 0
            self._dcache["cached_rrd"] = None

        if from_pid < to_pid:
            sql = f"""
                    SELECT {', '.join([qop['dcol'] for qop in self.qops])}
                    FROM {self.dbtable}
                    WHERE {qcond} AND pid >= {from_pid} AND pid < {to_pid}
                    SETTINGS max_threads = {loading_nthreads}
                """
            new_rrd = self.execute_sql(sql)

        if self._dcache["cached_rrd"] is None:
            self._dcache["cached_rrd"] = new_rrd
        else:
            if new_rrd is not None:
                self._dcache["cached_rrd"] = np.concatenate(
                    self._dcache["cached_rrd"], new_rrd, axis=0
                )
        return self._dcache["cached_rrd"]

    def load_rrdata_ks(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        qcond = self.get_query_condition(request)
        sql = f"""
                SELECT {', '.join([qop['dcol'] for qop in self.qops])}
                FROM {self.dbtable}
                WHERE {qcond}
                SETTINGS max_threads = {loading_nthreads}
            """
        return self.execute_sql(sql)

    def load_rrdata_tr(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        return np.array([[request[key] for key in self.dcols]])

    def execute_sql(self, sql: str) -> np.ndarray:
        df: pd.DataFrame = self.dbclient.query_df(sql)
        if df.empty:
            return None
        else:
            return df.values

    async def execute_sqls_async(self, sqls: List[str]):
        async with ClientSession() as s:
            client = ChClient(s, compress_response=False)
            results = await asyncio.gather(*[client.fetch(sql) for sql in sqls])
        return np.array([list(row.values()) for result in results for row in result])

    def compute_features(self, rrdata: np.ndarray, qsample: float) -> XIPFeatureVec:
        tsize = self.tsize
        fvecs = []
        for i in range(len(self.qops)):
            dcol, dops = self.qops[i]
            for op in dops:
                if isinstance(op, str):
                    fvec = self.festimator.extract(
                        rrdata[:, i : i + 1], qsample, tsize, op
                    )
                else:
                    fvals = np.array([op(rrdata)])
                    fvec = self.get_fvec_with_default_est(fvals)
                fvecs.append(fvec)
        return merge_fvecs(fvecs, new_names=self.fnames)

    def get_default_fvec(
        self, request: XIPRequest, qcfg: XIPQueryConfig
    ) -> XIPFeatureVec:
        fvals = np.zeros(len(self.fnames))
        if is_same_float(qcfg["qsample"], 1.0):
            self.logger.warning(f"no data for {request} in {self.qname}")
            fests = np.zeros(len(self.fnames))
        else:
            fests = np.ones(len(self.fnames)) * 1e9
        return XIPFeatureVec(
            fnames=self.fnames,
            fvals=fvals,
            fests=fests,
            fdists=["normal"] * len(self.fnames),
        )

    def get_fvec_with_default_est(self, fvals: np.array) -> XIPFeatureVec:
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        fvec = XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )
        return fvec
