from typing import List, TypedDict, Union, Callable
import logging
import numpy as np
import pandas as pd
import time
import json
from tap import Tap
import asyncio
from aiohttp import ClientSession
from aiochclient import ChClient

from beaker.cache import CacheManager

from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.festimator import XIPFeatureErrorEstimator, XIPFeatureEstimator

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
    card_est: int


class XIPQOperatorDescription(TypedDict):
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
        self.n_features = len(self.fnames)

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

    def get_query_condition(self, request: XIPRequest) -> str:
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
        self.logger.debug(f"{self.qname} running with {qcfg}")
        st = time.time()
        rrdata = self.load_rrdata(request, qcfg, loading_nthreads)
        loading_time = time.time() - st

        card_est = self.estimate_cardinality(rrdata, qcfg)

        st = time.time()
        if rrdata is None:
            fvec = self.get_default_fvec(request, qcfg)
        else:
            fvec = self.compute_features(rrdata, qcfg, computing_nthreads)
        computing_time = time.time() - st

        self.profiles.append(
            XIPQProfile(
                loading_nthreads=loading_nthreads,
                loading_time=loading_time,
                computing_time=computing_time,
                rrd_nrows=rrdata.shape[0] if rrdata is not None else 0,
                rrd_ncols=rrdata.shape[1] if rrdata is not None else 0,
                card_est=card_est,
            )
        )
        self.logger.debug(f"profile: {json.dumps(self.profiles[-1], indent=4)}")
        return fvec

    def load_rrdata(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        if self.qtype == XIPQType.AGG:
            rrdata = self.load_rrdata_agg(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.FSTORE, XIPQType.KeySearch]:
            rrdata = self.load_rrdata_ks(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.NORMAL, XIPQType.TRANSFORM]:
            rrdata = self.load_rrdata_tr(request, qcfg, loading_nthreads)
        else:
            raise NotImplementedError(f"{self.qtype} not supported yet.")
        if rrdata is None and is_same_float(qcfg["qsample"], 1.0):
            self.logger.warning(f"no rrdata for {request} in {self.qname}")
        return rrdata

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
                    [self._dcache["cached_rrd"], new_rrd], axis=0
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
        return np.array([[request[qop["dcol"]] for qop in self.qops]])

    def execute_sql(self, sql: str) -> np.ndarray:
        df: pd.DataFrame = self.dbclient.query_df(sql)
        if df.empty:
            self.logger.debug(f"{self.qname} sql return None")
            return None
        else:
            rrdata = df.values
            self.logger.debug(f"{self.qname} sql return {rrdata.dtype} {rrdata.shape}")
            return rrdata

    def compute_features(
        self, rrdata: np.ndarray, qcfg: XIPQueryConfig, computing_nthreads: int = 1
    ) -> XIPFeatureVec:
        qsample: float = qcfg["qsample"]
        tsize = self.tsize
        fvecs = []
        for i in range(len(self.qops)):
            qop: XIPQOperatorDescription = self.qops[i]
            for op in qop["dops"]:
                if isinstance(op, str):
                    fvec = self.festimator.extract(
                        rrdata[:, i : i + 1], qsample, tsize, op
                    )
                else:
                    fvals = np.array([op(rrdata)])
                    fvec = self.get_fvec_with_default_est(fvals)
                fvecs.append(fvec)
        return merge_fvecs(fvecs, new_names=self.fnames)

    def estimate_cardinality(self, rrdata: np.ndarray, qcfg: XIPQueryConfig) -> int:
        qsample = qcfg["qsample"]
        # card = self.festimator.extract(rrdata[:, 0], qsample, self.tsize, "count")
        # return card['fvals'][0]
        if rrdata is not None and qsample > 0:
            card = round(rrdata.shape[0] / qsample)
        else:
            card = None
        return card

    def get_latest_card_est(self) -> int:
        return self.profiles[-1]["card_est"]

    def get_default_fvec(
        self, request: XIPRequest, qcfg: XIPQueryConfig
    ) -> XIPFeatureVec:
        fvals = np.zeros(len(self.fnames))
        if is_same_float(qcfg["qsample"], 1.0):
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
        fdists = ["fixed"] * len(fvals)
        fvec = XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )
        return fvec

    def get_qcfg(self, cfg_id: int, sample: float, offset: float = 0.0):
        return XIPQueryConfig(
            qname=self.qname,
            qtype=self.qtype,
            qcfg_id=cfg_id,
            qoffset=max(offset, 0.0),
            qsample=min(sample, 1.0),
        )

    def get_dcol_embeddings(self, dcol: str) -> dict:
        if self.embeddings is None:
            self.embeddings = {}
        if self.embeddings.get(dcol, None) is None:
            sql = f"SELECT DISTINCT {dcol} FROM {self.dbtable} ORDER BY {dcol}"
            df: pd.DataFrame = self.dbclient.query_df(sql)
            embedding = {value: i + 1 for i, value in enumerate(df[dcol].values)}
            self.embeddings[dcol] = embedding
        return self.embeddings.get(dcol)

    async def run_async(
        self,
        request: XIPRequest,
        qcfg: XIPQueryConfig,
        loading_nthreads: int = 1,
        computing_nthreads: int = 1,
    ) -> XIPFeatureVec:
        self.logger.debug(f"{self.qname} running async with {qcfg}")
        self.async_db_client = ChClient(ClientSession(), compress_response=True)
        st = time.time()
        rrdata = await self.load_rrdata_async(request, qcfg, loading_nthreads)
        loading_time = time.time() - st

        card_est = self.estimate_cardinality(rrdata, qcfg)

        st = time.time()
        if rrdata is None:
            fvec = self.get_default_fvec(request, qcfg)
        else:
            fvec = self.compute_features(rrdata, qcfg, computing_nthreads)
        computing_time = time.time() - st

        self.profiles.append(
            XIPQProfile(
                loading_nthreads=loading_nthreads,
                loading_time=loading_time,
                computing_time=computing_time,
                rrd_nrows=rrdata.shape[0],
                rrd_ncols=rrdata.shape[1],
                card_est=card_est,
            )
        )
        self.logger.debug(f"profile: {json.dumps(self.profiles[-1], indent=4)}")
        await self.async_db_client.close()
        return fvec

    async def load_rrdata_async(
        self, request: XIPRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1
    ) -> np.ndarray:
        if self.qtype == XIPQType.AGG:
            rrdata = await self.load_rrdata_agg_async(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.FSTORE, XIPQType.KeySearch]:
            rrdata = self.load_rrdata_ks(request, qcfg, loading_nthreads)
        elif self.qtype in [XIPQType.NORMAL, XIPQType.TRANSFORM]:
            rrdata = self.load_rrdata_tr(request, qcfg, loading_nthreads)
        else:
            raise NotImplementedError(f"{self.qtype} not supported yet.")
        if rrdata is None and is_same_float(qcfg["qsample"], 1.0):
            self.logger.warning(f"no rrdata for {request} in {self.qname}")
        return rrdata

    async def load_rrdata_agg_async(
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
            new_rrd = await self.execute_sql_async(sql)

        if self._dcache["cached_rrd"] is None:
            self._dcache["cached_rrd"] = new_rrd
        else:
            if new_rrd is not None:
                self._dcache["cached_rrd"] = np.concatenate(
                    [self._dcache["cached_rrd"], new_rrd], axis=0
                )
        return self._dcache["cached_rrd"]

    async def execute_sql_async(self, sql: str) -> np.ndarray:
        rows = await self.async_db_client.fetch(sql, decode=True)
        nrows = len(rows)
        if nrows == 0:
            return None
        else:
            rrdata = [[d for d in row.values()] for row in rows]
            rrdata = np.array(rrdata, ndmin=2).reshape((nrows, -1))
            self.logger.debug(f"{self.qname} sql return {rrdata.dtype} {rrdata.shape}")
            return rrdata


class QueryTestArgs(Tap):
    verbose: bool = False
    qsample: float = 0.1
    ld_nthreads: int = 0
    cp_nthreads: int = 0


if __name__ == "__main__":
    import datetime as dt
    from apxinfer.examples.taxi.data import TaxiTripRequest

    args = QueryTestArgs().parse_args()

    class ExampleQP0(XIPQueryProcessor):
        def __init__(
            self,
            qname: str,
            qtype: XIPQType,
            data_loader: XIPDataLoader,
            fnames: List[str] = None,
            verbose: bool = False,
        ) -> None:
            super().__init__(qname, qtype, data_loader, fnames, verbose)
            self.embeddings = {}
            for dcol in ["pickup_ntaname", "dropoff_ntaname"]:
                self.get_dcol_embeddings(dcol)

        def get_query_ops(self) -> List[XIPQOperatorDescription]:
            dcols = [
                "req_trip_distance",
                "req_pickup_datetime",
                "req_passenger_count",
                "req_pickup_ntaname",
                "req_dropoff_ntaname",
            ]
            dcol_aggs = [
                [lambda x: float(x[0][0])],
                [
                    lambda x: pd.to_datetime(x[0][1]).weekday(),
                    lambda x: pd.to_datetime(x[0][1]).hour,
                ],
                [lambda x: int(x[0][2])],
                [lambda x: self.get_dcol_embeddings("pickup_ntaname")[x[0][3]]],
                [lambda x: self.get_dcol_embeddings("dropoff_ntaname")[x[0][4]]],
            ]
            qops = [
                XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
                for i, dcol in enumerate(dcols)
            ]
            return qops

    class ExampleQP1(XIPQueryProcessor):
        def get_query_condition(self, request: TaxiTripRequest) -> str:
            to_dt = pd.to_datetime(request["req_pickup_datetime"])
            from_dt = to_dt - dt.timedelta(hours=1)
            pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
            and_list = [
                f"pickup_datetime >= '{from_dt}'",
                f"pickup_datetime < '{to_dt}'",
                f"pickup_ntaname = '{pickup_ntaname}'",
                "dropoff_datetime IS NOT NULL",
                f"dropoff_datetime <= '{to_dt}'",
            ]
            qcond = " AND ".join(and_list)
            return qcond

        def get_query_ops(self) -> List[XIPQOperatorDescription]:
            dcols = ["trip_duration", "total_amount", "fare_amount"]
            dcol_aggs = [["sum"], ["sum"], ["stdPop"]]
            qops = [
                XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
                for i, dcol in enumerate(dcols)
            ]
            return qops

    class ExampleQP2(XIPQueryProcessor):
        def get_query_condition(self, request: TaxiTripRequest) -> str:
            to_dt = pd.to_datetime(request["req_pickup_datetime"])
            from_dt = to_dt - dt.timedelta(hours=24)
            pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
            dropoff_ntaname = request["req_dropoff_ntaname"].replace("'", r"\'")
            and_list = [
                f"pickup_datetime >= '{from_dt}'",
                f"pickup_datetime < '{to_dt}'",
                f"pickup_ntaname = '{pickup_ntaname}'",
                f"dropoff_ntaname = '{dropoff_ntaname}'",
                "dropoff_datetime IS NOT NULL",
                f"dropoff_datetime <= '{to_dt}'",
            ]
            qcond = " AND ".join(and_list)
            return qcond

        def get_query_ops(self) -> List[XIPQOperatorDescription]:
            dcols = ["trip_distance", "trip_duration", "tip_amount"]
            dcol_aggs = [["sum"], ["max"], ["max", "median"]]
            qops = [
                XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
                for i, dcol in enumerate(dcols)
            ]
            return qops

    class ExampleQP3(XIPQueryProcessor):
        def get_query_condition(self, request: TaxiTripRequest) -> str:
            to_dt = pd.to_datetime(request["req_pickup_datetime"])
            from_dt = to_dt - dt.timedelta(hours=24 * 7)
            pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
            dropoff_ntaname = request["req_dropoff_ntaname"].replace("'", r"\'")
            passenger_count = request["req_passenger_count"]
            and_list = [
                f"pickup_datetime >= '{from_dt}'",
                f"pickup_datetime < '{to_dt}'",
                f"pickup_ntaname = '{pickup_ntaname}'",
                f"dropoff_ntaname = '{dropoff_ntaname}'",
                f"passenger_count = '{passenger_count}'",
            ]
            qcond = " AND ".join(and_list)
            return qcond

        def get_query_ops(self) -> List[XIPQOperatorDescription]:
            dcols = ["trip_distance"]
            dcol_aggs = [["max"]]
            qops = [
                XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
                for i, dcol in enumerate(dcols)
            ]
            return qops

    class ExampleQP4(XIPQueryProcessor):
        def get_query_condition(self, request: TaxiTripRequest) -> str:
            to_dt = pd.to_datetime(request["req_pickup_datetime"])
            from_dt = to_dt - dt.timedelta(hours=8)
            # pickup_ntaname = request["req_pickup_ntaname"].replace("'", r"\'")
            passenger_count = request["req_passenger_count"]
            and_list = [
                f"pickup_datetime >= '{from_dt}'",
                f"pickup_datetime < '{to_dt}'",
                # f"pickup_ntaname = '{pickup_ntaname}'",
                f"passenger_count = '{passenger_count}'",
            ]
            qcond = " AND ".join(and_list)
            return qcond

        def get_query_ops(self) -> List[XIPQOperatorDescription]:
            # dcols = ["dropoff_ntaname"]
            dcols = ["pickup_ntaname", "dropoff_ntaname"]
            dcol_aggs = [["unique"], ["unique"]]
            qops = [
                XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
                for i, dcol in enumerate(dcols)
            ]
            return qops

    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table="trips",
        seed=0,
        enable_cache=False,
    )
    if args.verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    def get_qps() -> List[XIPQueryProcessor]:
        qp0 = ExampleQP0(
            qname="q0",
            qtype=XIPQType.NORMAL,
            data_loader=data_loader,
            fnames=None,
            verbose=args.verbose,
        )
        qp1 = ExampleQP1(
            qname="q1",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            fnames=None,
            verbose=args.verbose,
        )
        qp2 = ExampleQP2(
            qname="q2",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            fnames=None,
            verbose=args.verbose,
        )
        qp3 = ExampleQP3(
            qname="q3",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            fnames=None,
            verbose=args.verbose,
        )
        qp4 = ExampleQP4(
            qname="q4",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            fnames=None,
            verbose=args.verbose,
        )
        qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]
        # qps.append(qp4)
        ferr_est = XIPFeatureErrorEstimator(bs_nresamples=100)
        festimator = XIPFeatureEstimator(ferr_est)
        for qp in qps:
            qp.set_estimator(festimator)
        return qps

    request = {
        "req_id": 800,
        "req_trip_id": 1204066502,
        "req_pickup_datetime": "2015-08-02 11:00:04",
        "req_pickup_ntaname": "Turtle Bay-East Midtown",
        "req_dropoff_ntaname": "Lenox Hill-Roosevelt Island",
        "req_pickup_longitude": -73.96684265136719,
        "req_pickup_latitude": 40.76113128662109,
        "req_dropoff_longitude": -73.956787109375,
        "req_dropoff_latitude": 40.766700744628906,
        "req_passenger_count": 1,
        "req_trip_distance": 0.73,
    }

    qsample = args.qsample
    qcfgs = [qp.get_qcfg(0, qsample) for qp in get_qps()]
    qcfgs[0]["qsample"] = 1.0

    def run_sequential(
        ld_nthreads: int = 1, cp_nthrads: int = 1
    ) -> Union[XIPFeatureVec, float]:
        # run qps sequentially one-by-one
        st = time.time()
        qps = get_qps()
        prepare_time = time.time() - st
        st = time.time()
        fvecs = []
        for qp, qcfg in zip(qps, qcfgs):
            fvec = qp.run(
                request,
                qcfg,
                loading_nthreads=ld_nthreads,
                computing_nthreads=cp_nthrads,
            )
            fvecs.append(fvec)
        fvec = merge_fvecs(fvecs)
        tcost = time.time() - st

        print(f"prepare time: {prepare_time}")
        for qp in qps:
            print(f"qprofile-{qp.qname}: {qp.profiles[-1]}")
        print(f"run sequential with threads {ld_nthreads} + {cp_nthrads}: {tcost}")
        return fvec, tcost

    async def run_async(
        ld_nthreads: int = 1, cp_nthrads: int = 1
    ) -> Union[XIPFeatureVec, float]:
        # run qps asynchrously with asyncio
        st = time.time()
        qps = get_qps()
        prepare_time = time.time() - st
        st = time.time()
        fvecs = await asyncio.gather(
            *[
                qp.run_async(
                    request,
                    qcfg,
                    loading_nthreads=ld_nthreads,
                    computing_nthreads=cp_nthrads,
                )
                for qp, qcfg in zip(qps, qcfgs)
            ]
        )
        fvec = merge_fvecs(fvecs)
        tcost = time.time() - st

        print(f"prepare time: {prepare_time}")
        for qp in qps:
            print(f"qprofile-{qp.qname}: {qp.profiles[-1]}")

        print(f"run asynchronously with threads {ld_nthreads} + {cp_nthrads}: {tcost}")
        return fvec, tcost

    fvec, tcost = run_sequential(args.ld_nthreads, args.cp_nthreads)
    st = time.time()
    fvec_async, tcost_async = asyncio.run(run_async(args.ld_nthreads, args.cp_nthreads))
    print(f"end2end time: {time.time() - st}")
    print(f"sync v.s. async = {tcost} : {tcost_async} = {tcost / tcost_async}")

    # # iterative async run, 10 parts per iter
    # # only better than sequential OIP when qsample < 0.3
    # # only better than asynio OIP when qsample < 0.2
    # tcost_async_iter = 0.0
    # for i in range(0, int(100 * qsample), 10):
    #     for qid in range(len(qcfgs)):
    #         if qid > 0:
    #             qcfgs[qid]["qoffset"] = (i + 0.0) / 100
    #             qcfgs[qid]["qsample"] = (i + 10.0) / 100
    #     fvec_async_iter, itercost = asyncio.run(
    #         run_async(args.ld_nthreads, args.cp_nthreads)
    #     )
    #     tcost_async_iter += itercost
    # print(
    #     f"sync v.s. async-iter = {tcost} : {tcost_async_iter} = {tcost / tcost_async_iter}"
    # )

    fval_comp = np.abs(fvec_async["fvals"] - fvec["fvals"])
    fest_comp = np.abs(fvec_async["fests"] - fvec["fests"])
    fval_diff = np.where(
        fval_comp > 0,
        fval_comp
        / np.maximum(
            np.maximum(np.abs(fvec["fvals"]), np.abs(fvec_async["fvals"])), 1e-6
        ),
        0,
    )
    fest_diff = np.where(
        fest_comp > 0,
        fest_comp
        / np.maximum(
            np.maximum(np.abs(fvec["fests"]), np.abs(fvec_async["fests"])), 1e-3
        ),
        0,
    )
    assert np.all(fval_diff < 1e-6), f"fval_diff={fval_diff}, fval={fvec['fvals']}"
    # assert np.all(
    #     np.minimum(fest_diff, fest_comp) < 1e-3
    # ), f"fest_diff={fest_diff}, fest_comp={fest_comp}"

    # print(
    #     json.dumps(
    #         {
    #             "fnames": fvec_async["fnames"],
    #             "fvals": fvec_async["fvals"].tolist(),
    #             "fests": fvec_async["fests"].tolist(),
    #             "fdists": fvec_async["fdists"],
    #             # "fnames": ", ".join(fvec["fnames"]),
    #             # "fvals": ", ".join([str(val) for val in fvec["fvals"].tolist()]),
    #             # "fests": ", ".join([str(val) for val in fvec["fests"].tolist()]),
    #             # "fdists": ", ".join(fvec["fdists"]),
    #         },
    #         indent=4,
    #     )
    # )
