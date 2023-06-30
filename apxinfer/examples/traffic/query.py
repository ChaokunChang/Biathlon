import numpy as np
import pandas as pd

from apxinfer.core.utils import XIPQueryConfig, XIPFeatureVec, XIPQType
from apxinfer.core.utils import merge_fvecs
from apxinfer.core.data import DBHelper
from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import FEstimatorHelper

from apxinfer.examples.traffic.data import TrafficRequest
from apxinfer.examples.traffic.data import dt_to_req, req_to_dt
from apxinfer.examples.traffic.data import TrafficHourDataLoader
from apxinfer.examples.traffic.data import TrafficFStoreLoader


def get_borough_embedding() -> dict:
    """borough embedding, take borough str, output an int"""
    db_client = DBHelper.get_db_client()
    boroughs: pd.DataFrame = db_client.query_df(
        "select distinct borough from xip.traffic order by borough"
    )
    borough_map = {borough: i for i, borough in enumerate(boroughs["borough"].values)}
    return borough_map


class TrafficQP0(XIPQuery):
    def __init__(self, qname: str, enable_cache: bool = False) -> None:
        data_loader = None
        fnames = ["year", "month", "day", "hour", "borough"]
        super().__init__(qname, XIPQType.NORMAL, data_loader, fnames, enable_cache)
        self.borough_map = get_borough_embedding()

    def run(self, request: TrafficRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        borough = self.borough_map[request["req_borough"]]
        fvals = np.array(
            [
                request["req_year"],
                request["req_month"],
                request["req_day"],
                request["req_hour"],
                borough,
            ]
        )
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class TrafficQP1(XIPQuery):
    def __init__(
        self, qname: str, data_loader: TrafficFStoreLoader, enable_cache: bool = False
    ) -> None:
        assert data_loader.granularity == "hour", "data loader must be hour level"
        fnames = [
            "last_hour_cnt",
            "last_hour_avg_speed",
            "last_hour_avg_travel_time",
            "last_hour_std_speed",
            "last_hour_std_travel_time",
            "last_hour_min_speed",
            "last_hour_min_travel_time",
            "last_hour_max_speed",
            "last_hour_max_travel_time",
            "last_hour_median_speed",
            "last_hour_median_travel_time",
        ]
        super().__init__(qname, XIPQType.FSTORE, data_loader, fnames, enable_cache)

    def run(self, request: TrafficRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        fcols = [fname.replace("last_hour_", "") for fname in self.fnames]
        req_dt = req_to_dt(request)
        last_hour = req_dt - pd.Timedelta(hours=1)
        last_hour_req = dt_to_req(
            last_hour, req_id=request["req_id"], borough=request["req_borough"]
        )
        fvals = self.data_loader.load_data(last_hour_req, qcfg, fcols)
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class TrafficQP2(XIPQuery):
    def __init__(
        self,
        qname: str,
        data_loader: TrafficHourDataLoader,
        enable_cache: bool = False,
    ) -> None:
        fnames = [
            "this_hour_cnt",
            "this_hour_avg_speed",
            "this_hour_avg_travel_time",
            "this_hour_std_speed",
            "this_hour_std_travel_time",
            "this_hour_min_speed",
            "this_hour_min_travel_time",
            "this_hour_max_speed",
            "this_hour_max_travel_time",
            "this_hour_median_speed",
            "this_hour_median_travel_time",
        ]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

        self.cached_reqid = -1
        self.cached_qsample = 0
        self.cached_data = None

    def run(self, request: TrafficRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        fcols = ["speed", "travel_time"]
        req_data = self.data_loader.load_data(request, qcfg, fcols)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if qsample >= 1.0:
                self.logger.warning(f"no data for {request}")
                fests = np.zeros(len(self.fnames))
            else:
                # self.logger.warning(f'no data for {request} with qsample={qsample}')
                fests = np.ones(len(self.fnames)) * 1e9
            return XIPFeatureVec(
                fnames=self.fnames,
                fvals=fvals,
                fests=fests,
                fdists=["normal"] * len(self.fnames),
            )

        aggs = ["count", "avg", "std", "min", "max", "median"]
        fvecs = []
        for i, agg in enumerate(aggs):
            fvec: XIPFeatureVec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                req_data, qsample
            )
            fvecs.append(fvec)
        fvec = merge_fvecs(fvecs, new_names=self.fnames)
        return fvec


class TrafficQP3(XIPQuery):
    def __init__(
        self, qname: str, data_loader: TrafficFStoreLoader, enable_cache: bool = False
    ) -> None:
        assert data_loader.granularity == "day", "data loader must be day level"
        fnames = [
            "last_day_cnt",
            "last_day_avg_speed",
            "last_day_avg_travel_time",
            "last_day_std_speed",
            "last_day_std_travel_time",
            "last_day_min_speed",
            "last_day_min_travel_time",
            "last_day_max_speed",
            "last_day_max_travel_time",
            "last_day_median_speed",
            "last_day_median_travel_time",
        ]
        super().__init__(qname, XIPQType.FSTORE, data_loader, fnames, enable_cache)

    def run(self, request: TrafficRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        fcols = [fname.replace("last_day_", "") for fname in self.fnames]
        req_dt = req_to_dt(request)
        last_day = req_dt - pd.Timedelta(days=1)
        last_day_req = dt_to_req(
            last_day, req_id=request["req_id"], borough=request["req_borough"]
        )
        fvals = self.data_loader.load_data(last_day_req, qcfg, fcols)
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class TrafficQP4(XIPQuery):
    def __init__(
        self, qname: str, data_loader: TrafficFStoreLoader, enable_cache: bool = False
    ) -> None:
        assert data_loader.granularity == "hour", "data loader must be hour level"
        fnames = [
            "last_dayhour_cnt",
            "last_dayhour_avg_speed",
            "last_dayhour_avg_travel_time",
            "last_dayhour_std_speed",
            "last_dayhour_std_travel_time",
            "last_dayhour_min_speed",
            "last_dayhour_min_travel_time",
            "last_dayhour_max_speed",
            "last_dayhour_max_travel_time",
            "last_dayhour_median_speed",
            "last_dayhour_median_travel_time",
        ]
        super().__init__(qname, XIPQType.FSTORE, data_loader, fnames, enable_cache)

    def run(self, request: TrafficRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        fcols = [fname.replace("last_dayhour_", "") for fname in self.fnames]
        req_dt = req_to_dt(request)
        last_dayhour = req_dt - pd.Timedelta(days=1) + pd.Timedelta(hours=1)
        last_dayhour_req = dt_to_req(
            last_dayhour, req_id=request["req_id"], borough=request["req_borough"]
        )
        fvals = self.data_loader.load_data(last_dayhour_req, qcfg, fcols)
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )
