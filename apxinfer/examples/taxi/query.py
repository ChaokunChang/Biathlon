from typing import List
import numpy as np
import pandas as pd

from apxinfer.core.utils import XIPQueryConfig, XIPFeatureVec, XIPQType
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import FEstimatorHelper

from apxinfer.examples.taxi.data import TaxiTripRequest
from apxinfer.examples.taxi.data import TaxiTripLoader


class TaxiTripQ0(XIPQuery):
    def __init__(self, qname: str, enable_cache: bool = False) -> None:
        data_loader = None
        fnames = ["trip_distance", "day_of_week", "hour_of_day", "passenger_count"]
        super().__init__(qname, XIPQType.NORMAL, data_loader, fnames, enable_cache)

    def run(self, request: TaxiTripRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        req_pickup_datetime = pd.to_datetime(request["req_pickup_datetime"])
        fvals = np.array(
            [
                request["req_trip_distance"],
                req_pickup_datetime.weekday(),
                req_pickup_datetime.hour,
                request["req_passenger_count"],
            ]
        )
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class TaxiTripQ1(XIPQuery):
    def __init__(
        self,
        qname: str,
        enable_cache: bool = False,
        nparts: int = 100,
        seed: int = 0,
    ) -> None:
        self.window_hours = 1
        self.condition_cols = ["pickup_ntaname"]
        self.dcols = ["trip_duration", "total_amount", "fare_amount"]
        self.finished_only = True

        data_loader: XIPDataLoader = TaxiTripLoader(
            backend="clickhouse",
            database="xip",
            table="trips",
            seed=seed,
            enable_cache=enable_cache,
            nparts=nparts,
            window_hours=self.window_hours,
            condition_cols=self.condition_cols,
            finished_only=self.finished_only,
        )

        # f'count_{self.window_hours}h',
        fnames: List[str] = [
            f"sum_trip_duration_{self.window_hours}h",
            f"sum_total_amount_{self.window_hours}h",
            f"std_fare_amount_{self.window_hours}h",
        ]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

    def run(self, request: TaxiTripRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
                self.logger.warning(f"no data for {request}")
                fests = np.zeros(len(self.fnames))
            else:
                fests = np.ones(len(self.fnames)) * 1e9
            return XIPFeatureVec(
                fnames=self.fnames,
                fvals=fvals,
                fests=fests,
                fdists=["normal"] * len(self.fnames),
            )
        else:
            fvecs = []
            dcol_aggs = [["sum"], ["sum"], ["std"]]
            for i, aggs in enumerate(dcol_aggs):
                for agg in aggs:
                    fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                        req_data[:, i : i + 1],
                        qsample,
                        self.data_loader.statistics["tsize"],
                    )
                    fvecs.append(fvec)
            return merge_fvecs(fvecs, new_names=self.fnames)


class TaxiTripQ2(XIPQuery):
    def __init__(
        self,
        qname: str,
        enable_cache: bool = False,
        nparts: int = 100,
        seed: int = 0,
    ) -> None:
        self.window_hours = 24
        self.condition_cols = ["pickup_ntaname", "dropoff_ntaname"]
        self.dcols = ["trip_distance", "trip_duration", "tip_amount"]
        self.finished_only = True

        data_loader: XIPDataLoader = TaxiTripLoader(
            backend="clickhouse",
            database="xip",
            table="trips",
            seed=seed,
            enable_cache=enable_cache,
            nparts=nparts,
            window_hours=self.window_hours,
            condition_cols=self.condition_cols,
            finished_only=self.finished_only,
        )

        fnames: List[str] = [
            f"count_{self.window_hours}h",
            f"sum_trip_distance_{self.window_hours}h",
            f"max_trip_duration_{self.window_hours}h",
            f"max_tip_amount_{self.window_hours}h",
            f"median_tip_amount_{self.window_hours}h",
        ]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

    def run(self, request: TaxiTripRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
                self.logger.warning(f"no data for {request}")
                fests = np.zeros(len(self.fnames))
            else:
                fests = np.ones(len(self.fnames)) * 1e9
            return XIPFeatureVec(
                fnames=self.fnames,
                fvals=fvals,
                fests=fests,
                fdists=["normal"] * len(self.fnames),
            )
        else:
            fvecs = [
                FEstimatorHelper.SUPPORTED_AGGS["count"](
                    req_data, qsample, self.data_loader.statistics["tsize"]
                )
            ]
            dcol_aggs = [["sum"], ["max"], ["max", "median"]]
            for i, aggs in enumerate(dcol_aggs):
                for agg in aggs:
                    fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                        req_data[:, i : i + 1],
                        qsample,
                        self.data_loader.statistics["tsize"],
                    )
                    fvecs.append(fvec)
            return merge_fvecs(fvecs, new_names=self.fnames)


class TaxiTripQ3(XIPQuery):
    def __init__(
        self,
        qname: str,
        enable_cache: bool = False,
        nparts: int = 100,
        seed: int = 0,
    ) -> None:
        self.window_hours = 24 * 7
        self.condition_cols = ["pickup_ntaname", "dropoff_ntaname", "passenger_count"]
        self.dcols = ["trip_distance"]
        self.finished_only = False

        data_loader: XIPDataLoader = TaxiTripLoader(
            backend="clickhouse",
            database="xip",
            table="trips",
            seed=seed,
            enable_cache=enable_cache,
            nparts=nparts,
            window_hours=self.window_hours,
            condition_cols=self.condition_cols,
            finished_only=self.finished_only,
        )

        fnames: List[str] = [f"max_trip_distance_{self.window_hours}h"]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

    def run(self, request: TaxiTripRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
                self.logger.warning(f"no data for {request}")
                fests = np.zeros(len(self.fnames))
            else:
                fests = np.ones(len(self.fnames)) * 1e9
            return XIPFeatureVec(
                fnames=self.fnames,
                fvals=fvals,
                fests=fests,
                fdists=["normal"] * len(self.fnames),
            )
        else:
            fvecs = []
            dcol_aggs = [["max"]]
            for i, aggs in enumerate(dcol_aggs):
                for agg in aggs:
                    fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                        req_data[:, i : i + 1],
                        qsample,
                        self.data_loader.statistics["tsize"],
                    )
                    fvecs.append(fvec)
            return merge_fvecs(fvecs, new_names=self.fnames)


class TaxiTripAGGFull(XIPQuery):
    def __init__(
        self,
        qname: str,
        window_hours: int = 1,
        condition_cols: List[str] = ["pickup_ntaname"],
        finished_only: bool = False,
        dcols: List[str] = ["trip_distance"],
        aggs: List[str] = [
            "count",
            "sum",
            "avg",
            "min",
            "max",
            "median",
            "std",
            "variance",
        ],
        enable_cache: bool = False,
        nparts: int = 100,
        seed: int = 0,
    ) -> None:
        self.window_hours = window_hours
        self.condition_cols = condition_cols
        self.finished_only = finished_only
        self.dcols = dcols
        self.aggs = aggs

        data_loader: XIPDataLoader = TaxiTripLoader(
            backend="clickhouse",
            database="xip",
            table="trips",
            seed=seed,
            enable_cache=enable_cache,
            nparts=nparts,
            window_hours=self.window_hours,
            condition_cols=self.condition_cols,
            finished_only=self.finished_only,
        )
        fname_suffix = f"{self.window_hours}h_cond_{'-'.join(self.condition_cols)}"
        fname_suffix += f"_{'finished' if self.finished_only else 'all'}"
        fnames = []
        for agg in self.aggs:
            if agg == "count":
                fnames.append(f"count_{fname_suffix}")
            elif agg == "unique":
                for dcol in self.dcols:
                    if dcol in ["pickup_ntaname", "dropoff_ntaname", "passenger_count"]:
                        fnames.append(f"unique_{dcol}_{fname_suffix}")
            else:
                for dcol in self.dcols:
                    if dcol not in ["pickup_ntaname", "dropoff_ntaname"]:
                        fnames.append(f"{agg}_{dcol}_{fname_suffix}")

        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

    def run(self, request: TaxiTripRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
                self.logger.warning(f"no data for {request}")
                fests = np.zeros(len(self.fnames))
            else:
                fests = np.ones(len(self.fnames)) * 1e9
            return XIPFeatureVec(
                fnames=self.fnames,
                fvals=fvals,
                fests=fests,
                fdists=["normal"] * len(self.fnames),
            )
        else:
            fvecs = []
            for agg in self.aggs:
                if agg == "count":
                    fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                        req_data, qsample, self.data_loader.statistics["tsize"]
                    )
                    fvecs.append(fvec)
                elif agg == "unique":
                    for dcol in self.dcols:
                        if dcol in [
                            "pickup_ntaname",
                            "dropoff_ntaname",
                            "passenger_count",
                        ]:
                            col_id = self.dcols.index(dcol)
                            fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                                req_data[:, col_id : col_id + 1],
                                qsample,
                                self.data_loader.statistics["tsize"],
                            )
                            fvecs.append(fvec)
                else:
                    for dcol in self.dcols:
                        if dcol not in ["pickup_ntaname", "dropoff_ntaname"]:
                            col_id = self.dcols.index(dcol)
                            fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                                req_data[:, col_id : col_id + 1],
                                qsample,
                                self.data_loader.statistics["tsize"],
                            )
                            fvecs.append(fvec)
            return merge_fvecs(fvecs, new_names=self.fnames)
