from typing import List, TypedDict, Union, Callable
import logging
import numpy as np
import pandas as pd
import time
import json
from tap import Tap
import asyncio
import datetime as dt


from apxinfer.examples.taxi.data import TaxiTripRequest
from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.festimator import XIPFeatureErrorEstimator, XIPFeatureEstimator
from apxinfer.core.query import XIPQOperatorDescription, XIPQueryProcessor
from tap import Tap


class QueryTestArgs(Tap):
    verbose: bool = False
    qsample: float = 0.1
    ld_nthreads: int = 0
    cp_nthreads: int = 0


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


def get_qps(
    bs_nresamples: int = 100,
    cp_nthreads: int = 1,
    include_qp4: bool = False,
    verbose: bool = False,
) -> List[XIPQueryProcessor]:
    qp0 = ExampleQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = ExampleQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = ExampleQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp3 = ExampleQP3(
        qname="q3",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp4 = ExampleQP4(
        qname="q4",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2, qp3]
    if include_qp4:
        qps.append(qp4)
    ferr_est = XIPFeatureErrorEstimator(
        bs_nresamples=bs_nresamples,
        bs_max_nthreads=cp_nthreads if cp_nthreads > 0 else 8,
    )
    festimator = XIPFeatureEstimator(ferr_est)
    for qp in qps:
        qp.set_estimator(festimator)
    return qps


def get_qcfgs(
    qps: List[XIPQueryProcessor], qsample: float, qoffset: float = 0.0
) -> List[XIPQueryConfig]:
    qcfgs = [
        qp.get_qcfg(
            0,
            qsample if qp.qtype == XIPQType.AGG else 1.0,
            qoffset,
            loading_nthreads=args.ld_nthreads,
            computing_nthreads=args.cp_nthreads,
        )
        for qp in qps
    ]
    return qcfgs


def run_sequential(
    qps: List[XIPQueryProcessor], qcfgs: List[XIPQueryConfig]
) -> Union[XIPFeatureVec, float]:
    # run qps sequentially one-by-one
    st = time.time()
    prepare_time = time.time() - st
    st = time.time()
    fvecs = []
    for qp, qcfg in zip(qps, qcfgs):
        fvec = qp.run(request, qcfg)
        fvecs.append(fvec)
    fvec = merge_fvecs(fvecs)
    tcost = time.time() - st

    print(f"prepare time: {prepare_time}")
    for qp in qps:
        print(f"qprofile-{qp.qname}: {qp.profiles[-1]}")
    print(f"run sequential : {tcost}")
    return fvec, tcost


async def run_async(
    qps: List[XIPQueryProcessor], qcfgs: List[XIPQueryConfig], verbose: bool = True
) -> Union[XIPFeatureVec, float]:
    # run qps asynchrously with asyncio
    st = time.time()
    prepare_time = time.time() - st
    st = time.time()
    fvecs = await asyncio.gather(
        *[qp.run_async(request, qcfg) for qp, qcfg in zip(qps, qcfgs)]
    )
    fvec = merge_fvecs(fvecs)
    tcost = time.time() - st

    if verbose:
        print(f"prepare time: {prepare_time}")
        for qp in qps:
            print(f"qprofile-{qp.qname}: {qp.profiles[-1]}")
        print(f"run asynchronously : {tcost}")
    return fvec, tcost


if __name__ == "__main__":
    args = QueryTestArgs().parse_args()

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

    print(f"run with args: {args}")
    qsample = args.qsample

    qps = get_qps(verbose=args.verbose)
    qcfgs = get_qcfgs(qps, qsample)
    fvec, tcost = run_sequential(qps, qcfgs)

    qps = get_qps(verbose=args.verbose)
    qcfgs = get_qcfgs(qps, qsample)
    fvec_async, tcost_async = asyncio.run(run_async(qps, qcfgs))
    print(f"sync v.s. async = {tcost} : {tcost_async} = {tcost / tcost_async}")

    # iterative async run, 10 parts per iter
    # only better than sequential OIP when qsample < 0.3
    # only better than asynio OIP when qsample < 0.2
    qps = get_qps(verbose=args.verbose)
    tcost_async_iter = 0.0
    for i in range(0, int(100 * qsample), 10):
        qoffset = 0
        qsample = (i + 10.0) / 100
        qcfgs = get_qcfgs(qps, qsample, qoffset)
        fvec_async_iter, itercost = asyncio.run(run_async(qps, qcfgs, False))
        tcost_async_iter += itercost
    print(
        f"sync v.s. async-iter = {tcost} : {tcost_async_iter} = {tcost / tcost_async_iter}"
    )

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
