import numpy as np
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQueryConfig, XIPFeatureVec, XIPQType
from apxinfer.core.data import DBHelper
from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import get_fvec_auto

from apxinfer.examples.tick.data import TickRequest
from apxinfer.examples.tick.data import TickThisHourDataLoader, TickHourFStoreDataLoader


def get_embedding() -> dict:
    db_client = DBHelper.get_db_client()
    cpairs: pd.DataFrame = db_client.query_df(
        "select distinct cpair from xip.tick order by cpair"
    )
    currencies = []
    for cpair in cpairs["cpair"].values:
        currencies.append(cpair.split("/")[0])
        currencies.append(cpair.split("/")[1])
    currencies = sorted(list(set(currencies)))
    currency_map = {currency: i + 1 for i, currency in enumerate(currencies)}
    return currency_map


class TickQP0(XIPQuery):
    def __init__(self, qname: str, enable_cache: bool = False) -> None:
        qtype = XIPQType.NORMAL
        data_loader = None
        fnames = ["cfrom", "cto"]
        super().__init__(qname, qtype, data_loader, fnames, enable_cache)
        self.cmap = get_embedding()

    def run(self, request: TickRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        cpair = request["req_cpair"]
        cfrom = self.cmap.get(cpair.split("/")[0], 0)
        cto = self.cmap.get(cpair.split("/")[1], 0)
        fvals = np.array([cfrom, cto])
        return self.get_fvec_with_default_est(fvals=fvals)


class TickQP1(XIPQuery):
    def __init__(self, qname: str, enable_cache: bool = False, seed: int = 0) -> None:
        qtype = XIPQType.AGG
        data_loader = TickThisHourDataLoader(
            "clickhouse", "xip", "tick", seed, enable_cache
        )
        self.dcols = ["bid"]
        # self.dcol_aggs = [["avg", "stddevSamp"]]
        self.dcol_aggs = [["avg"]]
        fnames = [
            f"{agg}_{col}_hour"
            for col, aggs in zip(self.dcols, self.dcol_aggs)
            for agg in aggs
        ]
        super().__init__(qname, qtype, data_loader, fnames, enable_cache)

    def run(self, request: TickRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)
        if req_data is None or len(req_data) == 0:
            ret = self.get_default_fvec(request, qcfg)
        else:
            ret = get_fvec_auto(
                fnames=self.fnames,
                req_data=req_data,
                dcol_aggs=self.dcol_aggs,
                qsample=qcfg["qsample"],
                tsize=self.data_loader.statistics["tsize"],
            )
        return ret


class TickQP2(XIPQuery):
    def __init__(
        self, qname: str, enable_cache: bool = False, seed: int = 0, offset: int = 1
    ) -> None:
        qtype = XIPQType.FSTORE
        data_loader = TickHourFStoreDataLoader(
            "clickhouse", "xip", "tick_fstore_hour", seed, enable_cache
        )
        self.offset = offset
        self.dcols = ["bid"]
        # self.dcol_aggs = [["avg", "stddevSamp"]]
        self.dcol_aggs = [["avg"]]
        fnames = [
            f"fstore_{agg}_{col}_hour_{offset}"
            for col, aggs in zip(self.dcols, self.dcol_aggs)
            for agg in aggs
        ]
        super().__init__(qname, qtype, data_loader, fnames, enable_cache)
        self.fstore_cols = [
            f"{agg}_{col}"
            for col, aggs in zip(self.dcols, self.dcol_aggs)
            for agg in aggs
        ]

    def run(self, request: TickRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        last_hour = pd.to_datetime(request["req_dt"]) - dt.timedelta(hours=self.offset),
        req: TickRequest = TickRequest(
            req_id=request["req_id"],
            req_dt=last_hour.__str__(),
            req_cpair=request["req_cpair"],
        )
        fvals = self.data_loader.load_data(req, qcfg, cols=self.fstore_cols)
        return self.get_fvec_with_default_est(fvals=fvals)
