import numpy as np
import pandas as pd

from apxinfer.core.utils import XIPQueryConfig, XIPFeatureVec, XIPQType
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import DBHelper
from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import FEstimatorHelper

from apxinfer.examples.ccfraud.data import CCFraudRequest
from apxinfer.examples.ccfraud.data import CCFraudTxnsLoader
from apxinfer.examples.ccfraud.data import CCFraudCardsLoader
from apxinfer.examples.ccfraud.data import CCFraudUsersIngestor


def get_embedding(database: str, table: str, col: str) -> dict:
    """borough embedding, take borough str, output an int"""
    db_client = DBHelper.get_db_client()
    df: pd.DataFrame = db_client.query_df(
        f"select distinct {col} from {database}.{table} order by {col}"
    )
    embedding = {value: i + 1 for i, value in enumerate(df[col].values)}
    return embedding


class CCFraudQ0(XIPQuery):
    def __init__(
        self, qname: str, database: str, table: str, enable_cache: bool = False
    ) -> None:
        data_loader = None
        self.dt_fnames = ["hour"]
        self.num_fnames = ["amount", "zip_code", "mcc"]
        self.cat_fnames = [
            "use_chip",
            "merchant_name",
            "merchant_city",
            "merchant_state",
        ]
        fnames = self.dt_fnames + self.num_fnames + self.cat_fnames
        super().__init__(qname, XIPQType.NORMAL, data_loader, fnames, enable_cache)
        self.embeddings = {
            k: get_embedding(database, table, k) for k in self.cat_fnames
        }

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        txn_dt = pd.to_datetime(request["req_txn_datetime"])
        dt_fvals = np.array([txn_dt.hour])
        num_fvals = np.array([request[f"req_{key}"] for key in self.num_fnames])
        cat_keys = [f"req_{key}" for key in self.cat_fnames]
        cat_fvals = np.array(
            [
                self.embeddings[key.replace("req_", "")].get(f"{request[key]}", 0)
                for key in cat_keys
            ]
        )
        fvals = np.concatenate([dt_fvals, num_fvals, cat_fvals])
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class CCFraudQ1(XIPQuery):
    def __init__(
        self, qname: str, data_loader: CCFraudCardsLoader, enable_cache: bool = False
    ) -> None:
        self.num_fnames = [
            "cvv",
            "has_chip",
            "cards_issued",
            "credit_limit",
            "pin_last_changed",
            "card_on_dark_web",
        ]
        self.cat_fnames = ["card_brand", "card_type"]
        fnames = self.num_fnames + self.cat_fnames
        super().__init__(qname, XIPQType.KeySearch, data_loader, fnames, enable_cache)
        database = self.data_loader.database
        table = self.data_loader.table
        self.embeddings = {
            k: get_embedding(database, table, k) for k in self.cat_fnames
        }

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        fcols = self.fnames
        fvals = self.data_loader.load_data(request, qcfg, fcols, loading_nthreads)
        for i in range(len(self.num_fnames), len(fvals)):
            fvals[i] = self.embeddings[self.cat_fnames[i - len(self.num_fnames)]].get(
                fvals[i], 0
            )
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class CCFraudQ2(XIPQuery):
    def __init__(
        self, qname: str, data_loader: CCFraudUsersIngestor, enable_cache: bool = False
    ) -> None:
        self.num_fnames = [
            "current_age",
            "retirement_age",
            "birth_year",
            "birth_month",
            "gender",
            "per_capita_income",
            "yearly_income",
            "total_debt",
            "fico_score",
            "num_credit_cards",
        ]
        self.cat_fnames = ["uname", "address", "apartment", "city", "state", "zipcode"]
        fnames = self.num_fnames + self.cat_fnames
        super().__init__(qname, XIPQType.KeySearch, data_loader, fnames, enable_cache)
        database = self.data_loader.database
        table = self.data_loader.table
        self.embeddings = {
            k: get_embedding(database, table, k) for k in self.cat_fnames
        }

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        fcols = self.fnames
        fvals = self.data_loader.load_data(request, qcfg, fcols, loading_nthreads)
        for i in range(len(self.num_fnames), len(fvals)):
            fvals[i] = self.embeddings[self.cat_fnames[i - len(self.num_fnames)]].get(
                fvals[i], 0
            )
        fests = np.zeros_like(fvals)
        fdists = ["normal"] * len(fvals)
        return XIPFeatureVec(
            fnames=self.fnames, fvals=fvals, fests=fests, fdists=fdists
        )


class CCFraudQ3(XIPQuery):
    def __init__(
        self,
        qname: str,
        data_loader: CCFraudTxnsLoader,
        enable_cache: bool = False,
    ) -> None:
        wsize = data_loader.window_size
        fnames = [
            f"cnt_{wsize}days",
            f"avg_amount_{wsize}days",
            f"std_amount_{wsize}days",
            f"min_amount_{wsize}days",
            f"max_amount_{wsize}days",
            f"median_amount_{wsize}days",
        ]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

        self.cached_reqid = -1
        self.cached_qsample = 0
        self.cached_data = None

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        fcols = ["amount"]
        req_data = self.data_loader.load_data(request, qcfg, fcols, loading_nthreads)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
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
                req_data, qsample, self.data_loader.statistics["tsize"]
            )
            fvecs.append(fvec)
        fvec = merge_fvecs(fvecs, new_names=self.fnames)
        return fvec


class CCFraudQ4(XIPQuery):
    def __init__(
        self,
        qname: str,
        data_loader: CCFraudTxnsLoader,
        enable_cache: bool = False,
    ) -> None:
        wsize = data_loader.window_size
        col = "is_fraud"
        fnames = [f"sum_{col}_{wsize}days", f"avg_{col}_{wsize}days"]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

        self.cached_reqid = -1
        self.cached_qsample = 0
        self.cached_data = None

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        fcols = ["is_fraud"]
        req_data = self.data_loader.load_data(request, qcfg, fcols, loading_nthreads)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
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

        aggs = ["sum", "avg"]
        fvecs = []
        for i, agg in enumerate(aggs):
            fvec: XIPFeatureVec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                req_data, qsample, self.data_loader.statistics["tsize"]
            )
            fvecs.append(fvec)
        fvec = merge_fvecs(fvecs, new_names=self.fnames)
        return fvec


class CCFraudQ5(XIPQuery):
    def __init__(
        self,
        qname: str,
        data_loader: CCFraudTxnsLoader,
        enable_cache: bool = False,
    ) -> None:
        wsize = data_loader.window_size
        col = "errors"
        fnames = [f"unique_{col}_{wsize}days"]
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

        self.cached_reqid = -1
        self.cached_qsample = 0
        self.cached_data = None

    def run(self, request: CCFraudRequest, qcfg: XIPQueryConfig, loading_nthreads: int = 1) -> XIPFeatureVec:
        qsample = qcfg["qsample"]
        fcols = ["errors"]
        req_data = self.data_loader.load_data(request, qcfg, fcols, loading_nthreads)

        if req_data is None or len(req_data) == 0:
            fvals = np.zeros(len(self.fnames))
            if is_same_float(qsample, 1.0):
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

        aggs = ["unique"]
        fvecs = []
        for i, agg in enumerate(aggs):
            fvec: XIPFeatureVec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                req_data, qsample, self.data_loader.statistics["tsize"]
            )
            fvecs.append(fvec)
        fvec = merge_fvecs(fvecs, new_names=self.fnames)
        return fvec
