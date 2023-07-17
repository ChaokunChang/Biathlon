import numpy as np
import time
from typing import Callable, List, Union, Tuple
import logging
import itertools
from sklearn import metrics
from beaker.cache import CacheManager

from apxinfer.core.utils import (
    XIPFeatureVec,
    XIPRequest,
    XIPQueryConfig,
    QueryCostEstimation,
)
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.query import XIPQuery

fcache_manager = CacheManager(
    cache_regions={"feature": {"type": "memory", "expire": 3600}}
)
logging.basicConfig(level=logging.INFO)


def fest_profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        FEstimatorHelper.total_time += time.time() - started_at
        return result

    return wrap


class FEstimatorHelper:
    min_cnt = 30
    bs_nsamp: int = 100
    bias_correction: bool = True
    total_time: float = 0
    bs_time: float = 0

    @fest_profile
    def estimate_any(
        data: np.ndarray,
        p: float,
        func: Callable,
        tsize: int,
    ) -> XIPFeatureVec:
        features = func(data)
        fnames = [f"{func.__name__}_f{i}" for i in range(features.shape[0])]
        if features is None:
            features = np.zeros(len(fnames))
        if is_same_float(p, 1.0):
            return XIPFeatureVec(
                fnames=fnames,
                fvals=features,
                fests=np.zeros_like(features),
                fdists=["normal"] * features.shape[0],
            )
        cnt = data.shape[0]
        if cnt < FEstimatorHelper.min_cnt:
            scales = 1e9 * np.ones_like(features)
        else:
            st = time.time()
            estimations = []
            for _ in range(FEstimatorHelper.bs_nsamp):
                sample = data[np.random.choice(cnt, size=cnt, replace=True)]
                estimations.append(func(sample))
            features = np.mean(estimations, axis=0)
            scales = np.std(estimations, axis=0, ddof=1)
            FEstimatorHelper.bs_time += time.time() - st
        if FEstimatorHelper.bias_correction:
            # Bias Correction
            bias = func(data) - features
            features = func(data) + bias
        return XIPFeatureVec(
            fnames=fnames,
            fvals=features,
            fests=scales,
            fdists=["normal"] * features.shape[0],
        )

    @fest_profile
    def estimate_min(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(
            data, p, lambda x: np.min(x, axis=0), tsize
        )

    @fest_profile
    def estimate_max(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(
            data, p, lambda x: np.max(x, axis=0), tsize
        )

    def estimate_median(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(
            data, p, lambda x: np.median(x, axis=0), tsize
        )

    @fest_profile
    def estimate_stdPop(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        # if we enable bias_correction, ddof should be 0
        ddof = int(not FEstimatorHelper.bias_correction)
        if data.shape[0] < 2:
            ddof = 0
        return FEstimatorHelper.estimate_any(
            data, p, lambda x: np.std(x, axis=0, ddof=ddof), tsize
        )

    @fest_profile
    def estimate_unique(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        unique_func: Callable = lambda x: np.array(
            [len(np.unique(x[:, i])) for i in range(x.shape[1])]
        )
        return FEstimatorHelper.estimate_any(data, p, unique_func, tsize)

    @fest_profile
    def compute_dvars(data: np.ndarray, ddof: int = 1) -> np.ndarray:
        cnt = data.shape[0]
        if cnt < FEstimatorHelper.min_cnt:
            # if cnt is too small, set scale as big number
            return 1e9 * np.ones_like(data[0])
        else:
            return np.var(data, axis=0, ddof=ddof)

    @fest_profile
    def fstds_crop(fstds: np.ndarray, p: float, card: int) -> np.ndarray:
        if is_same_float(p, 1.0):
            return np.zeros_like(fstds)
        elif card < FEstimatorHelper.min_cnt:
            return 1e9 * np.ones_like(fstds)
        else:
            return fstds

    @fest_profile
    def estimate_avg(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        cnt = data.shape[0]
        features = np.mean(data, axis=0)
        dvars = FEstimatorHelper.compute_dvars(data)
        fscales = np.sqrt(dvars / cnt)
        scales = FEstimatorHelper.fstds_crop(fscales, p, cnt)

        # the following is a better estimator with better variance.
        # tcnt = cnt / p
        # fscales = np.sqrt(dvars) * np.sqrt((tcnt - cnt) / (tcnt * cnt - cnt))
        # scales = FEstimatorHelper.fstds_crop(fscales, p, cnt)

        fnames = [f"avg_f{i}" for i in range(features.shape[0])]
        return XIPFeatureVec(
            fnames=fnames,
            fvals=features,
            fests=scales,
            fdists=["normal"] * features.shape[0],
        )

    @fest_profile
    def estimate_count(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        ssize = int(tsize * p)
        scnt = data.shape[0]
        tcnt = int(scnt / p)
        slct = tcnt / tsize
        features = np.array([tcnt])
        if ssize * (tsize - 1) > 0:
            fstds = tsize * np.sqrt(
                (tsize - ssize) * slct * (1 - slct) / (ssize * (tsize - 1))
            )
        else:
            fstds = np.zeros_like(features)
        fstds = FEstimatorHelper.fstds_crop([fstds], p, scnt)
        fnames = ["cnt"]
        return XIPFeatureVec(
            fnames=fnames,
            fvals=features,
            fests=fstds,
            fdists=["normal"] * features.shape[0],
        )

    @fest_profile
    def estimate_sum(data: np.ndarray, p: float, tsize: int) -> XIPFeatureVec:
        ssize = int(tsize * p)
        scnt = data.shape[0]
        # tcnt = int(scnt / p)

        features = np.sum(data, axis=0) / p
        sdvars = FEstimatorHelper.compute_dvars(data)
        if ssize * (tsize - 1) > 0:
            fstds = tsize * sdvars * np.sqrt((tsize - ssize) / (ssize * (tsize - 1)))
        else:
            fstds = np.zeros_like(features)
        fstds = FEstimatorHelper.fstds_crop(fstds, p, scnt)

        fnames = [f"sum_f{i}" for i in range(features.shape[0])]
        return XIPFeatureVec(
            fnames=fnames,
            fvals=features,
            fests=fstds,
            fdists=["normal"] * features.shape[0],
        )

    SUPPORTED_AGGS = {
        "min": estimate_min,
        "max": estimate_max,
        "median": estimate_median,
        "std": estimate_stdPop,
        "stdPop": estimate_stdPop,
        "unique": estimate_unique,
        "avg": estimate_avg,
        "count": estimate_count,
        "sum": estimate_sum,
    }


def get_fvec_auto(
    fnames: List[str],
    req_data: np.ndarray,
    dcol_aggs: List[List[str]],
    qsample: float,
    tsize: int,
) -> XIPFeatureVec:
    fvecs = []
    for i, aggs in enumerate(dcol_aggs):
        for agg in aggs:
            fvec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                req_data[:, i : i + 1], qsample, tsize
            )
            fvecs.append(fvec)
    return merge_fvecs(fvecs, new_names=fnames)


SUPPORTED_DISTRIBUTIONS = {
    "normal": {"sampler": np.random.normal, "final_args": 0.0},
    "fixed": {"sampler": lambda x, size: np.ones(size) * x, "final_args": 0.0},
    "uniform": {"sampler": np.random.uniform, "final_args": []},
    "beta": {"sampler": np.random.beta, "final_args": []},
    "gamma": {"sampler": np.random.gamma, "final_args": []},
    "exponential": {"sampler": np.random.exponential, "final_args": []},
    "lognormal": {"sampler": np.random.lognormal, "final_args": []},
    "power": {"sampler": np.random.power, "final_args": []},
    "chisquare": {"sampler": np.random.chisquare, "final_args": []},
    "logistic": {"sampler": np.random.logistic, "final_args": []},
    "poisson": {"sampler": np.random.poisson, "final_args": []},
    "binomial": {"sampler": np.random.binomial, "final_args": []},
    "negative_binomial": {"sampler": np.random.negative_binomial, "final_args": []},
    "geometric": {"sampler": np.random.geometric, "final_args": []},
    "zipf": {"sampler": np.random.zipf, "final_args": []},
    "dirichlet": {"sampler": np.random.dirichlet, "final_args": []},
    "multinomial": {"sampler": np.random.multinomial, "final_args": []},
    "multivariate_normal": {"sampler": np.random.multivariate_normal, "final_args": []},
    "unknown": {"sampler": None, "final_args": []},
}


@fcache_manager.cache("feature", expire=3600)
def get_feature_samples(
    fvals: float,
    dist: str,
    dist_args: Union[list, float],
    seed: int,
    n_samples: int = 1000,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if dist == "fixed":
        return fvals * np.ones(n_samples)
    elif dist == "normal":
        scale = dist_args
        return rng.normal(fvals, scale, size=n_samples)
    elif dist == "unknown":
        # in this case, dist_args is the samples itself
        if dist_args is None or len(dist_args) == 0:
            return np.ones(n_samples) * fvals
        else:
            return dist_args[rng.randint(0, len(dist_args), size=n_samples)]
    return SUPPORTED_DISTRIBUTIONS[dist]["sampler"](*dist_args, size=n_samples)


# @fcache_manager.cache('feature', expire=3600)
def fvec_random_sample(
    fvec: List[XIPFeatureVec], n_samples: int, seed: int
) -> np.ndarray:
    fvals = fvec["fvals"]
    fests = fvec["fests"]
    fdists = fvec["fdists"]

    p = len(fvals)
    samples = np.zeros((n_samples, p))
    for i in range(p):
        samples[:, i] = get_feature_samples(
            fvals[i], fdists[i], fests[i], seed, n_samples
        )
    return samples


def get_final_dist_args(dist: str) -> Union[list, float]:
    return SUPPORTED_DISTRIBUTIONS[dist]["final_args"]


def evaluate_features(ext_fs: np.ndarray, apx_fs: np.ndarray) -> dict:
    # ext_fs.shape == apx_fs.shape = (n_samples, n_features)
    # calcuate mse, mae, r2, maxe for each feature, and avg of all features
    n_samples, n_features = ext_fs.shape
    mses = np.zeros(n_features)
    maes = np.zeros(n_features)
    mapes = np.zeros(n_features)
    r2s = np.zeros(n_features)
    maxes = np.zeros(n_features)
    for i in range(n_features):
        mses[i] = metrics.mean_squared_error(ext_fs[:, i], apx_fs[:, i])
        maes[i] = metrics.mean_absolute_error(ext_fs[:, i], apx_fs[:, i])
        mapes[i] = metrics.mean_absolute_percentage_error(ext_fs[:, i], apx_fs[:, i])
        r2s[i] = metrics.r2_score(ext_fs[:, i], apx_fs[:, i])
        maxes[i] = metrics.max_error(ext_fs[:, i], apx_fs[:, i])
    mse = np.mean(mses)
    mae = np.mean(maes)
    mape = np.mean(mapes)
    r2 = np.mean(r2s)
    maxe = np.mean(maxes)

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "maxe": maxe,
        "mses": mses.tolist(),
        "maes": maes.tolist(),
        "mapes": mapes.tolist(),
        "r2s": r2s.tolist(),
        "maxes": maxes.tolist(),
    }


class XIPFeatureExtractor:
    def __init__(
        self,
        queries: List[XIPQuery],
        enable_cache: bool = False,
        loading_nthreads: int = 1,
    ) -> None:
        self.queries = queries
        self.num_queries = len(queries)
        self.qnames = [qry.qname for qry in self.queries]
        assert len(self.qnames) == len(set(self.qnames)), "Query names must be unique"
        self.fnames = list(
            itertools.chain.from_iterable([qry.fnames for qry in self.queries])
        )
        assert len(self.fnames) == len(set(self.fnames)), "Feature names must be unique"

        self.logger = logging.getLogger("XIPFeatureExtractor")

        self.enable_cache = enable_cache
        if self.enable_cache:
            self.extract = fcache_manager.cache("feature", expire=3600)(self.extract)
        self.loading_nthreads = loading_nthreads

    def extract(
        self, requets: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> Tuple[XIPFeatureVec, List[QueryCostEstimation]]:
        qcosts = []
        fvecs = []
        for i in range(self.num_queries):
            st = time.time()
            fvecs.append(
                self.queries[i].run(
                    requets, qcfgs[i], loading_nthreads=self.loading_nthreads
                )
            )
            et = time.time()
            qcard = self.queries[i].estimate_cardinality(requets, qcfgs[i])
            qcosts.append(QueryCostEstimation(time=et - st, memory=None, qcard=qcard))
        return merge_fvecs(fvecs), qcosts

    def extract_fs_only(
        self, requets: XIPRequest, qcfgs: List[XIPQueryConfig]
    ) -> XIPFeatureVec:
        fvecs = []
        for i in range(self.num_queries):
            fvecs.append(
                self.queries[i].run(
                    requets, qcfgs[i], loading_nthreads=self.loading_nthreads
                )
            )
        return merge_fvecs(fvecs)
