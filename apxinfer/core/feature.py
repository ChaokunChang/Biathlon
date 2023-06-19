import numpy as np
import time
from typing import Callable, List, Union, Tuple
import logging
import itertools
from sklearn import metrics
from beaker.cache import CacheManager

from apxinfer.core.utils import XIPFeatureVec, XIPRequest, XIPQueryConfig, QueryCostEstimation
from apxinfer.core.utils import merge_fvecs
from apxinfer.core.query import XIPQuery

fcache_manager = CacheManager(cache_regions={'feature': {'type': 'memory', 'expire': 3600}})
logging.basicConfig(level=logging.INFO)


class FEstimatorHelper:
    min_cnt = 30

    def estimate_any(data: np.ndarray, p: float, func: Callable, nsamples: int = 100) -> XIPFeatureVec:
        if p >= 1.0:
            features = func(data)
            fnames = [f'{func.__name__}_f{i}' for i in range(features.shape[0])]
            return XIPFeatureVec(fnames=fnames, fvals=features,
                                 fests=np.zeros_like(features),
                                 fdists=['normal'] * features.shape[0])
        cnt = data.shape[0]
        estimations = []
        for _ in range(nsamples):
            sample = data[np.random.choice(cnt, size=cnt, replace=True)]
            estimations.append(func(sample))
        features = np.mean(estimations, axis=0)
        if cnt < FEstimatorHelper.min_cnt:
            scales = 1e9 * np.ones_like(features)
        else:
            scales = np.std(estimations, axis=0, ddof=1)
        fnames = [f'{func.__name__}_f{i}' for i in range(features.shape[0])]
        return XIPFeatureVec(fnames=fnames, fvals=features, fests=scales, fdists=['normal'] * features.shape[0])

    def estimate_min(data: np.ndarray, p: float) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(data, p, lambda x : np.min(x, axis=0))

    def estimate_max(data: np.ndarray, p: float) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(data, p, lambda x : np.max(x, axis=0))

    def estimate_median(data: np.ndarray, p: float) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(data, p, lambda x : np.median(x, axis=0))

    def estimate_stdPop(data: np.ndarray, p: float) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))

    def estimate_stdSamp(data: np.ndarray, p: float) -> XIPFeatureVec:
        return FEstimatorHelper.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))

    def estimate_unique(data: np.ndarray, p: float) -> XIPFeatureVec:
        unique_func: Callable = lambda x : np.array([len(np.unique(x[:, i])) for i in range(x.shape[1])])
        return FEstimatorHelper.estimate_any(data, p, unique_func)

    def compute_dvars(data: np.ndarray) -> np.ndarray:
        cnt = data.shape[0]
        if cnt < FEstimatorHelper.min_cnt:
            # if cnt is too small, set scale as big number
            return 1e9 * np.ones_like(data[0])
        else:
            return np.var(data, axis=0, ddof=1)

    def compute_closed_form_scale(features: np.ndarray, cnt: int,
                                  dvars: np.ndarray, p: float) -> np.ndarray:
        cnt = np.where(cnt < 1, 1.0, cnt)
        scales = np.sqrt(np.where(p >= 1.0, 0.0, dvars) / cnt)
        return scales

    def estimate_avg(data: np.ndarray, p: float) -> XIPFeatureVec:
        cnt = data.shape[0]
        features = np.mean(data, axis=0)
        dvars = FEstimatorHelper.compute_dvars(data)
        scales = FEstimatorHelper.compute_closed_form_scale(features, cnt, dvars, p)
        fnames = [f'avg_f{i}' for i in range(features.shape[0])]
        return XIPFeatureVec(fnames=fnames, fvals=features, fests=scales, fdists=['normal'] * features.shape[0])

    def estimate_count(data: np.ndarray, p: float) -> XIPFeatureVec:
        cnt = data.shape[0]
        features = np.array([cnt / p])
        scales = FEstimatorHelper.compute_closed_form_scale(features, cnt, np.array([cnt * (1 - p) * p]), p)
        fnames = ['cnt']
        return XIPFeatureVec(fnames=fnames, fvals=features, fests=scales, fdists=['normal'] * features.shape[0])

    def estimate_sum(data: np.ndarray, p: float) -> XIPFeatureVec:
        features = np.sum(data, axis=0) / p
        cnt = data.shape[0]
        dvars = FEstimatorHelper.compute_dvars(data)
        scales = FEstimatorHelper.compute_closed_form_scale(features, cnt, cnt * cnt * dvars, p)
        fnames = [f'sum_f{i}' for i in range(features.shape[0])]
        return XIPFeatureVec(fnames=fnames, fvals=features, fests=scales, fdists=['normal'] * features.shape[0])

    SUPPORTED_AGGS = {'min': estimate_min,
                      'max': estimate_max,
                      'median': estimate_median,
                      'std': estimate_stdPop,
                      'stdPop': estimate_stdPop,
                      'stdSamp': estimate_stdSamp,
                      'unique': estimate_unique,
                      'avg': estimate_avg,
                      'count': estimate_count,
                      'sum': estimate_sum}


SUPPORTED_DISTRIBUTIONS = {
    "normal": {"sampler": np.random.normal, "final_args": 0.0},
    "fixed": {"sampler": lambda x, size : np.array([x] * size), "final_args": []},
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
}


@fcache_manager.cache('feature', expire=3600)
def get_feature_samples(fvals: float, dist: str, dist_args: Union[list, float], seed: int, n_samples: int = 1000) -> np.ndarray:
    if dist == "fixed":
        return SUPPORTED_DISTRIBUTIONS[dist]['sampler'](fvals, size=n_samples)
    elif dist == "normal":
        scale = dist_args
        return SUPPORTED_DISTRIBUTIONS[dist]['sampler'](fvals, scale, size=n_samples)
    return SUPPORTED_DISTRIBUTIONS[dist]['sampler'](*dist_args, size=n_samples)


# @fcache_manager.cache('feature', expire=3600)
def fvec_random_sample(fvec: List[XIPFeatureVec], n_samples: int, seed: int) -> np.ndarray:
    fvals = fvec['fvals']
    fests = fvec['fests']
    fdists = fvec['fdists']

    p = len(fvals)
    np.random.seed(seed)
    samples = np.zeros((n_samples, p))
    for i in range(p):
        samples[:, i] = get_feature_samples(fvals[i], fdists[i], fests[i], seed, n_samples)
    return samples


def get_final_dist_args(dist: str) -> Union[list, float]:
    return SUPPORTED_DISTRIBUTIONS[dist]['final_args']


def evaluate_features(ext_fs: np.ndarray, apx_fs: np.ndarray) -> dict:
    # ext_fs.shape == apx_fs.shape = (n_samples, n_features)
    # calcuate mse, mae, r2, maxe for each feature, and avg of all features
    n_samples, n_features = ext_fs.shape
    mses = np.zeros(n_features)
    maes = np.zeros(n_features)
    r2s = np.zeros(n_features)
    maxes = np.zeros(n_features)
    for i in range(n_features):
        mses[i] = metrics.mean_squared_error(ext_fs[:, i], apx_fs[:, i])
        maes[i] = metrics.mean_absolute_error(ext_fs[:, i], apx_fs[:, i])
        r2s[i] = metrics.r2_score(ext_fs[:, i], apx_fs[:, i])
        maxes[i] = metrics.max_error(ext_fs[:, i], apx_fs[:, i])
    mse = np.mean(mses)
    mae = np.mean(maes)
    r2 = np.mean(r2s)
    maxe = np.mean(maxes)

    return {"mse": mse, "mae": mae, "r2": r2, "maxe": maxe,
            "mses": mses.tolist(), "maes": maes.tolist(),
            "r2s": r2s.tolist(), "maxes": maxes.tolist()}


class XIPFeatureExtractor:
    def __init__(self, queries: List[XIPQuery],
                 enable_cache: bool = False) -> None:
        self.queries = queries
        self.num_queries = len(queries)
        self.q_keys = [qry.key for qry in self.queries]
        assert len(self.q_keys) == len(set(self.q_keys)), "Query keys must be unique"
        self.fnames = list(itertools.chain.from_iterable([qry.fnames for qry in self.queries]))
        assert len(self.fnames) == len(set(self.fnames)), "Feature names must be unique"

        self.logger = logging.getLogger('XIPFeatureExtractor')

        self.enable_cache = enable_cache
        if self.enable_cache:
            self.extract = fcache_manager.cache('feature', expire=3600)(self.extract)

    def extract(self, requets: XIPRequest,
                qcfgs: List[XIPQueryConfig]) -> Tuple[List[XIPFeatureVec], List[QueryCostEstimation]]:
        qcosts = []
        fvecs = []
        for i in range(self.num_queries):
            st = time.time()
            fvecs.append(self.queries[i].run(requets, qcfgs[i]))
            et = time.time()
            qcosts.append(QueryCostEstimation(time=et - st, memory=None))
        return merge_fvecs(fvecs), qcosts

    def extract_fs_only(self, requets: XIPRequest, qcfgs: List[XIPQueryConfig]) -> List[XIPFeatureVec]:
        fvecs = []
        for i in range(self.num_queries):
            fvecs.append(self.queries[i].run(requets, qcfgs[i]))
        return merge_fvecs(fvecs)
