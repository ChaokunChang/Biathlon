import os
import time
import clickhouse_connect
import numpy as np
from typing import Literal, Tuple, Callable
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
import xgboost as xgb
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from beaker.cache import cache_regions, cache_region, Cache


cache_regions.update({
    'short_term': {
        'expire': 60,
        'type': 'memory'
    },
    'long_term': {
        'expire': 1800,
        'type': 'dbm',
        'data_dir': '/tmp/xip_cache',
    }
})


class DBConnector:
    def __init__(self, host="localhost", port=0, username="default", passwd="") -> None:
        self.host = host
        self.port = port
        self.username = username
        self.passwd = passwd
        # get current process id for identifying the session
        self.thread_id = os.getpid()
        self.session_time = time.time()
        self.session_id = f"session_{self.thread_id}_{self.session_time}"
        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.passwd,
            session_id=self.session_id,
        )

    def execute(self, sql):
        return self.client.query_df(sql)


class FEstimator:
    min_cnt = 30

    def estimate_any(data: np.ndarray, p: float, func: Callable, nsamples: int = 100) -> Tuple[np.ndarray, list]:
        if p >= 1.0:
            features = func(data)
            return features, [('norm', features[i], 0.0) for i in range(features.shape[0])]
        cnt = data.shape[0]
        feas = []
        for _ in range(nsamples):
            sample = data[np.random.choice(cnt, size=cnt, replace=True)]
            feas.append(func(sample))
        features = np.mean(feas, axis=0)
        if cnt < FEstimator.min_cnt:
            scales = 1e9 * np.ones_like(features)
        else:
            scales = np.std(feas, axis=0, ddof=1)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]
        return features, fests

    def estimate_min(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.min(x, axis=0))
        return features, fests

    def estimate_max(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.max(x, axis=0))
        return features, fests

    def estimate_median(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.median(x, axis=0))
        return features, fests

    def estimate_stdPop(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))
        return features, fests

    def estimate_stdSamp(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))
        return features, fests

    def estimate_unique(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.array([len(np.unique(x[:, i])) for i in range(x.shape[1])]))
        return features, fests

    def compute_dvars(data: np.ndarray):
        cnt = data.shape[0]
        if cnt < FEstimator.min_cnt:
            # if cnt is too small, set scale as big number
            return 1e9 * np.ones_like(data[0])
        else:
            return np.var(data, axis=0, ddof=1)

    def compute_closed_form_scale(features: np.ndarray, cnt: int, dvars: np.ndarray, p: float) -> np.ndarray:
        cnt = np.where(cnt < 1, 1.0, cnt)
        scales = np.sqrt(np.where(p >= 1.0, 0.0, dvars) / cnt)
        return scales

    def estimate_avg(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        cnt = data.shape[0]
        features = np.mean(data, axis=0)
        dvars = FEstimator.compute_dvars(data)
        scales = FEstimator.compute_closed_form_scale(features, cnt, dvars, p)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]
        return features, fests

    def estimate_count(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        cnt = data.shape[0]
        features = np.array([cnt / p])
        scales = FEstimator.compute_closed_form_scale(features, cnt, np.array([cnt * (1 - p) * p]), p)
        fests = [('norm', features[0], scales[0])]
        return features, fests

    def estimate_sum(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features = np.sum(data, axis=0) / p
        cnt = data.shape[0]
        dvars = FEstimator.compute_dvars(data)
        scales = FEstimator.compute_closed_form_scale(features, cnt, cnt * cnt * dvars, p)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]

        return features, fests

    def merge_ffests(ffests: list) -> list:
        # ffests: list of tuple[np.ndarray, list[tuple]]
        # return: tuple(np.ndarray, list[tuple])
        features = np.concatenate([ffest[0] for ffest in ffests], axis=0)
        fests = []
        for _, fest in ffests:
            fests.extend(fest)
        return features, fests


class BaseXIPQueryExecutor:
    def __init__(self) -> None:
        # the result will NOT be recomputed if anything with self changes.
        class_name = self.__class__.__name__
        self.executor: Callable = cache_region('short_term', class_name)(self.run)

    @classmethod
    def clear_cache(cls):
        from beaker.cache import cache_managers
        print(f'clearing cache for {cls.__name__}')
        for key, cache in cache_managers.items():
            cache.clear()
            print(f'cache {key}:{cache} cleared')

    def run(self, request: dict, cfg: dict) -> dict:
        """
        features: np.ndarray
        fests: np.ndarray
        types: List[str]
        return {'features': features, 'fests': fests, 'types': types}
        """
        raise NotImplementedError

    def __call__(self, request: dict, cfg: dict) -> dict:
        return self.executor(request, cfg)


SUPPORTED_MODELS = {
    "regressor": {"lgbm": LGBMRegressor,
                  "xgb": xgb.XGBRegressor,
                  "lr": LinearRegression,
                  "dt": DecisionTreeRegressor,
                  "rf": RandomForestRegressor,
                  "svm": SVR,
                  "mlp": MLPRegressor},
    "classifier": {"lgbm": LGBMClassifier,
                   "xgb": xgb.XGBClassifier,
                   "lr": LogisticRegression,
                   "dt": DecisionTreeClassifier,
                   "rf": RandomForestClassifier,
                   "svm": SVC,
                   "mlp": MLPClassifier},
}


def create_model(model_type: Literal["regressor", "classifier"], model_name: str, **kwargs) -> BaseEstimator:
    """ Create a model
    """
    if model_type == "regressor":
        return SUPPORTED_MODELS["regressor"][model_name](**kwargs)
    elif model_type == "classifier":
        return SUPPORTED_MODELS["classifier"][model_name](**kwargs)
    else:
        raise NotImplementedError


def get_model_type(ppl: Pipeline) -> Literal["regressor", "classifier"]:
    """ Get the model type of the pipeline
    """
    if isinstance(ppl.steps[-1][1], tuple(SUPPORTED_MODELS["classifier"].values())):
        return "classifier"
    elif isinstance(ppl.steps[-1][1], tuple(SUPPORTED_MODELS["regressor"].values())):
        return "regressor"
    else:
        raise NotImplementedError


def get_global_feature_importance(ppl: Pipeline, fnames: list) -> np.ndarray:
    """ Get the global feature importance of the pipeline
    """
    supoorted_regressor = [SUPPORTED_MODELS['regressor'][name] for name in ['lgbm', 'xgb', 'dt', 'rf']]
    supoorted_classifier = [SUPPORTED_MODELS['classifier'][name] for name in ['lgbm', 'xgb', 'dt', 'rf']]
    model_type = get_model_type(ppl)
    if model_type == "regressor":
        if isinstance(ppl.steps[-1][1], tuple(supoorted_regressor)):
            return ppl.steps[-1][1].feature_importances_
        else:
            return np.array([0] * len(fnames))
    elif model_type == "classifier":
        if isinstance(ppl.steps[-1][1], tuple(supoorted_classifier)):
            return ppl.steps[-1][1].feature_importances_
        else:
            return np.array([0] * len(fnames))
    else:
        raise NotImplementedError


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    expv = metrics.explained_variance_score(y_true, y_pred)
    maxe = metrics.max_error(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r2": r2, "expv": expv, "maxe": maxe, "size": len(y_true)}


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "size": len(y_true)}


def evaluate_pipeline(ppl: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """ Evaluate the pipeline
    """
    model_type = get_model_type(ppl)
    if model_type == "regressor":
        y_pred = ppl.predict(X)
        return evaluate_regressor(y, y_pred)
    elif model_type == "classifier":
        y_pred = ppl.predict(X)
        return evaluate_classifier(y, y_pred)
    else:
        raise NotImplementedError


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