import os
import time
import clickhouse_connect
import numpy as np
from typing import Literal, Tuple
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


def executor_example(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    """ Example of an executor
    """
    # reqid = request['request_id']
    req_f1 = request['request_f1']
    req_f2 = request['request_f2']
    req_f3 = request['request_f3']
    means = [req_f1, req_f2, req_f3]
    nof = len(means)
    scales = [10] * nof

    max_nsamples = 1000
    rate = cfg['sample']
    samples = np.random.normal(means, scales, (int(max_nsamples * rate), nof))
    apxf = np.mean(samples, axis=0)
    apxf_std = np.std(samples, axis=0)
    if rate >= 1.0:
        apxf = means
        apxf_std = [0] * nof
    return apxf, [('norm', apxf[i], apxf_std[i]) for i in range(nof)]


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