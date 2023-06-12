import os
import time
import clickhouse_connect
import numpy as np
from typing import Literal, Tuple
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
import xgboost as xgb
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


def create_model(model_type: Literal["regressor", "classifier"], model_name: str, **kwargs):
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
