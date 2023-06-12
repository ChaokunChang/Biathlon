import numpy as np
from typing import Literal
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor


def executor_example(request: dict, cfg: dict):
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


def get_model_type(ppl: Pipeline) -> Literal["regressor", "classifier"]:
    """ Get the model type of the pipeline
    """
    if isinstance(ppl.steps[-1][1], LGBMClassifier):
        return "classifier"
    elif isinstance(ppl.steps[-1][1], LGBMRegressor):
        return "regressor"
    else:
        raise NotImplementedError
