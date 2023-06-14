from typing import Tuple
import numpy as np
# import pandas as pd
# import os
# import joblib
from joblib import Memory

from apxinfer.online.online_utils import OnlineStageArgs, run_online_stage, get_exp_dir
from apxinfer.online_stage import XIPQuery


joblib_memory = Memory('/tmp/apxinf', verbose=0)
joblib_memory.clear()


def test_helper(means, rate) -> Tuple[np.ndarray, list]:
    nof = len(means)
    scales = [0.1] * nof

    max_nsamples = 1000
    samples = np.random.normal(means, scales, (int(max_nsamples * rate), nof))
    apxf = np.mean(samples, axis=0)
    apxf_std = np.std(samples, axis=0, ddof=1)

    if rate >= 1.0:
        apxf = means
        apxf_std = [0.0] * nof

    return apxf, [('norm', apxf[i], apxf_std[i]) for i in range(nof)]


@joblib_memory.cache
def test_executor_q1(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    req_f1 = request['request_f1']
    means = [req_f1]
    return test_helper(means, cfg['sample'])


@joblib_memory.cache
def test_executor_q2(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    req_f2 = request['request_f2']
    means = [req_f2]
    return test_helper(means, cfg['sample'])


@joblib_memory.cache
def test_executor_q3(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    req_f3 = request['request_f3']
    means = [req_f3]
    return test_helper(means, cfg['sample'])


if __name__ == "__main__":
    args: OnlineStageArgs = OnlineStageArgs().parse_args()
    exp_dir = get_exp_dir(task='test', args=args)

    cfgs = [
        {'sample': 0.1 * i, 'cost': 0.1 * i}
        for i in range(1, 10 + 1)
    ]
    q1 = XIPQuery(key='q1', fnames=['f_1'], cfgs=cfgs, executor=test_executor_q1)
    q2 = XIPQuery(key='q2', fnames=['f_2'], cfgs=cfgs, executor=test_executor_q2)
    q3 = XIPQuery(key='q3', fnames=['f_3'], cfgs=cfgs, executor=test_executor_q3)
    queries = [q1, q2, q3]

    run_online_stage(args, queries, exp_dir=exp_dir)
