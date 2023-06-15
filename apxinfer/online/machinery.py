from typing import Tuple
import numpy as np
import os
import json
import pickle
# import pandas as pd
# from sklearn.pipeline import Pipeline
# import joblib
from joblib import Memory

from apxinfer.utils import DBConnector, FEstimator
from apxinfer.online.online_utils import OnlineStageArgs, run_online_stage, get_exp_dir
from apxinfer.online_stage import XIPQuery


joblib_memory = Memory('/tmp/apxinf', verbose=0)
joblib_memory.clear()

task_name = 'machinery'


def base_executor(request: dict, cfg: dict, sensor_id: int) -> Tuple[np.ndarray, list]:
    bid = request['request_bid']
    sample = cfg['sample']
    max_nchunks = 100
    table_prefix = task_name
    noffset = 0
    nsamples = np.ceil(max_nchunks * sample)
    sql = f"""
            SELECT sensor_{sensor_id} AS f_{sensor_id}
            FROM xip.{table_prefix}_sensor_{sensor_id}
            WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
        """
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    if req_data.shape[0] == 0:
        print(f'request_bid={bid} has no data with sample={sample} on executor_q{sensor_id}')
    return FEstimator.estimate_avg(req_data, sample)


@joblib_memory.cache
def executor_q0(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 0)


@joblib_memory.cache
def executor_q1(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 1)


@joblib_memory.cache
def executor_q2(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 2)


@joblib_memory.cache
def executor_q3(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 3)


@joblib_memory.cache
def executor_q4(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 4)


@joblib_memory.cache
def executor_q5(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 5)


@joblib_memory.cache
def executor_q6(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 6)


@joblib_memory.cache
def executor_q7(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    return base_executor(request, cfg, 7)


if __name__ == "__main__":
    args: OnlineStageArgs = OnlineStageArgs().parse_args()
    exp_dir = get_exp_dir(task=task_name, args=args)

    online_dir = os.path.join(exp_dir, 'online')
    os.makedirs(online_dir, exist_ok=True)

    granularity = 100
    cfgs = [
        {'sample': i * 1.0 / granularity, 'cost': i * 1.0 / granularity}
        for i in range(1, granularity + 1)
    ]
    executors = [
        executor_q0,
        executor_q1,
        executor_q2,
        executor_q3,
        executor_q4,
        executor_q5,
        executor_q6,
        executor_q7,
    ]
    queries = [XIPQuery(key=f'q{i}', fnames=[f'f_{i}'], cfgs=cfgs, executor=executors[i])
               for i in range(8)]

    online_results, evals = run_online_stage(args, queries, exp_dir=exp_dir)

    # save online results to online_dir as pickle
    online_results_path = os.path.join(online_dir, 'online_results.pkl')
    with open(online_results_path, 'wb') as f:
        pickle.dump(online_results, f)

    # save evals to online_dir as json
    evals_path = os.path.join(online_dir, 'evals.json')
    with open(evals_path, 'w') as f:
        json.dump(evals, f, indent=4)
