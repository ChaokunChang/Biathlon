from typing import Tuple
import numpy as np
import os
import json
import pickle
# import pandas as pd
# from sklearn.pipeline import Pipeline
# import joblib
from joblib import Memory

from apxinfer.utils import DBConnector
from apxinfer.online.online_utils import OnlineStageArgs, run_online_stage, get_exp_dir
from apxinfer.online_stage import XIPQuery


joblib_memory = Memory('/tmp/apxinf', verbose=0)
joblib_memory.clear()

max_nsamples = 50000


def estimate_avg(data: np.array) -> Tuple[np.ndarray, list]:
    features = np.mean(data, axis=0)

    cnt = data.shape[0]
    cnt = np.where(cnt < 1, 1.0, cnt)
    var = np.var(data, axis=0, ddof=1)
    scale = np.sqrt(np.where(cnt >= 50000, 0.0, var) / cnt)
    # if cnt is too small, set scale as big number
    scale = np.where(cnt < 30, 1e9, scale)
    fests = [('norm', features[i], scale[i]) for i in range(features.shape[0])]

    return features, fests


@joblib_memory.cache
def executor_q0(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 0
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q1(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 1
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q2(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 2
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q3(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 3
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q4(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 4
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q5(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 5
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q6(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 6
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


@joblib_memory.cache
def executor_q7(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    sensor_id = 7
    bid = request['request_bid']
    noffset = 0
    nsamples = np.ceil(max_nsamples * cfg['sample'])
    sql = """
        SELECT sensor_{sensor_id} AS f_{sensor_id}
        FROM machinery_more.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsamples})
    """.format(
        sensor_id=sensor_id,
        bid=bid,
        noffset=noffset,
        nsamples=nsamples,
    )
    db_client = DBConnector().client
    req_data = db_client.query_np(sql)  # (nsamples, 1)
    return estimate_avg(req_data)


if __name__ == "__main__":
    args: OnlineStageArgs = OnlineStageArgs().parse_args()
    if args.multi_class:
        exp_dir = get_exp_dir(task='machinery_multi_class', args=args)
    else:
        exp_dir = get_exp_dir(task='machinery', args=args)

    online_dir = os.path.join(exp_dir, 'online')
    os.makedirs(online_dir, exist_ok=True)

    cfgs = [
        {'sample': 0.1 * i, 'cost': 0.1 * i}
        for i in range(1, 10 + 1)
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
