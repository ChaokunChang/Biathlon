from typing import Tuple, Callable
import numpy as np
import os
import json
import pickle
from datetime import datetime
# import pandas as pd
# from sklearn.pipeline import Pipeline
# import joblib
from joblib import Memory

from apxinfer.utils import DBConnector
import apxinfer.online.online_utils as online_utils
from apxinfer.online.online_utils import OnlineStageArgs, run_online_stage, get_exp_dir, FEstimator
from apxinfer.online_stage import XIPQuery


joblib_memory = Memory('/tmp/apxinf', verbose=0)
joblib_memory.clear()


@joblib_memory.cache
def executor_q0(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    assert cfg['sample'] == 1.0, "q0 only supports sample=1.0"
    trip_distance = request['request_trip_distance']
    pickup_datetime = request['request_pickup_datetime']  # format like 2015-08-15 00:00:00
    pickup_datetime = datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S')
    # get the day of week and hour of the day
    day_of_week = pickup_datetime.weekday()
    hour_of_day = pickup_datetime.hour

    features = np.array([trip_distance, day_of_week, hour_of_day])
    return features, [('norm', features[i], 0.0) for i in range(features.shape[0])]


@joblib_memory.cache
def executor_q1(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    pickup_datetime = request['request_pickup_datetime']
    pickup_ntaname = request['request_pickup_ntaname']

    # offset = 0
    sample = cfg['sample']

    sql = """
        SELECT
            trip_duration, total_amount, fare_amount
        FROM default.trips_w_samples SAMPLE {sample}
        WHERE pickup_ntaname = '{pickup_ntaname}' AND
              pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(1) ) AND
              pickup_datetime < '{pickup_datetime}'
    """.format(
        sample=sample,
        pickup_datetime=pickup_datetime,
        pickup_ntaname=pickup_ntaname,
    )
    if sample > 0.0:
        db_client = DBConnector().client
        req_data = db_client.query_np(sql)  # (nsamples, 3)
        if req_data.shape[0] == 0:
            req_data = np.array([[0, 0, 0]])
    else:
        req_data = np.array([[0, 0, 0]])

    features, fests = FEstimator.merge_ffests(
        [
            FEstimator.estimate_sum(req_data[:, 0:1], p=sample),
            FEstimator.estimate_sum(req_data[:, 1:2], p=sample),
            FEstimator.estimate_stdPop(req_data[:, 2:3], p=sample),
        ]
    )
    return features, fests


@joblib_memory.cache
def executor_q2(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    pickup_datetime = request['request_pickup_datetime']
    pickup_ntaname = request['request_pickup_ntaname']
    dropoff_ntaname = request['request_dropoff_ntaname']

    # offset = 0
    sample = cfg['sample']

    sql = """
        SELECT
            trip_distance, trip_duration, tip_amount
        FROM default.trips_w_samples SAMPLE {sample}
        WHERE pickup_ntaname = '{pickup_ntaname}' AND
              dropoff_ntaname = '{dropoff_ntaname}' AND
              pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(24) ) AND
              pickup_datetime < '{pickup_datetime}'
    """.format(
        sample=sample,
        pickup_datetime=pickup_datetime,
        pickup_ntaname=pickup_ntaname,
        dropoff_ntaname=dropoff_ntaname,
    )

    if sample > 0.0:
        db_client = DBConnector().client
        req_data = db_client.query_np(sql)  # (nsamples, 3)
        if req_data.shape[0] == 0:
            req_data = np.array([[0, 0, 0]])
    else:
        req_data = np.array([[0, 0, 0]])

    features, fests = FEstimator.merge_ffests(
        [
            FEstimator.estimate_count(req_data[:, 0:1], p=sample),
            FEstimator.estimate_sum(req_data[:, 0:1], p=sample),
            FEstimator.estimate_max(req_data[:, 1:2], p=sample),
            FEstimator.estimate_max(req_data[:, 2:3], p=sample),
            FEstimator.estimate_median(req_data[:, 2:3], p=sample),
        ]
    )
    return features, fests


@joblib_memory.cache
def executor_q3(request: dict, cfg: dict) -> Tuple[np.ndarray, list]:
    pickup_datetime = request['request_pickup_datetime']
    pickup_ntaname = request['request_pickup_ntaname']
    dropoff_ntaname = request['request_dropoff_ntaname']
    passenger_count = request['request_passenger_count']

    # offset = 0
    sample = cfg['sample']

    sql = """
        SELECT
            trip_distance
        FROM default.trips_w_samples SAMPLE {sample}
        WHERE pickup_ntaname = '{pickup_ntaname}' AND
              dropoff_ntaname = '{dropoff_ntaname}' AND
              passenger_count = {passenger_count} AND
              pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(168) ) AND
              pickup_datetime < '{pickup_datetime}'
    """.format(
        sample=sample,
        pickup_datetime=pickup_datetime,
        pickup_ntaname=pickup_ntaname,
        dropoff_ntaname=dropoff_ntaname,
        passenger_count=passenger_count,
    )

    if sample > 0.0:
        db_client = DBConnector().client
        req_data = db_client.query_np(sql)  # (nsamples, 1)
        if req_data.shape[0] == 0:
            req_data = np.array([[0]])
    else:
        req_data = np.array([[0]])

    features, fests = FEstimator.merge_ffests(
        [
            FEstimator.estimate_max(req_data[:, 0:1], p=sample),
        ]
    )
    return features, fests


if __name__ == "__main__":
    args: OnlineStageArgs = OnlineStageArgs().parse_args()
    assert not args.all_features, "use taxi_fare_all.py instead"
    exp_dir = get_exp_dir(task='taxi_fare', args=args)

    online_dir = os.path.join(exp_dir, 'online')
    online_dir = os.path.join(online_dir, f'allocator:{args.allocator}_max_round:{args.max_round}_target_bound:{args.target_bound}_target_conf:{args.target_conf}')
    os.makedirs(online_dir, exist_ok=True)

    num_feautres = 12
    qfnum = [3, 3, 5, 1]

    executors = [
        executor_q0,
        executor_q1,
        executor_q2,
        executor_q3,
    ]
    qfnum_cum = np.cumsum(qfnum)
    fnames = [f'f_{i}' for i in range(num_feautres)]
    queries = [XIPQuery(key=f'q{i}',
                        fnames=fnames[qfnum_cum[i] - qfnum[i] : qfnum_cum[i]],
                        cfgs=online_utils.get_default_cfgs(1) if i == 0 else online_utils.get_default_cfgs(10),
                        executor=executors[i])
                for i in range(len(executors))]

    online_results, evals = run_online_stage(args, queries, exp_dir=exp_dir)

    # save online results to online_dir as pickle
    online_results_path = os.path.join(online_dir, 'online_results.pkl')
    with open(online_results_path, 'wb') as f:
        pickle.dump(online_results, f)

    # save evals to online_dir as json
    evals_path = os.path.join(online_dir, 'evals.json')
    with open(evals_path, 'w') as f:
        json.dump(evals, f, indent=4)
