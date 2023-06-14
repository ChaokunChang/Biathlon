import os
import os.path as osp
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from tap import Tap
import glob
import json
import time
from tqdm import tqdm

from apxinfer.utils import DBConnector, create_model, evaluate_regressor, get_global_feature_importance

EXP_HOME = '/home/ckchang/.cache/apxinf'


class PrepareStageArgs(Tap):
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation
    all_features: bool = False  # whether to use all features

    raw_data_dir: str = '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/db_src'

    req_rate: float = 0.001  # sampling rate for requests


def get_exp_dir(task: str, args: PrepareStageArgs) -> str:
    task_dir = os.path.join(EXP_HOME, task)
    model_dir = os.path.join(task_dir, args.model)
    exp_dir = os.path.join(model_dir, f'seed-{args.seed}')
    return exp_dir


def raw_data_preprocessing(raw_data_dir: str, database: str, table_name: str, seed: int) -> None:
    db_client = DBConnector().client

    def create_tables():
        pass

    def ingest_data(raw_data_dir: str):
        pass

    if not db_client.command(f"SHOW DATABASES LIKE '{database}'"):
        print(f"Create database {database}")
        db_client.command(f"CREATE DATABASE {database}")
    else:
        print(f'Database "{database}" already exists')

    create_tables()
    ingest_data(raw_data_dir)


def train_valid_test_split(data: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    # shuffle the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # calculate the number of rows for each split
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # n_test = n_total - n_train - n_valid

    # split the data
    train_set = data[:n_train]
    valid_set = data[n_train:n_train + n_valid]
    test_set = data[n_train + n_valid:]

    return train_set, valid_set, test_set


def run(raw_data_dir: str, save_dir: str, seed: int, model: str, selected_features: bool, rate: float):
    """ in test.py: run, we create synthetic pipeline to test our system
    1. data pre-processing and save to clickhouse, with sampling supported // this is ingored in test.py
    2. prepare all requests and labels and save
        requests : (request_id, f1: a float from 0 to 1000, f2: a float from 0 to 1, f3: a float from 0 to 10)
        labels   : (f1+f2+f3 // 2).astype(int)
    3. extract all (exact) features and save along with request_id
        feature = request['f1'], request['f2'], request['f3']
    4. split the requests, labels, and features into train_set, valid_set, and test_set, 0.5, 0.3, 0.2
            For classification task, make sure stratified sample
    5. train model with train_set, save the model and save with joblib
    6. evaluation model with valid_set and test_set, save into json
    """

    # data pre-processing and save to clickhouse
    database = 'default'
    table_name = 'trips'

    raw_data_preprocessing(raw_data_dir, database, table_name, seed)

    # prepare all requests and labels and features and save
    all_fops_dict = {
        'w0': ['trip_distance', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'] + ['toDayOfYear(pickup_datetime)', 'toDayOfWeek(pickup_datetime, 1)', 'toHour(pickup_datetime)', 'toMinute(pickup_datetime)'],
        'w1': ['count(*) over w1'] + [f'{agg}({col}) over w1' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']],
        'w2': ['count(*) over w2'] + [f'{agg}({col}) over w2' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']],
        'w3': ['count(*) over w3'] + [f'{agg}({col}) over w3' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']]
    }
    all_fnames_dict = {
        'w0': ['f_trip_distance', 'f_passenger_count', 'f_pickup_longitude', 'f_pickup_latitude', 'f_dropoff_longitude', 'f_dropoff_latitude'] + ['f_pickup_day_of_year', 'f_pickup_day_of_week', 'f_pickup_hour', 'f_pickup_minute'],
        'w1': ['f_count_w1'] + [f'f_{agg}_{col}_w1' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']],
        'w2': ['f_count_w2'] + [f'f_{agg}_{col}_w2' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']],
        'w3': ['f_count_w3'] + [f'f_{agg}_{col}_w3' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']]
    }
    # all_fops = all_fops_dict['w0'] + all_fops_dict['w1'] + all_fops_dict['w2'] + all_fops_dict['w3']

    selected_fops_dict = {
        'w0': ['trip_distance', 'toDayOfWeek(pickup_datetime, 1)', 'toHour(pickup_datetime)'],
        'w1': ['sum(trip_duration) over w1', 'sum(total_amount) over w1', 'stddevPop(fare_amount) over w1'],
        'w2': ['count(*) over w2', 'sum(trip_distance) over w2', 'max(trip_duration) over w2', 'max(tip_amount) over w2', 'median(tip_amount) over w2'],
        'w3': ['max(trip_distance) over w3']
    }
    selected_fnames = {
        'w0': ['f_trip_distance', 'f_pickup_day_of_week', 'f_pickup_hour'],
        'w1': ['f_sum_trip_duration_w1', 'f_sum_total_amount_w1', 'f_stddevPop_fare_amount_w1'],
        'w2': ['f_count_w2', 'f_sum_trip_distance_w2', 'f_max_trip_duration_w2', 'f_max_tip_amount_w2', 'f_median_tip_amount_w2'],
        'w3': ['f_max_trip_distance_w3']
    }

    if selected_features:
        fops_dict = selected_fops_dict
        fnames_dict = selected_fnames
    else:
        fops_dict = all_fops_dict
        fnames_dict = all_fnames_dict

    fops = fops_dict['w0'] + fops_dict['w1'] + fops_dict['w2'] + fops_dict['w3']
    fnames = fnames_dict['w0'] + fnames_dict['w1'] + fnames_dict['w2'] + fnames_dict['w3']

    db_client = DBConnector().client

    sql = f"""
        SELECT trip_id as request_trip_id,
            toString(pickup_datetime) as request_pickup_datetime,
            pickup_ntaname as request_pickup_ntaname,
            dropoff_ntaname as request_dropoff_ntaname,
            pickup_latitude as request_pickup_latitude,
            pickup_longitude as request_pickup_longitude,
            dropoff_latitude as request_dropoff_latitude,
            dropoff_longitude as request_dropoff_longitude,
            passenger_count as request_passenger_count,
            trip_distance as request_trip_distance,
            fare_amount as request_label
        FROM {database}.{table_name}
        WHERE pickup_datetime >= '2015-08-01 00:00:00'
                AND pickup_datetime < '2015-08-15 00:00:00'
                AND fare_amount is not null
                AND intHash64(pickup_datetime) % {int(1 / rate)} == 0
        ORDER BY request_trip_id
    """
    requests: pd.DataFrame = db_client.query_df(sql)
    num_reqs = len(requests)
    requests.insert(0, 'request_id', list(range(num_reqs)))

    # extract exact features with request
    requests_features = []
    dbtable = f'{database}.{table_name}'
    st = time.time()
    for request in tqdm(requests.to_dict(orient='records'), desc='Extracting features', total=num_reqs):
        pickup_datetime = request['request_pickup_datetime']
        pickup_ntaname = request['request_pickup_ntaname']
        dropoff_ntaname = request['request_dropoff_ntaname']
        passenger_count = request['request_passenger_count']

        dsrcs = {'w0': """SELECT * FROM {dbtable} WHERE trip_id={trip_id}""".format(dbtable=dbtable, trip_id=request["request_trip_id"]),
                 'w1': """SELECT * FROM {dbtable}
                            WHERE pickup_ntaname = '{pickup_ntaname}' AND
                                  pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(1) )
                                  AND pickup_datetime < '{pickup_datetime}'
                        """.format(dbtable=dbtable, pickup_datetime=pickup_datetime,
                                   pickup_ntaname=pickup_ntaname),
                 'w2': """SELECT * FROM {dbtable}
                            WHERE pickup_ntaname = '{pickup_ntaname}' AND
                                  dropoff_ntaname = '{dropoff_ntaname}' AND
                                  pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(24) )
                                  AND pickup_datetime < '{pickup_datetime}'
                        """.format(dbtable=dbtable, pickup_datetime=pickup_datetime,
                                   pickup_ntaname=pickup_ntaname, dropoff_ntaname=dropoff_ntaname),
                 'w3': """SELECT * FROM {dbtable}
                            WHERE pickup_ntaname = '{pickup_ntaname}' AND
                                  dropoff_ntaname = '{dropoff_ntaname}' AND
                                  passenger_count = {passenger_count} AND
                                  pickup_datetime >= ( toDateTime('{pickup_datetime}') - toIntervalHour(168) )
                                  AND pickup_datetime < '{pickup_datetime}'
                        """.format(dbtable=dbtable, pickup_datetime=pickup_datetime,
                                   pickup_ntaname=pickup_ntaname, dropoff_ntaname=dropoff_ntaname,
                                   passenger_count=passenger_count),
                }

        features = []
        for key in ['w0', 'w1', 'w2', 'w3']:
            qfops = [fop.replace(f' over {key}', '') for fop in fops_dict[key]]
            dsrc = dsrcs[key]
            sql = f"""
                SELECT {', '.join(qfops)}
                FROM ({dsrc}) as tmp
                """
            # print(f'sql={sql}')
            features.extend(db_client.query_np(sql)[0].tolist())
        requests_features.append(features)
    et = time.time()

    requests_features = np.array(requests_features)
    requests_features = pd.DataFrame(requests_features, columns=fnames)
    requests_features.insert(0, 'request_id', list(range(num_reqs)))

    # merge requests and requests_features
    requests = requests.merge(requests_features, on='request_id', how='left')

    # remove requests with nan and null
    requests = requests.dropna()
    requests = requests[requests['request_pickup_ntaname'] != '']
    requests = requests[requests['request_dropoff_ntaname'] != '']

    # split the requests into train_set, valid_set, and test_set
    train_set, valid_set, test_set = train_valid_test_split(requests, 0.5, 0.3, seed)

    # train model with train_set, save the model and save with joblib
    model = create_model('regressor', model, random_state=seed)
    ppl = Pipeline(
        [
            ("model", model)
        ]
    )
    ppl.fit(train_set[fnames], train_set['request_label'])
    joblib.dump(ppl, osp.join(save_dir, "pipeline.pkl"))

    # 6. evaluation model with train_set, valid_set and test_set, save into json
    train_pred = ppl.predict(train_set[fnames])
    valid_pred = ppl.predict(valid_set[fnames])
    test_pred = ppl.predict(test_set[fnames])
    train_set['ppl_pred'] = train_pred
    valid_set['ppl_pred'] = valid_pred
    test_set['ppl_pred'] = test_pred
    train_set.to_csv(osp.join(save_dir, 'train_set.csv'), index=False)
    valid_set.to_csv(osp.join(save_dir, 'valid_set.csv'), index=False)
    test_set.to_csv(osp.join(save_dir, 'test_set.csv'), index=False)

    # set statistics
    train_set.describe().to_csv(osp.join(save_dir, 'train_set_stats.csv'))
    valid_set.describe().to_csv(osp.join(save_dir, 'valid_set_stats.csv'))
    test_set.describe().to_csv(osp.join(save_dir, 'test_set_stats.csv'))

    train_evals = evaluate_regressor(train_set['request_label'], train_pred)
    valid_evals = evaluate_regressor(valid_set['request_label'], valid_pred)
    test_evals = evaluate_regressor(test_set['request_label'], test_pred)
    model_evals = {
        'train_size': len(train_set),
        'valid_size': len(valid_set),
        'test_size': len(test_set),
        'train_evals': train_evals,
        'valid_evals': valid_evals,
        'test_evals': test_evals,
        'fcomp_time': et - st
    }

    print(f'model_evals: {model_evals}')

    with open(osp.join(save_dir, 'model_evals.json'), 'w') as f:
        json.dump(model_evals, f, indent=4)

    global_feature_importance = get_global_feature_importance(ppl, fnames)
    gfimps: pd.DataFrame = pd.DataFrame({'fname': fnames,
                                         'fops': fops,
                                         'importance': global_feature_importance},
                                        columns=['fname', 'fops', 'importance'])
    gfimps.to_csv(osp.join(save_dir, 'global_feature_importance.csv'), index=False)
    # print gfimps to console, sorted by importance desc
    print(gfimps.sort_values(by='importance', ascending=False))

    # save two dicts into json
    with open(os.path.join(save_dir, 'query_features.json'), 'w') as f:
        json.dump({'fops': fops_dict, 'fnames': fnames_dict}, f, indent=4)


if __name__ == '__main__':
    args = PrepareStageArgs().parse_args()

    if args.all_features:
        exp_dir = get_exp_dir('taxi_fare_all_features', args)
    else:
        exp_dir = get_exp_dir('taxi_fare', args)

    save_dir = os.path.join(exp_dir, 'prepare')
    print(f'Save dir: {save_dir}')
    os.makedirs(save_dir, exist_ok=True)

    run(args.raw_data_dir, save_dir=save_dir, seed=args.seed,
        model=args.model,
        selected_features=not args.all_features,
        rate=args.req_rate)
