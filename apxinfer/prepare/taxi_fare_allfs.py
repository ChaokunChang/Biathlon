import os
import os.path as osp
from typing import List, Tuple
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
import logging
import warnings

import apxinfer.utils as xutils
from apxinfer.utils import DBConnector
from apxinfer.prepare.prepare_utils import PrepareStageArgs, DBWorker, DatasetWorker
import apxinfer.prepare.prepare_utils as putils


class TaxiFareDBWorker(DBWorker):
    def __init__(self, database: str, tables: List[str],
                 data_src: str, src_type: str,
                 max_nchunks: int, seed: int) -> None:
        super().__init__(database, tables, data_src, src_type, max_nchunks, seed)

    def get_dbtable(self) -> str:
        assert len(self.tables) == 1, "TaxiDatasetWorker only supports one table"
        database = self.database
        table_name = self.tables[0]  # should be taxi_trips
        dbtable = f'{database}.{table_name}'
        return dbtable

    def create_tables(self) -> None:
        self.logger.info(f'Creating database {self.database} and tables {self.tables}')
        dbtable = self.get_dbtable()
        sql = f"""CREATE TABLE IF NOT EXISTS {dbtable} (
                    trip_id UInt32,
                    pickup_datetime DateTime,
                    dropoff_datetime DateTime,
                    pickup_longitude Nullable(Float64),
                    pickup_latitude Nullable(Float64),
                    dropoff_longitude Nullable(Float64),
                    dropoff_latitude Nullable(Float64),
                    passenger_count UInt8,
                    trip_distance Float32,
                    fare_amount Float32,
                    extra Float32,
                    tip_amount Float32,
                    tolls_amount Float32,
                    total_amount Float32,
                    payment_type Enum(
                        'CSH' = 1,
                        'CRE' = 2,
                        'NOC' = 3,
                        'DIS' = 4,
                        'UNK' = 5
                    ),
                    pickup_ntaname LowCardinality(String),
                    dropoff_ntaname LowCardinality(String),
                    trip_duration Float32,
                    pid UInt32 -- partition key, used for sampling
                ) ENGINE = MergeTree() ORDER BY (pid, pickup_datetime)
                SETTINGS index_granularity = 32
            """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        dbtable = self.get_dbtable()
        self.logger.info(f'Ingesting data into database {dbtable} from {self.src_type}::{self.data_src}')

        nchunks = self.max_nchunks
        seed = self.seed
        # get number of rows in data source
        if self.src_type == 'csv':
            # nrows = putils.get_csv_nrows(self.data_src)
            raise NotImplementedError
        elif self.src_type == 'clickhouse':
            data_src = self.data_src
            nrows = self.db_client.command(f'SELECT count() FROM {data_src}')
        else:
            raise ValueError(f'Unsupported data source type {self.src_type}')

        sql = f"""
            INSERT INTO {dbtable}
            SELECT trip_id, pickup_datetime, dropoff_datetime,
              pickup_longitude, pickup_latitude,
              dropoff_longitude, dropoff_latitude,
              passenger_count, trip_distance, fare_amount,
              extra, tip_amount, tolls_amount, total_amount,
              payment_type, pickup_ntaname, dropoff_ntaname,
              trip_duration, pid
            FROM
            (
                SELECT *, row_number() over () as row_id
                FROM {data_src}
            ) as tmp1
            JOIN
            (
                SELECT value % {nchunks} as pid, row_number() over () as row_id
                FROM (
                    SELECT *
                    FROM generateRandom('value UInt32', {seed})
                    LIMIT {nrows}
                )
            ) as tmp2
            ON tmp1.row_id = tmp2.row_id
        """
        self.db_client.command(sql)


class TaxiFareDatasetWorker(DatasetWorker):
    def __init__(self, working_dir: str, dbworker: DBWorker,
                 max_requests: int,
                 train_ratio: float, valid_ratio: float,
                 model_type: str, model_name: str,
                 seed: int) -> None:
        super().__init__(working_dir, dbworker, max_requests, train_ratio, valid_ratio, model_type, model_name, seed)

    def create_dataset(self) -> Tuple[pd.DataFrame, List[str], str]:
        self.logger.info(f'Creating dataset for {self.model_type} {self.model_name}')
        dbtable = self.dbworker.get_dbtable()
        db_client = self.db_client
        max_requests = self.max_requests

        def get_all_win_fops(window: str) -> Tuple[List[str], List[str]]:
            cnt = [f'count(*) over {window}']
            uniques = [f'uniqExact({col}) over {window}' for col in ['passenger_count', 'payment_type', 'pickup_ntaname', 'dropoff_ntaname']]
            aggs = [f'{agg}({col}) over {window}' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']]
            fops = cnt + uniques + aggs

            cnt_names = [f'f_count_{window}']
            uniques_names = [f'f_unique_{col}_{window}' for col in ['passenger_count', 'payment_type', 'pickup_ntaname', 'dropoff_ntaname']]
            aggs_names = [f'f_{agg}_{col}_{window}' for agg in ['sum', 'avg', 'stddevPop', 'median', 'min', 'max'] for col in ['trip_distance', 'fare_amount', 'tip_amount', 'trip_duration']]
            fnames = cnt_names + uniques_names + aggs_names
            return fops, fnames

        fops_dict = {
            'w0': ['trip_distance', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'] + ['toDayOfYear(pickup_datetime)', 'toDayOfWeek(pickup_datetime, 1)', 'toHour(pickup_datetime)', 'toMinute(pickup_datetime)'],
            'w1': get_all_win_fops('w1')[0],
            'w2': get_all_win_fops('w2')[0],
            'w3': get_all_win_fops('w3')[0],
        }
        fnames_dict = {
            'w0': ['f_trip_distance', 'f_passenger_count', 'f_pickup_longitude', 'f_pickup_latitude', 'f_dropoff_longitude', 'f_dropoff_latitude'] + ['f_pickup_day_of_year', 'f_pickup_day_of_week', 'f_pickup_hour', 'f_pickup_minute'],
            'w1': get_all_win_fops('w1')[1],
            'w2': get_all_win_fops('w2')[1],
            'w3': get_all_win_fops('w3')[1],
        }
        fnames = fnames_dict['w0'] + fnames_dict['w1'] + fnames_dict['w2'] + fnames_dict['w3']
        label_name = 'request_label'

        requests_range = """pickup_datetime >= '2015-08-01 00:00:00'
                            AND pickup_datetime < '2015-08-15 00:00:00'
                            AND fare_amount is not null"""
        total_num_reqs = db_client.command(f"SELECT count() FROM {dbtable} WHERE {requests_range}")
        rate = max_requests / total_num_reqs
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
                fare_amount as {label_name}
            FROM {dbtable}
            WHERE {requests_range} AND intHash64(pickup_datetime) % {int(1 / rate)} == 0
            ORDER BY request_trip_id
        """
        requests: pd.DataFrame = db_client.query_df(sql)
        num_reqs = len(requests)
        requests.insert(0, 'request_id', list(range(num_reqs)))

        # extract exact features with request
        requests_features = []
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

        requests_features = np.array(requests_features)
        requests_features = pd.DataFrame(requests_features, columns=fnames)
        requests_features.insert(0, 'request_id', list(range(num_reqs)))

        # merge requests and requests_features
        requests = requests.merge(requests_features, on='request_id', how='left')

        # remove requests with nan and null
        requests = requests.dropna()
        requests = requests[requests['request_pickup_ntaname'] != '']
        requests = requests[requests['request_dropoff_ntaname'] != '']
        return requests, fnames, label_name


if __name__ == "__main__":
    args = PrepareStageArgs().parse_args()

    db_worker = TaxiFareDBWorker(database='xip', tables=['taxi_trips'],
                                 data_src='default.trips', src_type='clickhouse',
                                 max_nchunks=100,
                                 seed=args.seed)
    db_worker.work()

    exp_dir = putils.get_exp_dir(task='taxi_fare_allfs', args=args)
    working_dir = os.path.join(exp_dir, 'prepare')
    os.makedirs(working_dir, exist_ok=True)

    model_type = 'regressor'
    model_name = args.model
    max_requests = args.max_requests
    dataset_worker = TaxiFareDatasetWorker(working_dir=working_dir, dbworker=db_worker, max_requests=max_requests,
                                           train_ratio=args.train_ratio, valid_ratio=args.valid_ratio,
                                           model_type=model_type, model_name=args.model,
                                           seed=args.seed)
    dataset_worker.work()
