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


class MachineryHealthDBWorker(DBWorker):
    def __init__(self, database: str, tables: List[str],
                 data_src: str, src_type: str,
                 max_nchunks: int, seed: int) -> None:
        super().__init__(database, tables, data_src, src_type, max_nchunks, seed)

    def create_tables(self) -> None:
        self.logger.info(f'Creating database {self.database} and tables {self.tables}')
        num_sensors = 8
        database = self.database
        assert len(self.tables) == (num_sensors + 1)
        base_table_name = self.tables[0]
        table_name = base_table_name
        self.logger.info(f"Create tables {table_name} to store data from all (8x) sensors")
        typed_signal = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        self.db_client.command(
            """CREATE TABLE IF NOT EXISTS {database}.{table_name}
                        (rid UInt64, label UInt32, tag String,
                            bid UInt32, pid UInt32,
                            {typed_signal})
                        ENGINE = MergeTree()
                        ORDER BY (pid, bid)
                        SETTINGS index_granularity = 32
                        """.format(
                database=database, table_name=table_name, typed_signal=typed_signal
            )
        )

        for i in range(num_sensors):
            sensor_table_name = f"{table_name}_sensor_{i}"
            assert sensor_table_name == self.tables[i + 1]
            self.logger.info(f"Create tables {table_name} to store data from sensor_{i}")
            self.db_client.command(
                """CREATE TABLE IF NOT EXISTS {database}.{sensor_table_name}
                            (rid UInt64, label UInt32, tag String,
                                bid UInt32, pid UInt32,
                                sensor_{i} Float32)
                            ENGINE = MergeTree()
                            ORDER BY (pid, bid)
                            SETTINGS index_granularity = 32
                            """.format(
                    database=database, sensor_table_name=sensor_table_name, i=i
                )
            )

    def get_raw_data_files_list(self, data_dir: str) -> list:
        normal_file_names = glob.glob(os.path.join(data_dir, "normal", "normal", "*.csv"))
        imnormal_file_names_6g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "6g", "*.csv")
        )
        imnormal_file_names_10g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "10g", "*.csv")
        )
        imnormal_file_names_15g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "15g", "*.csv")
        )
        imnormal_file_names_20g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "20g", "*.csv")
        )
        imnormal_file_names_25g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "25g", "*.csv")
        )
        imnormal_file_names_30g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "30g", "*.csv")
        )

        # concat all file names
        file_names = []
        file_names += normal_file_names
        file_names += imnormal_file_names_6g
        file_names += imnormal_file_names_10g
        file_names += imnormal_file_names_15g
        file_names += imnormal_file_names_20g
        file_names += imnormal_file_names_25g
        file_names += imnormal_file_names_30g

        return file_names

    def get_ingestion_query(self, database: str, table_name: str,
                            table_size: int, label: int, tag: str,
                            start_bid: int, file_nrows: int,
                            segment_nrows: int, nchunks: int) -> str:
        values = ", ".join([f"sensor_{i} AS sensor_{i}" for i in range(8)])
        values_w_type = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        query = f"""INSERT INTO {database}.{table_name} \
                    SELECT ({table_size} + row_number() OVER () - 1) AS rid, \
                            {label} AS label, {tag} AS tag, {start_bid} + floor(((row_number() OVER ()) - 1)/{segment_nrows}) AS bid, \
                            ((row_number() OVER ()) - 1) % {nchunks} AS pid, \
                            {values} \
                    FROM input('{values_w_type}') \
                    FORMAT CSV \
                """
        return query

    def ingest_to_base_table(self, database: str, table_name: str,
                             files: List[str], file_nrows: int,
                             segments_per_file: int) -> None:
        segment_nrows = file_nrows // segments_per_file
        nchunks = self.max_nchunks
        for bid, src in tqdm(enumerate(files),
                             desc=f"Ingesting data to {table_name}",
                             total=len(files)):
            filename = os.path.basename(src)
            tag = '.'.join(filename.split(".")[:-1])
            dirname = os.path.basename(os.path.dirname(src))
            label = ["normal", "6g", "10g", "15g", "20g", "25g", "30g"].index(dirname)
            cnt = self.db_client.command(f"SELECT count(*) FROM {database}.{table_name}")
            start_bid = bid * segments_per_file
            query = self.get_ingestion_query(database, table_name, cnt,
                                             label, tag, start_bid,
                                             file_nrows, segment_nrows,
                                             nchunks)
            command = f"""clickhouse-client --query "{query}" < {src}"""
            os.system(command)

    def ingest_base_from_csv(self) -> None:
        assert self.src_type == "csv", "Only support csv data source"
        all_file_names = self.get_raw_data_files_list(self.data_src)

        database = self.database
        table_name = self.tables[0]
        file_nrows = 250000
        segments_per_file = 5
        self.ingest_to_base_table(database, table_name, all_file_names, file_nrows, segments_per_file)

    def ingest_base_from_clickhouse(self) -> None:
        dbtable = f'{self.database}.{self.tables[0]}'
        file_nrows = 250000
        seed = self.seed
        dsrc = self.data_src

        # make sure dsrc exists
        database, table = dsrc.split(".")
        query = f"SELECT * FROM system.tables WHERE database = '{database}' AND name = '{table}'"
        assert len(self.db_client.command(query)) > 0, f"Database {dsrc} does not exist"

        values = ", ".join([f"sensor_{i} AS sensor_{i}" for i in range(8)])
        # values_w_type = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        sql = f"""
                INSERT INTO {dbtable}
                SELECT tmp1.rid as rid, tmp1.label as label, tmp1.tag as tag,
                        tmp1.bid as bid, tmp2.pid as pid,
                        {values}
                FROM
                (
                    SELECT (row_number() over ()) as row_id, * FROM {dsrc}
                ) as tmp1
                JOIN
                (
                    SELECT (row_number() over ()) as row_id, random_number
                    FROM (
                        SELECT *
                        FROM generateRandom('random_number UInt32', {seed})
                        LIMIT {file_nrows}
                    )
                ) as tmp2
                ON tmp1.row_id = tmp2.row_id
        """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        self.logger.info(f'Ingesting data into database {self.database} from {self.src_type}::{self.data_src}')

        database = self.database
        table_name = self.tables[0]

        if self.db_client.command(f"SELECT count(*) FROM {database}.{table_name}") == 0:
            print(f"Ingest data to base table {table_name}")
            if self.src_type == "csv":
                self.ingest_base_from_csv()
            elif self.src_type == 'clickhouse':
                self.ingest_base_from_clickhouse()
            else:
                raise NotImplementedError(f"Unsupported data source type {self.src_type}")

        num_sensors = 8
        for i in tqdm(range(num_sensors),
                      desc="Ingesting data to sub tables",
                      total=num_sensors):
            sensor_table_name = self.tables[i + 1]
            if self.db_client.command(f"SELECT count(*) FROM {database}.{sensor_table_name}") == 0:
                print(f"Ingest data to tables {sensor_table_name}")
                self.db_client.command(
                    f"INSERT INTO {database}.{sensor_table_name} SELECT rid, label, tag, bid, pid, sensor_{i} FROM {database}.{table_name}"
                )


class MachineryHealthShuffleDBWorker(MachineryHealthDBWorker):
    def __init__(self, database: str, tables: List[str],
                 data_src: str, src_type: str,
                 max_nchunks: int, seed: int) -> None:
        super().__init__(database, tables, data_src, src_type, max_nchunks, seed)

    def get_ingestion_query(self, database: str, table_name: str,
                            table_size: int, label: int, tag: str,
                            start_bid: int, file_nrows: int,
                            segment_nrows: int, nchunks: int) -> str:
        seed = self.seed  # can not make sure deterministic
        values = ", ".join([f"sensor_{i} AS sensor_{i}" for i in range(8)])
        values_w_type = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        query = f"""INSERT INTO {database}.{table_name} \
                    SELECT tmp1.rid as rid, tmp1.label as label, tmp1.tag as tag, \
                            tmp1.bid as bid, tmp2.pid as pid, \
                            {values} \
                    FROM \
                    ( \
                        SELECT \
                                ({table_size} + row_number() OVER () - 1) AS rid, \
                                {label} AS label, {tag} AS tag, \
                                {start_bid} + floor(((row_number() OVER ()) - 1)/{segment_nrows}) AS bid, \
                                {values} \
                        FROM input('{values_w_type}') \
                    ) as tmp1 \
                    JOIN \
                    ( \
                        SELECT ({table_size} + row_number() OVER () - 1) as rid, \
                                random_number % {nchunks} as pid \
                        FROM ( \
                            SELECT * \
                            FROM generateRandom('random_number UInt32', {seed}) \
                            LIMIT {file_nrows} \
                        ) \
                    ) as tmp2 \
                    ON tmp1.rid = tmp2.rid \
                    FORMAT CSV \
                """
        return query


class MachineryHealthDatasetWorker(DatasetWorker):
    def __init__(self, working_dir: str, dbworker: DBWorker,
                 max_requests: int,
                 train_ratio: float, valid_ratio: float,
                 model_type: str, model_name: str,
                 seed: int) -> None:
        super().__init__(working_dir, dbworker, max_requests, train_ratio, valid_ratio, model_type, model_name, seed)

    def create_dataset(self) -> Tuple[pd.DataFrame, List[str], str]:
        self.logger.info(f'Creating dataset for {self.model_type} {self.model_name}')
        # prepare all requests and labels and features and save
        num_sensors = 8
        database = self.database
        table_name = self.tables[0]

        fnames = [f'f_{i}' for i in range(num_sensors)]
        label_name = 'request_label'
        db_client = DBConnector().client
        sql = """
            SELECT bid as request_bid, label as {label_name},
            {fop_as_name}
            FROM {database}.{table_name} GROUP BY bid, label
            ORDER BY bid
        """.format(database=database, table_name=table_name,
                   label_name=label_name,
                   fop_as_name=', '.join([f'avg(sensor_{i}) as f_{i}' for i in range(num_sensors)])
                   )
        requests: pd.DataFrame = db_client.query_df(sql)
        num_reqs = len(requests)
        requests.insert(0, 'request_id', list(range(num_reqs)))
        return requests, fnames, label_name


class MachineryHealthBinaryDatasetWorker(MachineryHealthDatasetWorker):
    def __init__(self, working_dir: str, dbworker: DBWorker,
                 max_requests: int,
                 train_ratio: float, valid_ratio: float,
                 model_type: str, model_name: str,
                 seed: int) -> None:
        super().__init__(working_dir, dbworker, max_requests, train_ratio, valid_ratio, model_type, model_name, seed)

    def create_dataset(self) -> Tuple[pd.DataFrame, List[str], str]:
        requests, fnames, label_name = super().create_dataset()
        requests['request_label'] = (requests['request_label'] > 0).astype(int)
        return requests, fnames, label_name


if __name__ == "__main__":
    args = PrepareStageArgs().parse_args()
    data_dir = '/home/ckchang/ApproxInfer/data/machinery'

    if args.shuffle_table:
        task_name = 'machinery_shuffle'
        tables = [task_name] + [f'{task_name}_sensor_{i}' for i in range(8)]
        db_worker = MachineryHealthShuffleDBWorker(database='xip',
                                                   tables=tables,
                                                   data_src=data_dir,
                                                   src_type='csv',
                                                   max_nchunks=100,
                                                   seed=args.seed)
    else:
        task_name = 'machinery'
        tables = [task_name] + [f'{task_name}_sensor_{i}' for i in range(8)]
        db_worker = MachineryHealthDBWorker(database='xip',
                                            tables=tables,
                                            data_src=data_dir, src_type='csv',
                                            max_nchunks=100,
                                            seed=args.seed)
    db_worker.work()

    exp_dir = putils.get_exp_dir(task=task_name, args=args)
    working_dir = os.path.join(exp_dir, 'prepare')
    os.makedirs(working_dir, exist_ok=True)

    model_type = 'classifier'
    model_name = args.model
    max_requests = args.max_requests
    if args.multi_class:
        dataset_worker = MachineryHealthDatasetWorker(working_dir=working_dir, dbworker=db_worker,
                                                      max_requests=max_requests,
                                                      train_ratio=args.train_ratio, valid_ratio=args.valid_ratio,
                                                      model_type=model_type, model_name=args.model,
                                                      seed=args.seed)
    else:
        dataset_worker = MachineryHealthBinaryDatasetWorker(working_dir=working_dir, dbworker=db_worker,
                                                            max_requests=max_requests,
                                                            train_ratio=args.train_ratio,
                                                            valid_ratio=args.valid_ratio,
                                                            model_type=model_type, model_name=args.model,
                                                            seed=args.seed)
    dataset_worker.work()
