import sklearn
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import math
import pickle
import re
from rich import inspect
from tap import Tap
import logging
import threading
import clickhouse_connect
from typing import List, Dict, Tuple, Optional, Union, Any, Literal
import joblib
from tqdm import tqdm
tqdm.pandas()

HOME_DIR: str = '/home/ckchang/ApproxInfer'  # project home
DATA_HOME: str = os.path.join(HOME_DIR, 'data')  # data home
RESULTS_HOME: str = os.path.join(HOME_DIR, 'results')  # results home
LOG_DIR: str = os.path.join(HOME_DIR, 'logs')  # log home

SUPPORTED_AGGS = ['count', 'sum', 'avg', 'min', 'max', 'median', 'var', 'std']


def load_and_save_to_clickhouse(dataset_path, columns, table_name, sample=False, verbose=False):
    """ load all files from dataset_path and save to clickhouse table
    The filename is the timestamp, all the records in the file be tagged with the timestamp
    """
    dbconn = clickhouse_connect.get_client(
        host='localhost', port=0, username='default', password='', session_id=f'session_{table_name}')
    # create table if not exists
    columns_w_type = [f'{col} Float32' for col in columns]
    if not sample:
        dbconn.command(f'DROP TABLE IF EXISTS {table_name}')
        dbconn.command(
            f'CREATE TABLE IF NOT EXISTS {table_name} (timestamp DateTime, {", ".join(columns_w_type)}) ENGINE = MergeTree() ORDER BY timestamp')
        # get timezone info of the table's DateTime
        for filename in tqdm(os.listdir(dataset_path)):
            # conver filename with format '%Y.%m.%d.%H.%M.%S' to format '%Y-%m-%d %H:%M:%S'
            timestamp = pd.to_datetime(
                filename, format='%Y.%m.%d.%H.%M.%S').strftime('%Y-%m-%d %H:%M:%S')

            # insert data to clickhouse from file directly
            # generate command to insert data
            command = """
                    clickhouse-client \
                        --query \
                        "INSERT INTO {table_name} SELECT toDateTime('{timestamp}') as timestamp, {values} FROM input('{values_w_type}') FORMAT TSV" \
                         < {filepath}
                    """.format(table_name=table_name,
                               timestamp=timestamp,
                               values=', '.join(
                                   [f'{col} AS {col}' for col in columns]),
                               values_w_type=', '.join(columns_w_type),
                               filepath=os.path.join(dataset_path, filename))
            # print(command)
            os.system(command)
    else:
        table_name = f'{table_name}_w_samples'
        dbconn.command(
            f'CREATE TABLE IF NOT EXISTS {table_name} (row_id UInt32, timestamp DateTime, {", ".join(columns_w_type)}) ENGINE = MergeTree() PARTITION BY timestamp ORDER BY cityHash64(row_id) SAMPLE BY cityHash64(row_id)')
        dbconn.command(
            f'INSERT INTO {table_name} SELECT row_number() over (order by timestamp) as row_id, * FROM {table_name[:-len("_w_samples")]}')


if __name__ == "__main__":
    bearing_data_dir = os.path.join(DATA_HOME, 'bearing')
    bearing_1st_dir = os.path.join(
        bearing_data_dir, '1st_test', '1st_test')  # 2156 files
    bearing_2nd_dir = os.path.join(
        bearing_data_dir, '2nd_test', '2nd_test')  # 984 files
    bearing_3rd_dir = os.path.join(
        bearing_data_dir, '3rd_test', '4th_test', 'txt')  # 6324 files

    load_and_save_to_clickhouse(bearing_1st_dir, [
                                'B1X', 'B1Y', 'B2X', 'B2Y', 'B3X', 'B3Y', 'B4X', 'B4Y'], 'bearing')
    # load_and_save_to_clickhouse(bearing_1st_dir, ['B1X', 'B1Y', 'B2X', 'B2Y', 'B3X', 'B3Y', 'B4X', 'B4Y'], 'bearing_1st')
    # load_and_save_to_clickhouse(bearing_2nd_dir, ['B1', 'B2', 'B3', 'B4'], 'bearing_2nd')
    # load_and_save_to_clickhouse(bearing_3rd_dir, ['B1', 'B2', 'B3', 'B4'], 'bearing_3rd')

    # load_and_save_to_clickhouse(bearing_1st_dir, ['B1X', 'B1Y', 'B2X', 'B2Y', 'B3X', 'B3Y', 'B4X', 'B4Y'], 'bearing_1st', sample=True)
    # load_and_save_to_clickhouse(bearing_2nd_dir, ['B1', 'B2', 'B3', 'B4'], 'bearing_2nd', sample=True)
    # load_and_save_to_clickhouse(bearing_3rd_dir, ['B1', 'B2', 'B3', 'B4'], 'bearing_3rd', sample=True)
