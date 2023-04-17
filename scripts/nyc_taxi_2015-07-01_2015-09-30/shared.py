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
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

from pandarallel import pandarallel

from tqdm import tqdm
tqdm.pandas()

pandarallel.initialize(progress_bar=True)
# pandarallel.initialize(nb_workers=2)

HOME_DIR: str = '/home/ckchang/ApproxInfer'  # project home
DATA_HOME: str = os.path.join(HOME_DIR, 'data')  # data home
LOG_DIR: str = os.path.join(HOME_DIR, 'logs')  # log home

sql_template_example = """
SELECT 
count(*) as count_1h, 
avg(trip_duration) as avg_trip_duration_1h, 
avg(trip_distance) as avg_trip_distance_1h, 
avg(fare_amount) as avg_fare_amount_1h, 
avg(tip_amount) as avg_tip_amount_1h, 
stddevPop(trip_duration) as std_trip_duration_1h, 
stddevPop(trip_distance) as std_trip_distance_1h, 
stddevPop(fare_amount) as std_fare_amount_1h, 
stddevPop(tip_amount) as std_tip_amount_1h, 
min(trip_duration) as min_trip_duration_1h, 
min(trip_distance) as min_trip_distance_1h, 
min(fare_amount) as min_fare_amount_1h, 
min(tip_amount) as min_tip_amount_1h, 
max(trip_duration) as max_trip_duration_1h, 
max(trip_distance) as max_trip_distance_1h, 
max(fare_amount) as max_fare_amount_1h, 
max(tip_amount) as max_tip_amount_1h, 
median(trip_duration) as median_trip_duration_1h, 
median(trip_distance) as median_trip_distance_1h, 
median(fare_amount) as median_fare_amount_1h, 
median(tip_amount) as median_tip_amount_1h 
FROM trips 
WHERE (pickup_datetime >= (toDateTime('{pickup_datetime}') - toIntervalHour(1))) 
AND (pickup_datetime < '{pickup_datetime}') 
AND (dropoff_datetime <= '{pickup_datetime}') 
AND (passenger_count = {passenger_count}) 
AND (pickup_ntaname = '{pickup_ntaname}') 
"""


def load_sql_templates(filename: str):
    with open(filename, 'r') as f:
        sql_templates = f.read().split(';')
    # remove the comments in sql_template
    sql_templates = [re.sub(r'--.*', '', sql_template)
                     for sql_template in sql_templates]
    sql_templates = [sql_template.strip() for sql_template in sql_templates]
    sql_templates = [
        sql_template for sql_template in sql_templates if sql_template != '']
    return sql_templates


def approximation_rewrite(sql_template: str, sample: float):
    if sample == 0:
        return sql_template
    else:
        # repalce the table with table_w_samples SAMPLE {sample}
        assert sample > 0 and sample <= 1
        assert 'SAMPLE' not in sql_template
        assert 'FROM' in sql_template
        template = re.sub(
            r'FROM\s+(\w+)', fr'FROM \1_w_samples SAMPLE {sample}', sql_template)
        return template


class SimpleParser(Tap):
    data_dir: str = os.path.join(
        DATA_HOME, 'nyc_taxi_2015-07-01_2015-09-30')  # data dir
    req_src: str = 'requests_08-01_08-15_sample10000.csv'  # request source
    label_src: str = 'labels_08-01_08-15.csv'  # label source
    task: str = 'fare_prediction'  # task name
    outdir: str = os.path.join(HOME_DIR, 'results')  # output directory

    sql_template: str = sql_template_example  # sql template
    sql_templates_file: str = None  # sql templates file
    ffile_prefix = 'features'

    sample: float = 0  # sample rate of sql query. default 0 means disable sampling
    config: str = None  # config file

    random_state: int = 42  # random state

    model_test_size: int = 0.3  # train split for model training

    def process_args(self) -> None:
        self.task_dir = os.path.join(self.data_dir, self.task)
        # check existence of request source and label source file
        self.req_src = os.path.join(self.data_dir, self.req_src)
        self.label_src = os.path.join(self.data_dir, self.label_src)
        assert os.path.exists(self.req_src), f'{self.req_src} does not exist'
        assert os.path.exists(
            self.label_src), f'{self.label_src} does not exist'

        assert self.sql_template is not None or self.sql_templates_file is not None, 'sql_template or sql_templates_file must be specified'
        if self.sql_templates_file is not None:
            self.sql_templates = load_sql_templates(self.sql_templates_file)
        else:
            self.sql_templates = [self.sql_template]

        self.feature_dir = os.path.join(self.task_dir, 'features')
        self.outdir = os.path.join(self.outdir, self.task)
        if self.sample > 0:
            # args.sample means run query apprximately with args.sample rate
            # we need to rewrite the query to make it an approximate query
            self.sql_templates = [approximation_rewrite(
                sql_template, self.sample) for sql_template in self.sql_templates]
            self.feature_dir = os.path.join(
                self.feature_dir, f'sample_{self.sample}')
            self.outdir = os.path.join(self.outdir, f'sample_{self.sample}')

        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
