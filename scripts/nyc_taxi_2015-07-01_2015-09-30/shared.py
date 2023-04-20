from shap import TreeExplainer, KernelExplainer, LinearExplainer, DeepExplainer
import shap
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics, pipeline, set_config, tree
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


def load_requests(filename: str):
    df = pd.read_csv(filename)
    return df


def save_cat_vocab(req_path, cat_cols=None):
    reqs: pd.DataFrame = pd.read_csv(req_path)
    req_dir = os.path.dirname(req_path)
    filename = os.path.basename(req_path)
    filename_noext = os.path.splitext(filename)[0]

    if cat_cols is not None:
        for col in cat_cols:
            # there could be nan in the categorical columns, set them as 'nan'
            all_cats = sorted(reqs[col].fillna('nan').unique().tolist())
            # save cats to file
            with open(os.path.join(req_dir, f'{filename_noext}_{col}_cats.txt'), 'w') as f:
                f.write('\n'.join(all_cats))
        return all_cats
    else:
        return None


def sample_requests(req_path, sample=10000, cat_cols=None):
    reqs: pd.DataFrame = pd.read_csv(req_path)
    req_dir = os.path.dirname(req_path)
    filename = os.path.basename(req_path)
    filename_noext = os.path.splitext(filename)[0]
    filename_ext = os.path.splitext(filename)[1]
    reqs_w_samples = reqs.sample(sample, random_state=0)
    reqs_w_samples.to_csv(os.path.join(
        req_dir, f'{filename_noext}_sample{sample}{filename_ext}'), index=False)
    return reqs_w_samples


def save_features(features: pd.DataFrame, feature_dir: str, output_name: str = 'features.csv'):
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    features.to_csv(os.path.join(feature_dir, output_name), index=False)
    return None


def approximation_rewrite(sql_template: str, sample: float):
    if sample == 0:
        return sql_template
    else:
        # repalce the table with table_w_samples SAMPLE {sample}
        assert sample > 0 and sample <= 1
        assert 'SAMPLE' not in sql_template
        # assert 'FROM' in sql_template
        template = re.sub(
            r'FROM\s+(\w+)', fr'FROM \1_w_samples SAMPLE {sample}', sql_template)
        return template


class SimpleParser(Tap):
    data: str = 'nyc_taxi_2015-07-01_2015-09-30'  # data name
    task: str = 'fare_prediction_2015-08-01_2015-08-15_10000'  # task name
    keycol: str = 'trip_id'  # key column
    target: str = 'fare_amount'  # target column
    sort_by: str = 'pickup_datetime'  # sort by column

    sql_templates: str = [sql_template_example]  # sql template
    sql_templates_file: str = None  # sql templates file
    ffile_prefix = 'features'

    sample: float = 0  # sample rate of sql query. default 0 means disable sampling
    config: str = None  # config file

    random_state: int = 42  # random state

    model_test_size: int = 0.3  # train split for model training
    split_shuffle: bool = False  # shuffle data before split
    model_name: str = 'lgbm'  # model name
    model_type: str = 'regression'  # model type

    def process_args(self) -> None:
        self.data_dir = os.path.join(DATA_HOME, self.data)
        self.task_dir = os.path.join(self.data_dir, self.task)
        self.req_src = os.path.join(self.task_dir, 'requests.csv')
        self.label_src = os.path.join(self.task_dir, 'labels.csv')
        self.feature_dir = os.path.join(self.task_dir, 'features')

        self.outdir_base = os.path.join(
            RESULTS_HOME, self.task, self.model_name)
        self.pipeline_fpath = os.path.join(self.outdir_base, 'pipeline.pkl')

        if self.sql_templates_file is not None:
            self.sql_templates = load_sql_templates(self.sql_templates_file)

        self.outdir = self.outdir_base
        if self.sample > 0:
            # args.sample means run query apprximately with args.sample rate
            # we need to rewrite the query to make it an approximate query
            self.sql_templates = [approximation_rewrite(
                sql_template, self.sample) for sql_template in self.sql_templates]
            self.feature_dir = os.path.join(
                self.feature_dir, f'sample_{self.sample}')
            self.outdir = os.path.join(
                self.outdir, f'sample_{self.sample}')

        assert os.path.exists(self.req_src), f'{self.req_src} does not exist'
        assert os.path.exists(
            self.label_src), f'{self.label_src} does not exist'
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)
