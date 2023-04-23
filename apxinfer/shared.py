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
from sklearn.linear_model import LinearRegression, LogisticRegression
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


def save_features(features: pd.DataFrame, feature_dir: str, output_name: str = 'features.csv'):
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    features.to_csv(os.path.join(feature_dir, output_name), index=False)
    return None


def load_features(feature_dir: str, input_name: str = 'features.csv') -> pd.DataFrame:
    features = pd.read_csv(os.path.join(feature_dir, input_name))
    return features


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
        # FROM could also be from
        template = re.sub(
            r'from\s+(\w+)', fr'from \1_w_samples SAMPLE {sample}', template)
        return template


def get_query_feature_map(templates: list[str]):
    query_feature_map = {}
    feature_query_map = {}
    for i, template in enumerate(templates):
        # get the feature name, which is the last word after 'as' or 'AS'
        features = re.findall(r'[as|AS]\s+(\w+)', template)
        query_feature_map[i] = features
        for feature in features:
            assert feature not in feature_query_map
            feature_query_map[feature] = i
    return query_feature_map, feature_query_map


class SQLTemplates:
    def __init__(self) -> None:
        self.templates: list[str] = None
        self.q2f: dict = None
        self.f2q: dict = None
        pass

    def from_file(self, input_file: str):
        with open(input_file, 'r') as f:
            self.templates = f.read().split(';')
        # remove the comments in sql_template
        self.templates = [re.sub(r'--.*', '', sql_template)
                          for sql_template in self.templates]
        self.templates = [sql_template.replace(r'\s+', ' ').strip()
                          for sql_template in self.templates]
        self.templates = [
            sql_template for sql_template in self.templates if sql_template != '']
        self.q2f, self.f2q = get_query_feature_map(self.templates)
        return self

    def make_apx_queries(self, samples: list[float]):
        if samples is None or len(samples) == 0:
            return self
        else:
            # repalce the table with table_w_samples SAMPLE {sample}
            for i, sample in enumerate(samples):
                assert sample > 0 and sample <= 1
                assert 'SAMPLE' not in self.templates[i]
                self.templates[i] = re.sub(
                    r'[FROM|from]\s+(\w+)', fr'FROM \1_w_samples SAMPLE {sample}', self.templates[i])
            return self


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

    fcols: str = None  # feature columns

    model_test_size: int = 0.3  # train split for model training
    split_shuffle: bool = False  # shuffle data before split
    model_name: str = 'lgbm'  # model name
    model_type: Literal['regressor', 'classifier'] = 'regressor'  # model type
    multi_class: bool = False  # multi class classification

    apx_training: bool = False  # whether to use approximation model

    topk_features: int = 10  # top k features to show

    def process_args(self) -> None:
        self.data_dir = os.path.join(DATA_HOME, self.data)
        self.task_dir = os.path.join(self.data_dir, self.task)
        self.req_src = os.path.join(self.task_dir, 'requests.csv')
        self.label_src = os.path.join(self.task_dir, 'labels.csv')

        if self.sql_templates_file is not None:
            self.sql_templates = load_sql_templates(self.sql_templates_file)

        if self.sample > 0:
            # we need to rewrite the query to make it an approximate query
            self.sql_templates = [approximation_rewrite(
                sql_template, self.sample) for sql_template in self.sql_templates]

        self.feature_dir = os.path.join(self.task_dir, 'features') if self.sample == 0 else os.path.join(
            self.task_dir, 'features', f'sample_{self.sample}')

        if self.fcols is not None:
            if self.fcols.endswith('feature_importance.csv'):
                # fcols is filename to feature_importance.csv
                # we select topk features from feature_importance.csv
                fimps = pd.read_csv(self.fcols)
                self.fcols = fimps.sort_values(by='importance', ascending=False).head(
                    self.topk_features)['fname'].values.tolist()
                self.experiment_dir = os.path.join(
                    RESULTS_HOME, self.data, self.task, f'{self.model_name}_top{self.topk_features}')
            else:
                # fcols is a list of feature names splited by ,
                self.fcols = self.fcols.split(',')
                self.experiment_dir = os.path.join(
                    RESULTS_HOME, self.data, self.task, f'{self.model_name}_num{len(self.fcols)}')
            assert len(self.fcols) > 0, f'fcols is empty'
        else:
            self.experiment_dir = os.path.join(
                RESULTS_HOME, self.data, self.task, self.model_name)

        self.pipelines_dir = os.path.join(self.experiment_dir, 'pipelines')
        if self.apx_training:
            self.pipelines_dir = os.path.join(
                self.pipelines_dir, f'sample_{self.sample}')
        # self.pipeline_fpath = os.path.join(self.pipelines_dir, 'pipeline.pkl')

        self.evals_dir = os.path.join(self.experiment_dir, 'evals') if self.sample == 0 else os.path.join(
            self.experiment_dir, 'evals', f'sample_{self.sample}')
        # self.outdir = self.experiment_dir if self.sample > 0 else os.path.join(self.experiment_dir, f'sample_{self.sample}')

        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.pipelines_dir, exist_ok=True)
        os.makedirs(self.evals_dir, exist_ok=True)
