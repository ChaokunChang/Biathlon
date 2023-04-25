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


def is_agg_feature(fname: str):
    return fname.find('_') > 0 and fname.split('_', 1)[0] in SUPPORTED_AGGS


def feature_dtype_inference(df: pd.DataFrame, keycol: str, target: str):
    num_features = []
    cat_features = []
    dt_features = []
    for col in df.columns:
        if col == keycol or col == target:
            continue
        dtype = df[col].dtype
        if dtype == 'object':
            if 'datetime' in col:
                dt_features.append(col)
            else:
                cat_features.append(col)
        elif dtype == 'int64' or dtype == 'float64':
            # for low cardinality, treat as categorical
            if df[col].nunique() < 10:
                # if the col is aggregation feature, treat it as numerical and warn
                if is_agg_feature(col):
                    print(
                        f'WARNING: col={col} is low cardinality, but it is an aggregation feature, treat it as numerical')
                    num_features.append(col)
                else:
                    cat_features.append(col)
            else:
                num_features.append(col)
        else:
            raise Exception(f'Unknown dtype={dtype} for col={col}')
    # print(f'feature_type_inference: num_features={num_features}')
    # print(f'feature_type_inference: dt_features={dt_features}')
    print(f'feature_type_inference: cat_features={cat_features}')
    return num_features, cat_features, dt_features


def feature_ctype_inference(cols: list, keycol: str, target: str):
    agg_features = []
    nonagg_features = []
    for col in cols:
        if col == keycol or col == target:
            continue
        # if col starts with '{agg}_', where agg is in [count, avg, sum, var, std, min, max, median], it is an aggregated feature
        if is_agg_feature(col):
            agg_features.append(col)
        else:
            nonagg_features.append(col)
    # print(f'feature_type_inference: agg_features={agg_features}')
    print(f'feature_type_inference: nonagg_features={nonagg_features}')
    return agg_features, nonagg_features


def feature_type_inference(df: pd.DataFrame, keycol: str, target: str):
    # print(f'feature_type_inference: df.columns={df.columns}')
    typed_features = {'num_features': [], 'cat_features': [], 'dt_features': [],
                      'agg_features': [], 'nonagg_features': [],
                      'keycol': keycol, 'target': target,
                      }
    num_features, cat_features, dt_features = feature_dtype_inference(
        df, keycol, target)
    typed_features['num_features'] = num_features
    typed_features['cat_features'] = cat_features
    typed_features['dt_features'] = dt_features
    agg_features, nonagg_features = feature_ctype_inference(
        df.columns.to_list(), keycol, target)
    typed_features['agg_features'] = agg_features
    typed_features['nonagg_features'] = nonagg_features
    return typed_features


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


def save_to_csv(features: pd.DataFrame, feature_dir: str, output_name: str = 'features.csv'):
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    features.to_csv(os.path.join(feature_dir, output_name), index=False)
    return None


def load_from_csv(feature_dir: str, input_name: str = 'features.csv') -> pd.DataFrame:
    features = pd.read_csv(os.path.join(feature_dir, input_name))
    return features


def approximation_rewrite(sql_template: str, sample: float = None):
    if sample is None or sample == 0:
        return sql_template
    else:
        assert sample > 0 and sample <= 1
        assert 'SAMPLE' not in sql_template, f'sql_template={sql_template} already has SAMPLE clause'
        # repalce the 'from table' and 'FROM table' with 'FROM table_w_samples SAMPLE {sample}'
        template = re.sub(
            r'FROM\s+(\w+)', fr'FROM \1_w_samples SAMPLE {sample}', sql_template)
        template = re.sub(
            r'from\s+(\w+)', fr'from \1_w_samples SAMPLE {sample}', template)

        scale = 1 / sample
        # replace the count(*) with count(*) * {scale}
        template = re.sub(
            r'count\(\*\)', fr'count(*) * {scale}', template)
        # replace the count(col) with count(col) * {scale}
        template = re.sub(
            r'count\((\w+)\)', fr'count(\1) * {scale}', template)
        # replace the sum(col) with sum(col) * {scale}
        template = re.sub(
            r'sum\((\w+)\)', fr'sum(\1) * {scale}', template)
        # replace the varPop(col) with varSamp(col)
        template = re.sub(
            r'varPop\((\w+)\)', fr'varSamp(\1)', template)
        # replace the stddevPop(col) with stddevSamp(col)
        template = re.sub(
            r'stddevPop\((\w+)\)', fr'stddevSamp(\1)', template)

        return template


def aggfname_rewrite(template: str, qcols: list) -> str:
    """ rewrite the sql template, such only fname in qcols will be returned
    the sql template will be like select fop as fname, ... from table
    we only keep the col that is in qcols
    """
    if len(qcols) == 0:
        return ""

    assert 'select ' in template or 'SELECT ' in template
    template = template.replace('select ', 'SELECT ')
    template = template.replace('from ', 'FROM ')
    select_ = template.split('SELECT ')[1].split('FROM ')[0]
    if (len(template.split('SELECT ')[1].split('FROM ')) > 1):
        from_ = " FROM " + template.split('SELECT ')[1].split('FROM ')[1]
    else:
        from_ = ""

    fop_as_fnames = [fop_as_fname.strip()
                     for fop_as_fname in select_.split(',')]

    new_fop_as_fnames = []
    for i, fop_as_fname in enumerate(fop_as_fnames):
        fop, fname = fop_as_fname.split(' as ')
        fop, fname = fop.strip(), fname.strip()
        if fname in qcols:
            new_fop_as_fnames.append(fop_as_fname)

    new_select = 'SELECT ' + ', '.join(new_fop_as_fnames)
    new_template = new_select + from_

    return new_template.strip()


def compute_valid_qcols(qtemplates: list[str], valid_fcols: list[str]) -> list:
    q2f_map, f2q_map = get_query_feature_map(qtemplates)
    qcols = [[] for _ in range(len(qtemplates))]
    for fcol in valid_fcols:
        qid = f2q_map[fcol]
        qcols[qid].append(fcol)
    return qcols


def compuet_query_importance(qtemplates: list[str], fcols: list[str], fimps: list[float]) -> list[float]:
    q2f_map, f2q_map = get_query_feature_map(qtemplates)
    qimps = np.zeros(len(qtemplates))
    for fcol, fimp in zip(fcols, fimps):
        qid = f2q_map[fcol]
        qimps[qid] += fimp
    return qimps.tolist()


def compuet_query_agg_importance(qtemplates: list[str], fcols: list[str], fimps: list[float]) -> list[float]:
    q2f_map, f2q_map = get_query_feature_map(qtemplates)
    qimps = np.zeros(len(qtemplates))
    for fcol, fimp in zip(fcols, fimps):
        qid = f2q_map[fcol]
        if is_agg_feature(fcol):
            qimps[qid] += fimp
    return qimps.tolist()


def get_query_feature_map(templates: list[str]):
    query_feature_map = {}
    feature_query_map = {}
    for i, template in enumerate(templates):
        # get the feature name, which is the last word after 'as' or 'AS'
        features = re.findall(r'\s+as\s+(\w+)', template)
        features += re.findall(r'\s+AS\s+(\w+)', template)
        # print(features)
        query_feature_map[i] = features
        for feature in features:
            assert feature not in feature_query_map, f'{feature} is duplicated'
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

    def apx_transform(self, samples: list[float] | float = None):
        if samples is None:
            return self
        if isinstance(samples, float):
            samples = [samples] * len(self.templates)
        if len(samples) > 0:
            # repalce the table with table_w_samples SAMPLE {sample}
            for i, sample in enumerate(samples):
                self.templates[i] = approximation_rewrite(
                    self.templates[i], sample)
        return self


def to_sample(string: str) -> Union[float, str]:
    # if the string represent a float, return the float
    # otherwise return the string directly
    try:
        return float(string)
    except:
        return string


class SimpleParser(Tap):
    data: str = 'nyc_taxi_2015-07-01_2015-09-30'  # data name
    task: str = 'fare_prediction_2015-08-01_2015-08-15_10000'  # task name
    keycol: str = 'trip_id'  # key column
    target: str = 'fare_amount'  # target column
    sort_by: str = 'pickup_datetime'  # sort by column

    # sql_templates: str = [sql_template_example]  # sql template
    templator: SQLTemplates = None  # sql template
    sql_templates_file: str = None  # sql templates file
    ffile_prefix = 'features'

    # sample rate of sql query. default 0 means disable sampling
    sample: Union[float, str] = None
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

    def configure(self):
        self.add_argument('--sample', type=to_sample)

    def process_args(self) -> None:
        self.data_dir = os.path.join(DATA_HOME, self.data)
        self.task_dir = os.path.join(self.data_dir, self.task)
        self.req_src = os.path.join(self.task_dir, 'requests.csv')
        self.label_src = os.path.join(self.task_dir, 'labels.csv')

        assert self.sql_templates_file is not None, 'sql_templates_file is required'
        self.templator = SQLTemplates().from_file(self.sql_templates_file)
        self.sql_templates = [t for t in self.templator.templates]
        if isinstance(self.sample, float):
            self.templator = self.templator.apx_transform(self.sample)

        self.feature_dir = os.path.join(self.task_dir, 'features') if self.sample is None else os.path.join(
            self.task_dir, 'features', f'sample_{self.sample}')

        if self.fcols is not None:
            if self.fcols.endswith('feature_importance.csv'):
                # fcols is filename to feature_importance.csv
                # we select topk features from feature_importance.csv
                fimps_df = pd.read_csv(self.fcols)
                topkfimps = fimps_df.sort_values(
                    by='importance', ascending=False).head(self.topk_features)
                self.fcols = topkfimps['fname'].values.tolist()
                # self.fimps = topkfimps['importance'].values.tolist()
                self.experiment_dir = os.path.join(
                    RESULTS_HOME, self.data, self.task, f'{self.model_name}_top{self.topk_features}')
                self.feature_dir = os.path.join(
                    self.feature_dir, f'{self.model_name}_top{self.topk_features}')
            else:
                # fcols is a list of feature names and imps splited by ,
                # each element will be fname:fimp
                self.fcols_imps = self.fcols.split(',')
                self.fcols = [fcol_imp.split(':')[0]
                              for fcol_imp in self.fcols_imps]
                # self.fimps = [float(fcol_imp.split(':')[1]) if len(
                #     fcol_imp.split(':')) > 1 else 0.0 for fcol_imp in self.fcols_imps]
                self.experiment_dir = os.path.join(
                    RESULTS_HOME, self.data, self.task, f'{self.model_name}_num{len(self.fcols)}')
                self.feature_dir = os.path.join(
                    self.feature_dir, f'{self.model_name}_num{len(self.fcols)}')
            assert len(self.fcols) > 0, f'fcols is empty'
        else:
            self.experiment_dir = os.path.join(
                RESULTS_HOME, self.data, self.task, self.model_name)

        self.evals_dir = os.path.join(self.experiment_dir, 'evals') if self.sample is None else os.path.join(
            self.experiment_dir, 'evals', f'sample_{self.sample}')

        # pipelines_dir stores the built pipelines
        # for pipeline built with exact features, store in pipelines_dir
        self.pipelines_dir = os.path.join(self.experiment_dir, 'pipelines')
        if self.apx_training:
            # for pipeline built with approximate features, store in pipelines_dir/sample_{sample}
            assert self.sample is not None, 'sample is required for apx_training'
            self.pipelines_dir = os.path.join(
                self.pipelines_dir, f'sample_{self.sample}')

        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.pipelines_dir, exist_ok=True)
        os.makedirs(self.evals_dir, exist_ok=True)


if __name__ == "__main__":
    args = SimpleParser().parse_args()
    print(args)
