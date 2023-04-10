# %%
from pandarallel import pandarallel
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
from rich import inspect
import clickhouse_connect

from tqdm import tqdm
tqdm.pandas()

pandarallel.initialize(progress_bar=True)
# pandarallel.initialize(nb_workers=2)

# %%
client = clickhouse_connect.get_client(
    host='localhost', username='default', password='')
HOME_DIR = '/home/ckchang/ApproxInfer'
data_dir = os.path.join(HOME_DIR, 'data/nyc_taxi_2015-07-01_2015-09-30')
feature_dir = os.path.join(data_dir, 'features')
# %%
df = pd.read_csv(os.path.join(data_dir, 'requests_08-01_08-08.csv'))
df.head()
# %%


def extract_features_0(x, aggcols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    sql_template = """
    select {aggcols} from trips 
    where (pickup_datetime >= (toDateTime('{pickup_datetime}') - toIntervalHour(1))) AND (pickup_datetime < '{pickup_datetime}') AND (dropoff_datetime <= '{pickup_datetime}') 
    AND (passenger_count = {passenger_count})
    """

    sql = sql_template.format(aggcols=','.join(
        aggcols), pickup_datetime=x['pickup_datetime'], passenger_count=x['passenger_count'])
    # print(f'sql={sql}')
    # rows_df = client.query_df(sql)
    clt = clickhouse_connect.get_client(
        host='localhost', username='default', password='', session_id=f'session_extract_features_1_{x["trip_id"]}')
    # print(f'clt.session_id: {clt.params["session_id"]}')
    rows_df = clt.query_df(sql)
    # compute aggregation on rows
    agg_count = rows_df.count().add_prefix('count_')
    agg_mean = rows_df.mean().add_prefix('mean_')
    agg_sum = rows_df.sum().add_prefix('sum_')
    agg_std = rows_df.std().add_prefix('std_')
    agg_std = rows_df.var().add_prefix('var_')
    agg_min = rows_df.min().add_prefix('min_')
    agg_max = rows_df.max().add_prefix('max_')
    agg_median = rows_df.median().add_prefix('median_')
    aggregations = pd.concat(
        [agg_count, agg_mean, agg_sum, agg_std, agg_min, agg_max, agg_median])
    # print(f'aggregations={aggregations}')
    clt.close()
    return aggregations

# %%


def extract_features_1(x, aggcols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    sql_template = """
    select {aggs} from trips 
    where (pickup_datetime >= (toDateTime('{pickup_datetime}') - toIntervalHour(1))) AND (pickup_datetime < '{pickup_datetime}') AND (dropoff_datetime <= '{pickup_datetime}') 
    AND (passenger_count = {passenger_count})
    """
    aggops = ['count', 'avg', 'sum', 'stddevPop',
              'varPop', 'min', 'max', 'median']
    agg_prefixs = ['count', 'mean', 'sum',
                   'std', 'var', 'min', 'max', 'median']
    aggs = [f'{op}({col}) as {agg_prefixs[i]}_{col}' for i,
            op in enumerate(aggops) for col in aggcols]
    sql = sql_template.format(aggs=', '.join(
        aggs), pickup_datetime=x['pickup_datetime'], passenger_count=x['passenger_count'])
    # print(f'sql={sql}')
    # rows_df = client.query_df(sql)
    clt = clickhouse_connect.get_client(
        host='localhost', username='default', password='', session_id=f'session_extract_features_2_{x["trip_id"]}')
    # print(f'clt.session_id: {clt.params["session_id"]}')
    rows_df = clt.query_df(sql)
    # compute aggregation on rows
    aggregations = rows_df
    # print(f'aggregations={aggregations}')
    clt.close()
    return aggregations

# %%


def run_extraction(running_df=df.iloc[:1000], fn=extract_features_0):
    st = time.time()
    feas = running_df.parallel_apply(fn, axis=1)
    feas = pd.concat([running_df, feas], axis=1)
    print(f'Elapsed time: {time.time() - st}')
    return feas


# %%
all_feas = run_extraction(df, fn=extract_features_1)
# save to csv
all_feas.to_csv(os.path.join(
    feature_dir, 'requests_08-01_08-08.feas.csv'), index=False)
# %%
