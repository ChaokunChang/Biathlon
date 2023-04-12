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
from tap import Tap
import clickhouse_connect

from tqdm import tqdm
tqdm.pandas()

pandarallel.initialize(progress_bar=True)
# pandarallel.initialize(nb_workers=2)

# %%
HOME_DIR = '/home/ckchang/ApproxInfer'
data_dir = os.path.join(HOME_DIR, 'data/nyc_taxi_2015-07-01_2015-09-30')

# %%
sql_template_example = """
select {aggs} from trips 
where (pickup_datetime >= (toDateTime('{pickup_datetime}') - toIntervalHour({hours}))) 
AND (pickup_datetime < '{pickup_datetime}') AND (dropoff_datetime <= '{pickup_datetime}') 
AND (passenger_count = {passenger_count})
"""


class FeatureExtractor:
    def __init__(self, sql_template=sql_template_example,
                 interval_hours=1,
                 aggcols=['trip_distance', 'fare_amount',
                          'tip_amount', 'total_amount'],
                 aggops=['count', 'avg', 'sum', 'stddevPop',
                         'varPop', 'min', 'max', 'median'],
                 agg_prefixs=['count', 'mean', 'sum',
                              'std', 'var', 'min', 'max', 'median']
                 ) -> None:
        self.sql_template = sql_template
        self.interval_hours = interval_hours
        self.aggcols = aggcols
        self.aggops = aggops
        self.agg_prefixs = agg_prefixs
        self.agg_prefixs = [f'{x}_{interval_hours}h' for x in self.agg_prefixs]
        self.aggs = [f'{op}({col}) as {self.agg_prefixs[i]}_{col}' for i,
                     op in enumerate(aggops) for col in aggcols]
        self.sql_template = self.sql_template.replace(
            "{aggs}", ", ".join(self.aggs))
        self.hashid = hash(self.sql_template) + hash(time.time())

    def extract(self, x):
        sql = self.sql_template.format(
            pickup_datetime=x['pickup_datetime'], hours=self.interval_hours, passenger_count=x['passenger_count'])
        clt = clickhouse_connect.get_client(
            host='localhost', username='default', password='', session_id=f'session_{self.hashid}_{x["trip_id"]}')
        rows_df = clt.query_df(sql)
        rows_df['trip_id'] = x['trip_id']
        aggregations = rows_df.iloc[0]
        # print(f'aggregations={aggregations}')
        clt.close()
        del clt
        return aggregations

    def apply_on(self, df):
        st = time.time()
        features = df.parallel_apply(self.extract, axis=1)
        print(f'Elapsed time: {time.time() - st}')
        return features

# %%


def run_extraction(running_df, fn, **kwargs):
    st = time.time()
    feas = running_df.parallel_apply(fn, axis=1, **kwargs)
    print(f'Elapsed time: {time.time() - st}')
    return feas


# %%
if __name__ == '__main__':
    feature_dir = os.path.join(data_dir, 'features')
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    df = pd.read_csv(os.path.join(data_dir, 'requests_08-01_08-15.csv'))
    df.head()

    # extract features and save to csv
    # sample 10000 from df
    # df = df.sample(n=10000, random_state=0)
    extractor_1 = FeatureExtractor(interval_hours=1)
    agg_feas_1 = extractor_1.apply_on(df)
    agg_feas_1.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_1.csv'), index=False)

    extractor_2 = FeatureExtractor(interval_hours=24)
    agg_feas_2 = extractor_2.apply_on(df)
    agg_feas_2.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_2.csv'), index=False)

    extractor_3 = FeatureExtractor(interval_hours=24*7)
    agg_feas_3 = extractor_3.apply_on(df)
    agg_feas_3.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_3.csv'), index=False)

    # merge three agg features on trip_id
    all_feas = df.merge(agg_feas_1, on='trip_id').merge(
        agg_feas_2, on='trip_id').merge(agg_feas_3, on='trip_id')
    all_feas.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.feas.csv'), index=False)
