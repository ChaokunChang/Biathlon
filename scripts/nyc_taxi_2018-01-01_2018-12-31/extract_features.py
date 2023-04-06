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


def extrac_all(in_df, idx_col='pickup_datetime', groupby_col='vendor_id', window='1H', agg_cols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    sorted_df = in_df.set_index(idx_col).sort_index()
    gby_df = sorted_df.groupby(groupby_col)
    tids = gby_df.apply(lambda x: x['tid'])
    gby_df = gby_df[agg_cols]
    gby_sum = gby_df.rolling(window).sum().add_prefix('sum_')
    gby_count = gby_df.rolling(window).count().add_prefix('count_')
    gby_mean = gby_df.rolling(window).mean().add_prefix('mean_')
    gby_median = gby_df.rolling(window).median().add_prefix('median_')
    gby_min = gby_df.rolling(window).min().add_prefix('min_')
    gby_max = gby_df.rolling(window).max().add_prefix('max_')
    gby_var = gby_df.rolling(window).var().add_prefix('var_')

    # combine all results into single dataframe
    df_all = pd.concat([tids, gby_sum, gby_count, gby_mean,
                       gby_median, gby_min, gby_max, gby_var], axis=1)
    return df_all


# load ApproxInfer/data/nyc_taxi_2018-01-01_2018-12-31/cleaned_nyc_taxi_data_2018.csv
data_dir = '/home/ckchang/ApproxInfer/data/nyc_taxi_2018-01-01_2018-12-31'
src_data = os.path.join(data_dir, 'cleaned_nyc_taxi_data_2018.csv')
df = pd.read_csv(src_data)
# df.rename(columns={'Unnamed: 0': 'tid'}, inplace=True)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

feature_dir = os.path.join(data_dir, 'features')
if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)

extrac_all(df, groupby_col='vendor_id', window='1H').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv'))
extrac_all(df, groupby_col='vendor_id', window='1D').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1day.csv'))
extrac_all(df, groupby_col='vendor_id', window='7D').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1week.csv'))

extrac_all(df, groupby_col='passenger_count', window='1H').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1hour.csv'))
extrac_all(df, groupby_col='passenger_count', window='1D').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1day.csv'))
extrac_all(df, groupby_col='passenger_count', window='7D').to_csv(
    os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1week.csv'))
