import dask.dataframe as dd
import dask
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
# from parallel_pandas import ParallelPandas
# ParallelPandas.initialize(n_cpu=8, split_factor=4, disable_pr_bar=False)
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


def extract_all(in_df, idx_col='pickup_datetime', groupby_col='vendor_id', window='1H', agg_cols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    # sorted_df = in_df.set_index(idx_col).sort_index()
    # gby_df = sorted_df.groupby(groupby_col)
    gby_df = in_df.groupby(groupby_col)
    def group_func(x):
        x = x.set_index(idx_col)
        x.sort_values(idx_col, inplace=True)
        st = time.time()
        x_sum = x[agg_cols].rolling(window).sum().add_prefix('sum_')
        x_mean = x[agg_cols].rolling(window).mean().add_prefix('mean_')
        x_median = x[agg_cols].rolling(window).median().add_prefix('median_')
        x_min = x[agg_cols].rolling(window).min().add_prefix('min_')
        x_max = x[agg_cols].rolling(window).max().add_prefix('max_')
        x_var = x[agg_cols].rolling(window).var().add_prefix('var_')
        et = time.time()
        print('agg Time elapsed: {} seconds'.format(et - st))
        return pd.concat([x_sum, x_mean, x_median, x_min, x_max, x_var], axis=1)
    st = time.time()
    gby_sum = gby_df.apply(group_func)
    et = time.time()
    print('agg in original Time elapsed: {} seconds'.format(et - st))
    # gby_count = gby_df.rolling(window).count().add_prefix('count_')
    # gby_mean = gby_df.rolling(window).mean().add_prefix('mean_')
    # gby_median = gby_df.rolling(window).median().add_prefix('median_')
    # gby_min = gby_df.rolling(window).min().add_prefix('min_')
    # gby_max = gby_df.rolling(window).max().add_prefix('max_')
    # gby_var = gby_df.rolling(window).var().add_prefix('var_')

    # tids = gby_df.apply(lambda x: x['tid'])

    # combine all results into single dataframe
    # df_all = pd.concat([tids, gby_sum, gby_count, gby_mean,
    #                    gby_median, gby_min, gby_max, gby_var], axis=1)
    df_all = gby_sum
    # print(f'df_all: {df_all.reset_index()}')
    # return df_all.reset_index().set_index('pickup_datetime').drop(groupby_col, axis=1).sort_index()
    return df_all.reset_index().set_index('pickup_datetime').sort_index()

def extract_all_pby_pandasparallel(in_df, idx_col='pickup_datetime', groupby_col='vendor_id', window='1H', agg_cols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    # sorted_df = in_df.set_index(idx_col).sort_index()
    # gby_df = sorted_df.groupby(groupby_col)
    gby_df = in_df.groupby(groupby_col)
    def group_func(x):
        x = x.set_index(idx_col)
        x.sort_values(idx_col, inplace=True)
        st = time.time()
        x_sum = x[agg_cols].rolling(window).sum().add_prefix('sum_')
        x_mean = x[agg_cols].rolling(window).mean().add_prefix('mean_')
        x_median = x[agg_cols].rolling(window).median().add_prefix('median_')
        x_min = x[agg_cols].rolling(window).min().add_prefix('min_')
        x_max = x[agg_cols].rolling(window).max().add_prefix('max_')
        x_var = x[agg_cols].rolling(window).var().add_prefix('var_')
        et = time.time()
        print('agg in parallel Time elapsed: {} seconds'.format(et - st))
        return pd.concat([x_sum, x_mean, x_median, x_min, x_max, x_var], axis=1)
    st = time.time()
    gby_sum = gby_df.parallel_apply(group_func)
    et = time.time()
    print('agg in original Time elapsed: {} seconds'.format(et - st))
    # gby_count = gby_df.rolling(window).count().add_prefix('count_')
    # gby_mean = gby_df.rolling(window).mean().add_prefix('mean_')
    # gby_median = gby_df.rolling(window).median().add_prefix('median_')
    # gby_min = gby_df.rolling(window).min().add_prefix('min_')
    # gby_max = gby_df.rolling(window).max().add_prefix('max_')
    # gby_var = gby_df.rolling(window).var().add_prefix('var_')

    # tids = gby_df.apply(lambda x: x['tid'])

    # combine all results into single dataframe
    # df_all = pd.concat([tids, gby_sum, gby_count, gby_mean,
    #                    gby_median, gby_min, gby_max, gby_var], axis=1)
    df_all = gby_sum
    # print(f'df_all: {df_all.reset_index()}')
    # return df_all.reset_index().set_index('pickup_datetime').drop(groupby_col, axis=1).sort_index()
    return df_all.reset_index().set_index('pickup_datetime').sort_index()


def extract_all_parallel(in_df, idx_col='pickup_datetime', groupby_col='vendor_id', window='1H', agg_cols=['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']):
    ddf:dd.DataFrame = dd.from_pandas(in_df, npartitions=4)
    gby_ddf = ddf.groupby(groupby_col)

    def group_func(x):
        x = x.set_index(idx_col)
        x.sort_values(idx_col, inplace=True)
        st = time.time()
        # x = x[agg_cols].rolling(window).sum().add_prefix('sum_')
        x_sum = x[agg_cols].rolling(window).sum().add_prefix('sum_')
        x_mean = x[agg_cols].rolling(window).mean().add_prefix('mean_')
        x_median = x[agg_cols].rolling(window).median().add_prefix('median_')
        x_min = x[agg_cols].rolling(window).min().add_prefix('min_')
        x_max = x[agg_cols].rolling(window).max().add_prefix('max_')
        x_var = x[agg_cols].rolling(window).var().add_prefix('var_')
        et = time.time()
        print('agg in parallel Time elapsed: {} seconds'.format(et - st))
        return pd.concat([x_sum, x_mean, x_median, x_min, x_max, x_var], axis=1)
    gby_sum = gby_ddf.apply(group_func)
    return gby_sum.compute().reset_index().set_index('pickup_datetime').sort_index()


# load data
HOME_DIR = '/Users/chaokunchang/repos/ApproxInfer'
data_dir = os.path.join(HOME_DIR, 'data/nyc_taxi_2018-01-01_2018-12-31')
# src_data = os.path.join(data_dir, 'head100000.csv')
src_data = os.path.join(data_dir, 'cleaned_nyc_taxi_data_2018.csv')
df = pd.read_csv(src_data)
# df.rename(columns={'Unnamed: 0': 'tid'}, inplace=True)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

feature_dir = os.path.join(data_dir, 'features')
if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)

if __name__ == '__main__':
    st = time.time()
    extract_all(df, groupby_col='vendor_id', window='1H').to_csv(
        os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv'))
    # extract_all(df, groupby_col='vendor_id', window='1D').to_csv(
    #     os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1day.csv'))
    # extract_all(df, groupby_col='vendor_id', window='7D').to_csv(
    #     os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1week.csv'))

    # extract_all(df, groupby_col='passenger_count', window='1H').to_csv(
    #     os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1hour.csv'))
    # extract_all(df, groupby_col='passenger_count', window='1D').to_csv(
    #     os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1day.csv'))
    # extract_all(df, groupby_col='passenger_count', window='7D').to_csv(
    #     os.path.join(feature_dir, 'nyc_taxi_data_2018_gbypc_winbypickup_1week.csv'))

    et = time.time()
    print('Synchron Time elapsed: {} seconds'.format(et - st))

    st = time.time()
    extract_all(df, groupby_col='vendor_id', window='1H').to_csv(
        os.path.join(feature_dir, 'parallel_apply_nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv'))
    et = time.time()
    print('Parallel Time elapsed: {} seconds'.format(et - st))

    st = time.time()
    extract_all_parallel(df, groupby_col='vendor_id', window='1H').to_csv(
        os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv'))
    # extract_all_parallel(df, groupby_col='vendor_id', window='1D').to_csv(
    #     os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbyvid_winbypickup_1day.csv'))
    # extract_all_parallel(df, groupby_col='vendor_id', window='7D').to_csv(
    #     os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbyvid_winbypickup_1week.csv'))

    # extract_all_parallel(df, groupby_col='passenger_count', window='1H').to_csv(
    #     os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbypc_winbypickup_1hour.csv'))
    # extract_all_parallel(df, groupby_col='passenger_count', window='1D').to_csv(
    #     os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbypc_winbypickup_1day.csv'))
    # extract_all_parallel(df, groupby_col='passenger_count', window='7D').to_csv(
    #     os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbypc_winbypickup_1week.csv'))

    et = time.time()
    print('Parallel Time elapsed: {} seconds'.format(et - st))

    assert pd.read_csv(os.path.join(feature_dir, 'nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv')).equals(
        pd.read_csv(os.path.join(feature_dir, 'p_nyc_taxi_data_2018_gbyvid_winbypickup_1hour.csv')))