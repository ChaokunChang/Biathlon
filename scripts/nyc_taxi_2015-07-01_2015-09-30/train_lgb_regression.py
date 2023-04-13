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
import pickle
import clickhouse_connect
from rich import inspect
from tap import Tap
from tqdm import tqdm
tqdm.pandas()
pandarallel.initialize(progress_bar=True)

HOME_DIR = '/home/ckchang/ApproxInfer'
data_dir = os.path.join(HOME_DIR, 'data/nyc_taxi_2015-07-01_2015-09-30')


class SimpleArgs(Tap):
    sampling_rate: float = 0.1  # sample rate of sql query. default 0.1 means 10% of data
    num_reqs: int = 0  # number of requests sampled for testing. default 0 means no sampling
    random_state: int = 42  # random state for train_test_split
    test_size: float = 0.3  # test size for train_test_split


args = SimpleArgs().parse_args()
sampling_rate = args.sampling_rate
num_reqs = args.num_reqs
random_state = args.random_state
test_size = args.test_size

if num_reqs > 0:
    feature_dir = os.path.join(data_dir, f'sample_x{num_reqs}', 'features')
    apx_feature_dir = os.path.join(
        data_dir, f'sample_x{num_reqs}', f'apx_features_{sampling_rate}')
else:
    feature_dir = os.path.join(data_dir, 'features')
    apx_feature_dir = os.path.join(data_dir, f'apx_features_{sampling_rate}')

# %%
df_labels = pd.read_csv(os.path.join(data_dir, 'trips_labels.csv'))
df = pd.read_csv(os.path.join(feature_dir, 'requests_08-01_08-15.feas.csv'))
apx_df = pd.read_csv(os.path.join(
    apx_feature_dir, 'requests_08-01_08-15.feas.csv'))

# %%


def df_preprocessing(df, apx_df):
    df = df.merge(df_labels, on='trip_id')
    apx_df = apx_df.merge(df_labels, on='trip_id')

    assert df['trip_id'].equals(apx_df['trip_id'])
    # assert df.isna().equals(apx_df.isna())

    # if the row in df contains NaN, remove that row from both df and apx_df
    isna = df.isna().any(axis=1)
    df = df[~isna]
    apx_df = apx_df[~isna]
    # if the row in apx_df contains NaN value, set that NaN value to 0
    apx_df = apx_df.fillna(0)

    assert df.isna().sum().sum() == 0 and apx_df.isna().sum(
    ).sum() == 0 and df['trip_id'].equals(apx_df['trip_id'])

    def encode_datetime_features(df):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['pickup_year'] = df['pickup_datetime'].dt.year
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['pickup_day'] = df['pickup_datetime'].dt.day
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['pickup_is_weekend'] = df['pickup_weekday'].apply(
            lambda x: 1 if x in [5, 6] else 0)
        return df

    def encode_cat_features(df):
        df['pickup_ntaname'] = df['pickup_ntaname'].astype('category')
        # df['dropoff_ntaname'] = df['dropoff_ntaname'].astype('category')
        return df

    df = encode_datetime_features(df)
    df = encode_cat_features(df)
    apx_df = encode_datetime_features(apx_df)
    apx_df = encode_cat_features(apx_df)

    count_names = [x for x in df.columns if x.startswith('count_')]
    sum_names = [x for x in df.columns if x.startswith('sum_')]
    apx_df[count_names] = apx_df[count_names].apply(
        lambda x: x/sampling_rate, axis=0)
    apx_df[sum_names] = apx_df[sum_names].apply(
        lambda x: x/sampling_rate, axis=0)

    df['is_long_trip'] = df['trip_distance'].apply(lambda x: 1 if x > 5 else 0)
    df['is_high_fare'] = df['fare_amount'].apply(lambda x: 1 if x > 10 else 0)
    df['is_high_tip'] = df['tip_amount'].apply(lambda x: 1 if x > 0 else 0)
    return df, apx_df


df, apx_df = df_preprocessing(df, apx_df)

# %%
corr = df.corr()
corr

# %%
nonagg_feature_names = ['pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour',
                        'pickup_weekday', 'pickup_is_weekend', 'pickup_ntaname',
                        'passenger_count', 'pickup_latitude', 'pickup_longitude']
aggops = ['count', 'mean', 'sum', 'std', 'var', 'min', 'max', 'median']
aggcols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']
winhours = ['1h', '24h', '168h']
agg_feature_names = [
    f'{op}_{win}_{col}' for op in aggops for col in aggcols for win in winhours]
target_feature_names = ['trip_distance',
                        'fare_amount', 'tip_amount', 'total_amount',
                        'is_long_trip', 'is_high_fare', 'is_high_tip']
feature_names = nonagg_feature_names + agg_feature_names + target_feature_names
# make sure feature_names are in df's cloumns
assert set(feature_names).issubset(set(df.columns)), 'feature_names are not in df, difference is {}'.format(
    set(feature_names) - set(df.columns))

print(df[target_feature_names].describe())
# %%
target_label = 'trip_distance'
target_label = 'fare_amount'
# target_label = 'tip_amount'
# target_label = 'total_amount'
# target_label = 'is_long_trip'
# target_label = 'is_high_fare'
# target_label = 'is_high_tip'
# show correlation of target_label in order, expect target_feature_names
selected_w_corr = corr[target_label].sort_values(
    ascending=False).drop(target_feature_names)
print(f'corrs to {target_label}: {selected_w_corr}')
print("prediction value counts: ", pd.Series(
    df[target_label]).value_counts(normalize=True))

# %%
selected_fnames = selected_w_corr.index.tolist()
selected_nonagg_features = [
    name for name in nonagg_feature_names if name in selected_fnames]
selected_agg_features = [
    name for name in agg_feature_names if name in selected_fnames]
# selected_nonagg_features = ['trip_distance']
# selected_agg_features = ['mean_trip_distance', 'std_trip_distance', 'mean_fare_amount', 'std_fare_amount']
print(selected_nonagg_features, selected_agg_features)

# %%
df_raw_features = df[selected_nonagg_features]
df_agg_features = df[selected_agg_features]
df_features = df_raw_features.join(df_agg_features)
df_target = df[target_label]

# %%
# split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=test_size, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_target, test_size=test_size, shuffle=False)

# %%
# model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=random_state)
# model = DecisionTreeClassifier(min_samples_leaf=300, random_state=random_state)
# model.fit(X_train, y_train)
# print(
#     f'tree depth = {model.get_depth()}, number of leaf nodes = {model.get_n_leaves()}, params: {model.get_params()}')
# plt.figure(figsize=(20, 20))
# plot_tree(model, filled=True)
# plt.title("Decision tree trained on all the iris features")
# plt.savefig('tree.pdf')

# %%
# build lightgbm model
# lgb_params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'binary_logloss',
#     'num_leaves': 31,
#     'learning_rate': 0.01,
#     'n_estimators': 100,
#     'subsample_for_bin': 200000,
#     'class_weight': None,
#     'min_split_gain': 0.0,
#     'min_child_weight': 0.001,
#     'min_child_samples': 20,
#     'subsample': 1.0,
#     'subsample_freq': 0,
#     'colsample_bytree': 1.0,
#     'reg_alpha': 0.0,
#     'reg_lambda': 0.0,
#     'random_state': random_state,
#     'n_jobs': -1,
#     'verbose': 2,
#     'importance_type': 'split',
# }
lgb_params = {
    'objective': 'regression',
    'num_leaves': 10,
    'learning_rate': 0.1,
    'random_state': random_state,
    'verbose': 1,

}
# train_data = lgb.Dataset(X_train, label=y_train)
# test_data = lgb.Dataset(X_test, label=y_test)
# model = lgb.train(lgb_params, train_data, valid_sets=[test_data], num_boost_round=1000)
# print(f'feature_importance: {model.feature_importance()}')
model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_train, y_train)
print(f'feature_importance: {model.feature_importances_}')

# %%


def get_importance_features(model, thr=0, topk=0):
    # print name of feature with non-zero importance
    important_fnames = []
    important_fid = []
    important_fimps = []
    # for i, imp in enumerate(model.feature_importance()):
    for i, imp in enumerate(model.feature_importances_):
        important_fid.append(i)
        important_fnames.append(X_train.columns[i])
        important_fimps.append(imp)
    topfnames = []
    for fid, fname, fimp in sorted(zip(important_fid, important_fnames, important_fimps), key=lambda x: x[2], reverse=True):
        print(f'f{fid}({fname}) importance: {fimp}')
        if topk > 0 and len(topfnames) >= topk:
            break
        if fimp > thr:
            topfnames.append(fname)
    return topfnames


important_fnames = get_importance_features(model, topk=100)

# %%


def evaluate_model(model, xs, ys):
    # evaluate the lgb regression model
    y_pred = model.predict(xs)
    print("--------------------------------------")
    print('Mean Absolute Error:', metrics.mean_absolute_error(ys, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(ys, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(ys, y_pred)))
    print('R2 score:', metrics.r2_score(ys, y_pred))
    print("--------------------------------------")
    return None


print("The model performance for training set")
evaluate_model(model, X_train, y_train)
print("The model performance for testing set")
evaluate_model(model, X_test, y_test)

# %%
# show distribution of y_test and y_pred
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title('y_test vs y_pred')
plt.savefig('y_test_vs_y_pred.pdf')


# %% [markdown]
# ## Train model with feature with non-zero importance

# %%
# important_fnames = [fname for fname in important_fnames if fname in agg_feature_names]
new_X_train = X_train[important_fnames]
new_X_test = X_test[important_fnames]

# new_model = DecisionTreeClassifier(max_leaf_nodes=20, random_state=random_state)
# new_model = DecisionTreeClassifier(min_samples_leaf=300, random_state=random_state)
# new_model.fit(new_X_train, y_train)
# print(f'tree depth = {new_model.get_depth()}, number of leaf nodes = {new_model.get_n_leaves()}, params: {new_model.get_params()}')
# plt.figure(figsize=(20, 20))
# plot_tree(new_model, filled=True)
# plt.title("Decision tree trained on all the iris features")
# plt.savefig('tree.pdf')
# train_data = lgb.Dataset(new_X_train, label=y_train)
# test_data = lgb.Dataset(new_X_test, label=y_test)
# new_model = lgb.train(lgb_params, train_data, valid_sets=[test_data], num_boost_round=1000)
# print(f'feature_importance: {new_model.feature_importance()}')

new_model = lgb.LGBMRegressor(**lgb_params)
new_model.fit(new_X_train, y_train)
print(f'feature_importance: {new_model.feature_importances_}')

_ = get_importance_features(new_model)


# %%
print("The model performance for training set")
evaluate_model(new_model, new_X_train, y_train)
print("The model performance for testing set")
evaluate_model(new_model, new_X_test, y_test)

# show distribution of y_test and y_pred
new_y_pred = new_model.predict(new_X_test)
plt.figure(figsize=(10, 10))
plt.scatter(y_test, new_y_pred)
plt.xlabel('y_test')
plt.ylabel('new_y_pred')
plt.title('y_test vs new_y_pred')
plt.savefig('y_test_vs_new_y_pred.pdf')

# %%
apx_df_raw_features = apx_df[selected_nonagg_features]
apx_df_agg_features = apx_df[selected_agg_features]
apx_df_features = apx_df_raw_features.join(apx_df_agg_features)
apx_X_train, apx_X_test, apx_y_train, apx_y_test = train_test_split(
    apx_df_features, df_target, test_size=test_size, shuffle=False)
apx_X_train = apx_X_train[important_fnames]
apx_X_test = apx_X_test[important_fnames]

# %%

print("The model performance for training set")
evaluate_model(new_model, apx_X_train, apx_y_train)
print("The model performance for testing set")
evaluate_model(new_model, apx_X_test, apx_y_test)

print("The model performance for training set to exact")
evaluate_model(new_model, apx_X_train, new_model.predict(new_X_train))
print("The model performance for testing set to exact")
evaluate_model(new_model, apx_X_test, new_model.predict(new_X_test))

# show distribution of apx_y_test and apx_y_pred
apx_y_pred = new_model.predict(apx_X_test)
plt.figure(figsize=(10, 10))
plt.scatter(apx_y_test, apx_y_pred)
plt.xlabel('apx_y_test')
plt.ylabel('apx_y_pred')
plt.title('apx_y_test vs apx_y_pred')
plt.savefig('apx_y_test_vs_apx_y_pred.pdf')

# %%
tmp_X_train = new_X_train.copy()
tmp_X_test = new_X_test.copy()
for name in important_fnames:
    if name in agg_feature_names:
        default_value = X_train[name].mean()
        # default_value = 0
        print(f'feature {name} default value: {default_value}')
        tmp_X_train[name] = default_value
        tmp_X_test[name] = default_value
print("The model performance for training set")
evaluate_model(new_model, tmp_X_train, y_train)
print("The model performance for testing set")
evaluate_model(new_model, tmp_X_test, y_test)

print("The model performance for training set to exact")
evaluate_model(new_model, tmp_X_train, new_model.predict(new_X_train))
print("The model performance for testing set to exact")
evaluate_model(new_model, tmp_X_test, new_model.predict(new_X_test))


# # train set
# tmp_node_train = new_model.apply(tmp_X_train)
# print(metrics.classification_report(
#     tmp_node_train, node_train, digits=5, zero_division=1))

# # test set
# tmp_node_test = new_model.apply(tmp_X_test)
# print(metrics.classification_report(
#     tmp_node_test, node_test, digits=5, zero_division=1))

# show distribution of y_test and tmp_y_pred
tmp_y_pred = new_model.predict(tmp_X_test)
plt.figure(figsize=(10, 10))
plt.scatter(y_test, tmp_y_pred)
plt.xlabel('y_test')
plt.ylabel('tmp_y_pred')
plt.title('y_test vs tmp_y_pred')
plt.savefig('y_test_vs_tmp_y_pred.pdf')


# %%
