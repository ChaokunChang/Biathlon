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


args = SimpleArgs().parse_args()
sampling_rate = args.sampling_rate
num_reqs = args.num_reqs

if num_reqs > 0:
    feature_dir = os.path.join(data_dir, f'test_{num_reqs}xReqs', 'features')
    apx_feature_dir = os.path.join(
        data_dir, f'test_{num_reqs}xReqs', f'features_apx_{sampling_rate}')
else:
    feature_dir = os.path.join(data_dir, 'features')
    apx_feature_dir = os.path.join(data_dir, f'features_apx_{sampling_rate}')

# %%
df_labels = pd.read_csv(os.path.join(data_dir, 'labels_08-01_08-15.csv'))
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
    df['is_high_fare'] = df['fare_amount'].apply(lambda x: 1 if x > 20 else 0)
    df['is_high_tip'] = df['tip_amount'].apply(lambda x: 1 if x > 2.5 else 0)
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


# %%
target_label = 'fare_amount'
target_label = 'trip_distance'
target_label = 'is_long_trip'
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
# X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    df_features, df_target, test_size=0.3, shuffle=False)

# %%
# model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=77)
model = DecisionTreeClassifier(min_samples_leaf=300, random_state=77)
model.fit(X_train, y_train)


# %%
print(
    f'tree depth = {model.get_depth()}, number of leaf nodes = {model.get_n_leaves()}, params: {model.get_params()}')
print(f'feature_importance: {model.feature_importances_}')
# print name of feature with non-zero importance
important_fnames = []
for i, imp in enumerate(model.feature_importances_):
    if imp > 0:
        print(f'feature {i} {X_train.columns[i]} importance: {imp}')
        important_fnames.append(X_train.columns[i])


plt.figure(figsize=(20, 20))
plot_tree(model, filled=True)
plt.title("Decision tree trained on all the iris features")
# plt.savefig('tree.pdf')

# %%


def evaluate_model(model, xs, ys):
    y_predicted = model.predict(xs)

    accuracy = metrics.accuracy_score(ys, y_predicted)
    precision = metrics.precision_score(ys, y_predicted, zero_division=1)
    recall = metrics.recall_score(ys, y_predicted, zero_division=1)
    f1_score = metrics.f1_score(ys, y_predicted, zero_division=1)

    print("--------------------------------------")
    # print('Accuracy is  {}'.format(accuracy))
    # print('Precision is {}'.format(precision))
    # print('Recall is    {}'.format(recall))
    # print('F1 score is  {}'.format(f1_score))
    # accuracy for each class
    print(metrics.classification_report(
        ys, y_predicted, digits=5, zero_division=1))
    print("--------------------------------------")
    return accuracy, precision, recall, f1_score


print("The model performance for training set")
evaluate_model(model, X_train, y_train)
print("The model performance for testing set")
evaluate_model(model, X_test, y_test)

# %%
y_predicted = model.predict(X_test)
# show percentage of different values
print("prediction value counts: ", pd.Series(
    y_predicted).value_counts(normalize=True))

# %% [markdown]
# ## Train model with feature with non-zero importance

# %%
important_fnames = [
    fname for fname in important_fnames if fname in agg_feature_names]
new_X_train = X_train[important_fnames]
new_X_test = X_test[important_fnames]

# new_model = DecisionTreeClassifier(max_leaf_nodes=20, random_state=77)
new_model = DecisionTreeClassifier(min_samples_leaf=300, random_state=77)
new_model.fit(new_X_train, y_train)

print(f'tree depth = {new_model.get_depth()}, number of leaf nodes = {new_model.get_n_leaves()}, params: {new_model.get_params()}')
print(f'feature_importance: {new_model.feature_importances_}')
# print name of feature with non-zero importance
for i, imp in enumerate(new_model.feature_importances_):
    if imp > 0:
        print(f'feature {i} {new_X_train.columns[i]} importance: {imp}')

plt.figure(figsize=(20, 20))
plot_tree(new_model, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.savefig('tree.pdf')

# %%
print("The model performance for training set")
evaluate_model(new_model, new_X_train, y_train)
print("The model performance for testing set")
evaluate_model(new_model, new_X_test, y_test)

# show percentage of different values
print("prediction value counts: ", pd.Series(
    new_model.predict(new_X_test)).value_counts(normalize=True))

# %%
apx_df_raw_features = apx_df[selected_nonagg_features]
apx_df_agg_features = apx_df[selected_agg_features]
apx_df_features = apx_df_raw_features.join(apx_df_agg_features)
apx_X_train, apx_X_test, apx_y_train, apx_y_test = train_test_split(
    apx_df_features, df_target, test_size=0.3, shuffle=False)
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

# show percentage of different values
print("prediction value counts: ", pd.Series(
    new_model.predict(apx_X_test)).value_counts(normalize=True))

# %%
# train set
node_train = new_model.apply(new_X_train)
apx_node_train = new_model.apply(apx_X_train)
print(metrics.classification_report(
    apx_node_train, node_train, digits=5, zero_division=1))

# test set
node_test = new_model.apply(new_X_test)
apx_node_test = new_model.apply(apx_X_test)
print(metrics.classification_report(
    apx_node_test, node_test, digits=5, zero_division=1))

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


# train set
tmp_node_train = new_model.apply(tmp_X_train)
print(metrics.classification_report(
    tmp_node_train, node_train, digits=5, zero_division=1))

# test set
tmp_node_test = new_model.apply(tmp_X_test)
print(metrics.classification_report(
    tmp_node_test, node_test, digits=5, zero_division=1))

print("prediction value counts: ", pd.Series(
    new_model.predict(tmp_X_train)).value_counts(normalize=True))

# %%
