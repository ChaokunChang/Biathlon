# %%
from shared import *

args = SimpleParser().parse_args()
print(f'args={args}')

sampling_rate = args.sample
random_state = args.random_state
test_size = args.model_test_size
# %%
df_labels = pd.read_csv(args.label_src)
df_request = pd.read_csv(args.req_src)
ffilename = 'features'
df = pd.read_csv(os.path.join(args.feature_dir, f'{ffilename}.csv'))
apx_df = pd.read_csv(os.path.join(args.feature_dir, '../', f'{ffilename}.csv'))
# %%
df = df.merge(df_labels, on='trip_id').merge(df_request, on='trip_id')
apx_df = apx_df.merge(df_labels, on='trip_id').merge(df_request, on='trip_id')
assert df['trip_id'].equals(apx_df['trip_id'])
# %%


def df_preprocessing(df, apx_df):
    # assert df.isna().equals(apx_df.isna())

    # if the row in df contains NaN, remove that row from both df and apx_df
    isna = df.isna().any(axis=1)
    df = df[~isna]
    apx_df = apx_df[~isna]
    # if the row in apx_df contains NaN value, set that NaN value to 0
    apx_df = apx_df.fillna(0)

    assert df.isna().sum().sum() == 0 and apx_df.isna().sum(
    ).sum() == 0 and df['trip_id'].equals(apx_df['trip_id'])

    def encode_datetime_features(df: pd.DataFrame):
        new_df = df.copy(deep=True)
        # convert pickup_datetime to datetime, and extract year, month, day, hour, weekday, weekend
        new_df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        new_df['pickup_year'] = new_df['pickup_datetime'].dt.year
        new_df['pickup_month'] = new_df['pickup_datetime'].dt.month
        new_df['pickup_day'] = new_df['pickup_datetime'].dt.day
        new_df['pickup_hour'] = new_df['pickup_datetime'].dt.hour
        new_df['pickup_weekday'] = new_df['pickup_datetime'].dt.weekday
        new_df['pickup_is_weekend'] = new_df['pickup_weekday'].apply(
            lambda x: 1 if x in [5, 6] else 0)
        return new_df

    def encode_cat_features(df):
        df['pickup_ntaname'] = df['pickup_ntaname'].astype('category')
        df['dropoff_ntaname'] = df['dropoff_ntaname'].astype('category')
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

    def add_class_labels(df):
        df['is_long_trip'] = df['trip_duration'].apply(
            lambda x: 1 if x > 3600 else 0)
        df['is_high_fare'] = df['fare_amount'].apply(
            lambda x: 1 if x > 10 else 0)
        df['is_high_tip'] = df['tip_amount'].apply(lambda x: 1 if x > 0 else 0)
        return df

    df = add_class_labels(df)
    apx_df = add_class_labels(apx_df)

    return df, apx_df


df, apx_df = df_preprocessing(df, apx_df)

# %%
corr = df.corr(numeric_only=True)
corr

# %%
nonagg_feature_names = ['trip_distance', 'passenger_count',
                        'pickup_year', 'pickup_month', 'pickup_day',
                        'pickup_weekday', 'pickup_is_weekend', 'pickup_hour',
                        'pickup_latitude', 'pickup_longitude', 'pickup_ntaname',
                        'dropoff_latitude', 'dropoff_longitude', 'dropoff_ntaname']
aggops = ['avg', 'sum', 'std', 'var', 'min', 'max', 'median']
aggcols = ['trip_duration', 'trip_distance',
           'fare_amount', 'tip_amount', 'total_amount']
winhours = ['1h', '24h', '168h']
agg_feature_names = [f'count_{win}' for win in winhours] + [
    f'{op}_{col}_{win}' for op in aggops for col in aggcols for win in winhours]
target_feature_names = ['trip_duration',
                        'fare_amount', 'tip_amount', 'total_amount',
                        'is_long_trip', 'is_high_fare', 'is_high_tip']
feature_names = nonagg_feature_names + agg_feature_names + target_feature_names
# make sure feature_names are in df's cloumns
assert set(feature_names).issubset(set(df.columns)), 'feature_names are not in df, difference is {}'.format(
    set(feature_names) - set(df.columns))

print(df[target_feature_names].describe())
# %%
target_label = 'trip_duration'
target_label = 'fare_amount'
# target_label = 'tip_amount'
# target_label = 'total_amount'
# target_label = 'is_long_trip'
# target_label = 'is_high_fare'
# target_label = 'is_high_tip'
# show correlation of target_label in order, expect target_feature_names
selected_w_corr = corr[target_label].sort_values(
    ascending=False).drop(target_feature_names)
print(f'corrs to {target_label}: \n{selected_w_corr}')
print("prediction value distribution: ", df[target_label].describe())

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
    'num_leaves': 31,
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


def get_importance_features(model, fnames, thr=0, topk=0):
    # print name of feature with non-zero importance
    important_fnames = []
    important_fid = []
    important_fimps = []
    # for i, imp in enumerate(model.feature_importance()):
    for i, imp in enumerate(model.feature_importances_):
        important_fid.append(i)
        important_fnames.append(fnames[i])
        important_fimps.append(imp)
    topfnames = []
    for fid, fname, fimp in sorted(zip(important_fid, important_fnames, important_fimps), key=lambda x: x[2], reverse=True):
        print(f'f{fid}({fname}) importance: {fimp}')
        if topk > 0 and len(topfnames) >= topk:
            break
        if fimp > thr:
            topfnames.append(fname)
    return topfnames


important_fnames = get_importance_features(model, X_train.columns, topk=100)
print(f'important features: {important_fnames}')
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
plt.savefig(os.path.join(args.outdir, 'y_test_vs_y_pred.pdf'))


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

_ = get_importance_features(new_model, new_X_train.columns)


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
plt.savefig(os.path.join(args.outdir, 'y_test_vs_new_y_pred.pdf'))

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

# show distribution of apx_y_test and apx_y_pred
apx_y_pred = new_model.predict(apx_X_test)
plt.figure(figsize=(10, 10))
plt.scatter(apx_y_test, apx_y_pred)
plt.xlabel('apx_y_test')
plt.ylabel('apx_y_pred')
plt.title('apx_y_test vs apx_y_pred')
plt.savefig(os.path.join(args.outdir, 'apx_y_test_vs_apx_y_pred.pdf'))

print("The model performance for training set to exact")
evaluate_model(new_model, apx_X_train, new_model.predict(new_X_train))
print("The model performance for testing set to exact")
evaluate_model(new_model, apx_X_test, new_model.predict(new_X_test))
# show distribution of apx_y_pred and apx_y_pred_exact
apx_y_pred_exact = new_model.predict(new_X_test)
plt.figure(figsize=(10, 10))
plt.scatter(apx_y_pred_exact, apx_y_pred)
plt.xlabel('apx_y_pred_exact')
plt.ylabel('apx_y_pred')
plt.title('apx_y_pred_exact vs apx_y_pred')
plt.savefig(os.path.join(args.outdir, 'apx_y_pred_exact_vs_apx_y_pred.pdf'))


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
plt.savefig(os.path.join(args.outdir, 'y_test_vs_tmp_y_pred.pdf'))

# %%
