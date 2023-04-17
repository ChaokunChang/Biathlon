from shared import *

args = SimpleParser().parse_args()
# print(f'args={args}')
assert args.sample >= 0 and args.sample <= 1

sampling_rate = args.sample

# load features


def load_data(args: SimpleParser):
    desc = {'key': 'trip_id', 'labels': [],
            'nonagg_features': [], 'agg_features': []}
    df_labels = pd.read_csv(args.label_src)
    df_request = pd.read_csv(args.req_src)

    ffilename = args.ffile_prefix
    fdf = pd.read_csv(os.path.join(
        args.feature_dir, '../', f'{ffilename}.csv'))
    df = fdf.merge(df_labels, on='trip_id').merge(df_request, on='trip_id')

    apx_fdf = pd.read_csv(os.path.join(
        args.feature_dir, f'{ffilename}.csv'))
    apx_df = apx_fdf.merge(df_labels, on='trip_id').merge(
        df_request, on='trip_id')
    assert df['trip_id'].equals(apx_df['trip_id'])

    desc['labels'] = [col for col in df_labels.columns if col != 'trip_id']
    desc['nonagg_features'] = [
        col for col in df_request.columns if col not in ['trip_id', 'pickup_datetime']]
    desc['agg_features'] = [col for col in fdf.columns if col not in desc['labels'] +
                            desc['nonagg_features'] + ['trip_id', 'pickup_datetime']]
    desc['datetime_features'] = ['pickup_datetime']
    desc['cat_features'] = ['pickup_ntaname', 'dropoff_ntaname']
    desc['num_features'] = [col for col in desc['nonagg_features'] + desc['agg_features']
                            if col not in desc['cat_features'] + desc['datetime_features']]
    # print(f'desc={desc}')
    return desc, df, apx_df


def data_preprocessing(args: SimpleParser, desc: dict,  df: pd.DataFrame, apx_df: pd.DataFrame, invalid_apx=0):
    def remove_invalid(df: pd.DataFrame, apx_df: pd.DataFrame, invalid_apx):
        if invalid_apx is None:
            # if the row in df or apx_df contains NaN, remove that row from both df and apx_df
            df_isna = df.isna().any(axis=1)
            apx_df_isna = apx_df.isna().any(axis=1)
            isna = df_isna | apx_df_isna
            df = df[~isna]
            apx_df = apx_df[~isna]
        else:
            isna = df.isna().any(axis=1)
            df = df[~isna]
            apx_df = apx_df[~isna]
            # if the row in apx_df still contains NaN value, set that NaN value to 0
            apx_df = apx_df.fillna(invalid_apx)

        assert df.isna().sum().sum() == 0 and apx_df.isna().sum(
        ).sum() == 0 and df['trip_id'].equals(apx_df['trip_id'])
        return df, apx_df

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

    def encode_cat_features(df: pd.DataFrame, cat_features: list = ['pickup_ntaname', 'dropoff_ntaname']):
        new_df = df.copy(deep=True)
        le = LabelEncoder()
        for fname in cat_features:
            new_df[fname] = le.fit_transform(df[fname])
        return new_df

    def apx_value_estimation(apx_df: pd.DataFrame):
        count_names = [x for x in df.columns if x.startswith('count_')]
        sum_names = [x for x in df.columns if x.startswith('sum_')]
        apx_df[count_names] = apx_df[count_names].apply(
            lambda x: x/sampling_rate, axis=0)
        apx_df[sum_names] = apx_df[sum_names].apply(
            lambda x: x/sampling_rate, axis=0)
        return apx_df

    df, apx_df = remove_invalid(df, apx_df, invalid_apx=invalid_apx)

    df = encode_datetime_features(df)
    df = encode_cat_features(df, cat_features=desc['cat_features'])

    apx_df = encode_datetime_features(apx_df)
    apx_df = encode_cat_features(apx_df, cat_features=desc['cat_features'])
    apx_df = apx_value_estimation(apx_df)
    return df, apx_df


def get_features_and_labels(args: SimpleParser, desc: dict, df: pd.DataFrame, apx_df: pd.DataFrame, target: str = 'fare_amount'):
    features = df[desc['nonagg_features'] + desc['agg_features']]
    assert features.isna().sum().sum() == 0
    assert target in desc['labels']
    labels = df[target]
    apx_features = apx_df[desc['nonagg_features'] + desc['agg_features']]
    return features, labels, apx_features


def create_model(args: SimpleParser):
    if args.model_name == 'lgbm':
        lgb_params = {
            'objective': 'regression',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = lgb.LGBMRegressor(**lgb_params)
    elif args.model_name == 'xgb':
        xgb_params = {
            'objective': 'regression',
            'learning_rate': 0.1,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = xgb.XGBRegressor(**xgb_params)
    elif args.model_name == 'rf':
        rf_params = {
            'n_estimators': 100,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = RandomForestRegressor(**rf_params)
    elif args.model_name == 'lr':
        model = LinearRegression()
    else:
        raise ValueError("model name not supported")

    return model


def create_pipeline(args: SimpleParser, desc: dict, df: pd.DataFrame, apx_df: pd.DataFrame, target: str = 'fare_amount'):
    features, labels, apx_features = get_features_and_labels(
        args, desc, df, apx_df)

    # split data into train and test
    X_train, X_test, apx_X_train, apx_X_test, y_train, y_test = train_test_split(
        features, apx_features, labels, test_size=args.model_test_size, shuffle=False)

    numerical_features = desc['num_features']
    cat_features = desc['cat_features']

    model = create_model(args)
    # create pipeline, keep feature names
    set_config(transform_output='pandas')
    pipe = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False,
             handle_unknown='ignore'), cat_features),
        ], remainder='passthrough', verbose=True, verbose_feature_names_out=False)),
        ('model', model),
    ])

    return pipe, X_train, X_test, apx_X_train, apx_X_test, y_train, y_test


def get_feature_importance(pipe: Pipeline):
    model = pipe[-1]
    if args.model_name == 'lgbm':
        return model.feature_name_, model.feature_importances_
    elif args.model_name == 'xgb':
        return model.feature_names_in, model.feature_importances_
    elif args.model_name == 'rf':
        return model.feature_names_in, model.feature_importances_
    elif args.model_name == 'lr':
        return model.feature_names_in, model.coef_
    else:
        raise ValueError("model name not supported")


def get_importance_features(pipe: Pipeline, fcols: list, thr=0, topk=0):
    fnames, fimps = get_feature_importance(pipe)
    # print(f'pipe.steps[-1][-1] = {inspect(pipe.steps[-1][-1])}')
    assert len(fimps) == len(
        fnames), f'len(fimps)={len(fimps)}, len(fnames)={len(fnames)}'

    important_fnames = []
    important_fid = []
    important_fimps = []
    for i, imp in enumerate(fimps):
        important_fid.append(i)
        important_fnames.append(fnames[i])
        important_fimps.append(imp)
    topfnames = []
    for fid, fname, fimp in sorted(zip(important_fid, important_fnames, important_fimps), key=lambda x: x[2], reverse=True):
        if fimp > thr:
            if topk > 0 and len(topfnames) >= topk:
                break
            print(f'f{fid}({fname}) importance: {fimp}')
            if fname in fcols:
                topfnames.append(fname)
            else:
                # must be cat featuresƒ
                catname = '_'.join(fname.split('_')[:-1])
                if catname in fcols and catname not in topfnames:
                    topfnames.append(catname)
    return topfnames


def baseline_expected_default(X_train, X_test, aggcols):
    default_values = X_train[aggcols].mean()
    exp_test = X_test.copy(deep=True)
    for col in aggcols:
        exp_test[col] = default_values[col]
    return exp_test


def evaluate_pipeline(args: SimpleParser, pipe: Pipeline, X_train, y_train, X_test, y_test, apx_X_test):
    important_fnames = get_importance_features(
        pipe, X_train.columns.tolist(), topk=10)
    print(f'selected features: {important_fnames}')

    y_train_pred = pipe.predict(X_train)
    y_pred = pipe.predict(X_test)
    apx_y_pred = pipe.predict(apx_X_test)
    exp_X_test = baseline_expected_default(
        X_train, X_test, desc['agg_features'])
    exp_y_pred = pipe.predict(exp_X_test)

    print('MSE of train      : ', metrics.mean_squared_error(y_train, y_train_pred))
    print('MSE of ext_y      : ', metrics.mean_squared_error(y_test, y_pred))
    print('MSE of apx_y      : ', metrics.mean_squared_error(y_test, apx_y_pred))
    print('MSE of apx_y(sim) : ', metrics.mean_squared_error(y_pred, apx_y_pred))
    print('MSE of exp_y      : ', metrics.mean_squared_error(y_test, exp_y_pred))
    print('MSE of exp_y(sim) : ', metrics.mean_squared_error(y_pred, exp_y_pred))

    print('MAE of train      : ', metrics.mean_absolute_error(y_train, y_train_pred))
    print('MAE of ext_y      : ', metrics.mean_absolute_error(y_test, y_pred))
    print('MAE of apx_y      : ', metrics.mean_absolute_error(y_test, apx_y_pred))
    print('MAE of apx_y(sim) : ', metrics.mean_absolute_error(y_pred, apx_y_pred))
    print('MAE of exp_y      : ', metrics.mean_absolute_error(y_test, exp_y_pred))
    print('MAE of exp_y(sim) : ', metrics.mean_absolute_error(y_pred, exp_y_pred))

    print('R2 of train      : ', metrics.r2_score(y_train, y_train_pred))
    print('R2 of ext_y      : ', metrics.r2_score(y_test, y_pred))
    print('R2 of apx_y      : ', metrics.r2_score(y_test, apx_y_pred))
    print('R2 of apx_y(sim) : ', metrics.r2_score(y_pred, apx_y_pred))
    print('R2 of exp_y      : ', metrics.r2_score(y_test, exp_y_pred))
    print('R2 of exp_y(sim) : ', metrics.r2_score(y_pred, exp_y_pred))

    print('Explained Variance of train      : ',
          metrics.explained_variance_score(y_train, y_train_pred))
    print('Explained Variance of ext_y      : ',
          metrics.explained_variance_score(y_test, y_pred))
    print('Explained Variance of apx_y      : ',
          metrics.explained_variance_score(y_test, apx_y_pred))
    print('Explained Variance of apx_y(sim) : ',
          metrics.explained_variance_score(y_pred, apx_y_pred))
    print('Explained Variance of exp_y      : ',
          metrics.explained_variance_score(y_test, exp_y_pred))
    print('Explained Variance of exp_y(sim) : ',
          metrics.explained_variance_score(y_pred, exp_y_pred))

    print('Max Error of train      : ', metrics.max_error(y_train, y_train_pred))
    print('Max Error of ext_y      : ', metrics.max_error(y_test, y_pred))
    print('Max Error of apx_y      : ', metrics.max_error(y_test, apx_y_pred))
    print('Max Error of apx_y(sim) : ', metrics.max_error(y_pred, apx_y_pred))
    print('Max Error of exp_y      : ', metrics.max_error(y_test, exp_y_pred))
    print('Max Error of exp_y(sim) : ', metrics.max_error(y_pred, exp_y_pred))


if __name__ == '__main__':
    desc, df, apx_df = load_data(args)
    df, apx_df = data_preprocessing(args, desc, df, apx_df, invalid_apx=0)
    pipe, X_train, X_test, apx_X_train, apx_X_test, y_train, y_test = create_pipeline(
        args, desc, df, apx_df)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    apx_y_pred = pipe.predict(apx_X_test)
    exp_y_pred = pipe.predict(baseline_expected_default(
        X_train, X_test, desc['agg_features']))

    evaluate_pipeline(args, pipe, X_train, y_train, X_test, y_test, apx_X_test)
