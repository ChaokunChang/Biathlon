from shared import *


def get_desc(args: SimpleParser):
    desc = {}
    ffilename = args.ffile_prefix
    desc['key'] = args.keycol

    desc['labels'] = pd.read_csv(args.label_src, nrows=1).columns.tolist()
    desc['nonagg_features'] = pd.read_csv(
        args.req_src, nrows=1).columns.tolist()
    desc['agg_features'] = pd.read_csv(os.path.join(
        args.feature_dir, f'{ffilename}.csv'), nrows=1).columns.tolist()
    desc['datetime_features'] = ['pickup_datetime']

    desc['labels'].remove(args.keycol)
    desc['nonagg_features'].remove(args.keycol)
    desc['nonagg_features'].remove('pickup_datetime')
    desc['agg_features'].remove(args.keycol)

    desc['cat_features'] = ['pickup_ntaname',
                            'dropoff_ntaname', 'passenger_count']
    desc['num_features'] = [col for col in desc['nonagg_features'] + desc['agg_features']
                            if col not in desc['cat_features'] + desc['datetime_features']]
    return desc


def load_data(args: SimpleParser, sort_by='pickup_datetime', keyids=None):
    df_labels = pd.read_csv(args.label_src)
    df_request = pd.read_csv(args.req_src)

    ffilename = args.ffile_prefix
    fdf = pd.read_csv(os.path.join(args.feature_dir, f'{ffilename}.csv'))
    if keyids is not None:
        fdf = fdf[fdf[args.keycol].isin(keyids)]
    df = fdf.merge(df_labels, on=args.keycol).merge(df_request, on=args.keycol)
    df = df.sort_values(by=sort_by)

    desc = get_desc(args)
    return desc, df


def data_preprocessing(args: SimpleParser, desc: dict,  df: pd.DataFrame, dropna=True, apx_cols=None):
    def remove_invalid(df: pd.DataFrame, dropna):
        if dropna:
            df_isna = df.isna().any(axis=1)
            isna = df_isna
            df = df[~isna]
        else:
            isna = df.isna().any(axis=1)
            df = df.fillna(0)
        assert df.isna().sum().sum() == 0
        return df

    def encode_datetime_features(df: pd.DataFrame):
        new_df = df.copy(deep=True)
        # convert pickup_datetime to datetime, and extract year, month, day, hour, weekday, weekend
        new_df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        new_df['pickup_year'] = new_df['pickup_datetime'].dt.year
        new_df['pickup_month'] = new_df['pickup_datetime'].dt.month
        new_df['pickup_day'] = new_df['pickup_datetime'].dt.day
        new_df['pickup_hour'] = new_df['pickup_datetime'].dt.hour
        new_df['pickup_minute'] = new_df['pickup_datetime'].dt.minute
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

    def apx_value_estimation(df: pd.DataFrame, apx_cols):
        sampling_rate = args.sample
        count_names = [x for x in apx_cols if x.startswith('count_')]
        sum_names = [x for x in apx_cols if x.startswith('sum_')]
        df[count_names] = df[count_names].apply(
            lambda x: x/sampling_rate, axis=0)
        df[sum_names] = df[sum_names].apply(
            lambda x: x/sampling_rate, axis=0)
        return df

    df = remove_invalid(df, dropna=dropna)
    df = encode_datetime_features(df)
    # df = encode_cat_features(df, cat_features=desc['cat_features'])
    if apx_cols is not None:
        df = apx_value_estimation(df, apx_cols)

    return df


def get_features_and_labels(args: SimpleParser, desc: dict, df: pd.DataFrame, target: str):
    features = df[desc['nonagg_features'] + desc['agg_features']]
    assert features.isna().sum().sum() == 0
    assert target in desc['labels']
    labels = df[target]
    return features, labels


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
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
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
    elif args.model_name == 'dt':
        dt_params = {
            'random_state': args.random_state,
        }
        model = DecisionTreeRegressor(**dt_params)
    elif args.model_name == 'svr':
        svr_params = {
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'auto',
            'coef0': 0.0,
            'tol': 0.001,
            'C': 1.0,
            'epsilon': 0.1,
            'shrinking': True,
            'cache_size': 200,
            'verbose': 1,
            'max_iter': 1000,
        }
        model = SVR(**svr_params)
    elif args.model_name == 'knn':
        knn_params = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'metric_params': None,
            'n_jobs': None,
        }
        model = KNeighborsRegressor(**knn_params)
    elif args.model_name == 'mlp':
        mlp_params = {
            'hidden_layer_sizes': (100, 100),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = MLPRegressor(**mlp_params)
    else:
        raise ValueError("model name not supported")

    return model


def create_pipeline(args: SimpleParser, desc: dict, df: pd.DataFrame, target: str = 'fare_amount'):
    features, labels = get_features_and_labels(
        args, desc, df, target=target)

    # split data into train and test
    X_train, X_test, y_train, y_test, kids_train, kids_test = train_test_split(
        features, labels, df[args.keycol], test_size=args.model_test_size, shuffle=False)

    numerical_features = desc['num_features']
    cat_features = desc['cat_features']

    model = create_model(args)
    # create pipeline, keep feature names
    set_config(transform_output='pandas')
    pipe = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False,
             handle_unknown='ignore'), cat_features)
        ], remainder='passthrough', verbose=True, verbose_feature_names_out=False)),
        ('model', model),
    ])

    return pipe, X_train, X_test, y_train, y_test, kids_train, kids_test


def compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True):
    # We can compute feature importance of model's feature, or pipeline's feature.
    if use_pipe_feature:
        return X.columns, permutation_importance(pipe, X, y, n_repeats=10, max_samples=1000, random_state=0, n_jobs=-1).importances_mean
    else:
        X = pipe[:-1].transform(X)
        return X.columns, permutation_importance(pipe[-1], X, y, n_repeats=10, max_samples=1000, random_state=0, n_jobs=-1).importances_mean


def get_feature_importance(args: SimpleParser, pipe: Pipeline, X, y):
    model = pipe[-1]
    if args.model_name == 'lgbm':
        return model.feature_name_, model.feature_importances_
    elif args.model_name == 'xgb':
        return model.feature_names_in_, model.feature_importances_
    elif args.model_name == 'rf':
        return model.feature_names_in_, model.feature_importances_
    elif args.model_name == 'lr':
        return model.feature_names_in_, model.coef_
    elif args.model_name == 'dt':
        return model.feature_names_in_, model.feature_importances_
    elif args.model_name == 'svr':
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif args.model_name == 'knn':
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif args.model_name == 'mlp':
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    else:
        raise ValueError("model name not supported")


def get_importance_features(args: SimpleParser, fnames, fimps, fcols: list, thr=0, topk=0):
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


def evaluate_pipeline(args: SimpleParser, pipe: Pipeline, X, y, tag, verbose=False):
    y_pred = pipe.predict(X)
    mse = metrics.mean_squared_error(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    expv = metrics.explained_variance_score(y, y_pred)
    maxe = metrics.max_error(y, y_pred)
    if verbose:
        print(f'evaluate_pipeline: {tag} y_pred.shape={y_pred.shape}')
        print(f'MSE  of {tag} : ', mse)
        print(f'MAE  of {tag} : ', mae)
        print(f'R2   of {tag} : ', r2)
        print(f'ExpV of {tag} : ', expv)
        print(f'MaxE of {tag} : ', maxe)
    return tag, mse, mae, r2, expv, maxe


def plot_hist_and_save(args: SimpleParser, data, fname, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    n, bins, patches = ax.hist(data, bins=100)
    n, bins, patches = ax2.hist(
        data, cumulative=1, histtype='step', bins=100, color='tab:orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(args.outdir, fname))
    plt.close()


def build_pipeline(args: SimpleParser):
    desc, df = load_data(args)
    df = data_preprocessing(args, desc, df, dropna=True)
    desc['nonagg_features'] += ['pickup_month', 'pickup_day',
                                'pickup_hour', 'pickup_minute', 'pickup_weekday']
    desc['cat_features'] += ['pickup_month', 'pickup_day',
                             'pickup_hour', 'pickup_minute', 'pickup_weekday']
    pipe, X_train, X_test, y_train, y_test, kids_train, kids_test = create_pipeline(
        args, desc, df)
    pipe.fit(X_train, y_train)
    # save pipeline to file
    joblib.dump(pipe, os.path.join(args.outdir, 'pipe.pkl'))
    desc['kids'] = df[args.keycol].values
    return pipe, X_train, X_test, y_train, y_test, kids_train, kids_test, desc


if __name__ == '__main__':
    args = SimpleParser().parse_args()
    assert args.sample >= 0 and args.sample <= 1, f'sample={args.sample} must be in [0, 1]'
    # print(f'args={args}')

    pipe, X_train, X_test, y_train, y_test, kids_train, kids_test, desc = build_pipeline(
        SimpleParser().parse_args().from_dict({'feature_dir': os.path.join(args.feature_dir, '../')}))

    fnames, fimps = get_feature_importance(args, pipe, X_train, y_train)
    # print(f'fnames = {fnames} \nfimps = {fimps}')
    important_fnames = get_importance_features(
        args, fnames, fimps, X_train.columns.tolist(), topk=10)
    print(f'selected importanct features: {important_fnames}')

    evals = []
    evals.append(evaluate_pipeline(args, pipe, X_train, y_train, 'train'))
    evals.append(evaluate_pipeline(args, pipe, X_test, y_test, 'test'))

    # args.feature_dir = apx_feature_dir
    apx_desc, apx_df = load_data(args, keyids=desc['kids'])
    apx_df = data_preprocessing(
        args, apx_desc, apx_df, dropna=False, apx_cols=apx_desc['agg_features'])
    apx_desc['nonagg_features'] += ['pickup_month', 'pickup_day',
                                    'pickup_hour', 'pickup_minute', 'pickup_weekday']
    apx_desc['cat_features'] += ['pickup_month', 'pickup_day',
                                 'pickup_hour', 'pickup_minute', 'pickup_weekday']
    apx_features, apx_labels = get_features_and_labels(
        args, apx_desc, apx_df, target=args.target)
    apx_X_train, apx_X_test, apx_y_train, apx_y_test = train_test_split(
        apx_features, apx_labels, test_size=args.model_test_size, shuffle=False)

    evals.append(evaluate_pipeline(
        args, pipe, apx_X_train, apx_y_train, 'apx_train'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_X_test, apx_y_test, 'apx_test'))
    evals.append(evaluate_pipeline(args, pipe, apx_X_train,
                 pipe.predict(X_train), 'apx_train sim'))
    evals.append(evaluate_pipeline(args, pipe, apx_X_test,
                 pipe.predict(X_test), 'apx_test sim'))

    exp_X_test = baseline_expected_default(
        X_train, X_test, desc['agg_features'])
    evals.append(evaluate_pipeline(args, pipe, exp_X_test, y_test, 'exp_test'))
    evals.append(evaluate_pipeline(args, pipe, exp_X_test,
                 pipe.predict(X_test), 'exp_test sim'))

    # show evals as pandas dataframe
    evals_df = pd.DataFrame(
        evals, columns=['tag', 'mse', 'mae', 'r2', 'expv', 'maxe'])
    print(evals_df)

    plot_hist_and_save(args, y_train, 'y_train.png', 'y_train', 'y', 'count')
    plot_hist_and_save(args, y_test, 'y_test.png', 'y_test', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(X_train),
                       'y_train_pred.png', 'y_train_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(X_test),
                       'y_test_pred.png', 'y_test_pred', 'y', 'count')
    plot_hist_and_save(args, apx_y_train, 'apx_y_train.png',
                       'apx_y_train', 'y', 'count')
    plot_hist_and_save(args, apx_y_test, 'apx_y_test.png',
                       'apx_y_test', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(apx_X_train),
                       'apx_y_train_pred.png', 'apx_y_train_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(apx_X_test),
                       'apx_y_test_pred.png', 'apx_y_test_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(exp_X_test),
                       'exp_y_test_pred.png', 'exp_y_test_pred', 'y', 'count')
