from shared import *
set_config(transform_output='pandas')


def apx_value_estimation(df: pd.DataFrame, apx_cols: list, sampling_rate: float):
    count_names = [x for x in apx_cols if x.startswith('count_')]
    sum_names = [x for x in apx_cols if x.startswith('sum_')]
    df[count_names] = df[count_names].apply(
        lambda x: x/sampling_rate, axis=0)
    df[sum_names] = df[sum_names].apply(
        lambda x: x/sampling_rate, axis=0)
    return df


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
            'max_iter': 100,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = MLPRegressor(**mlp_params)
    else:
        raise ValueError("model name not supported")

    return model


def compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True):
    # We can compute feature importance of model's feature, or pipeline's feature.
    if use_pipe_feature:
        return X.columns, permutation_importance(pipe, X, y, n_repeats=10, max_samples=min(1000, len(X)), random_state=random_state, n_jobs=-1).importances_mean
    else:
        X = pipe[:-1].transform(X)
        return X.columns, permutation_importance(pipe[-1], X, y, n_repeats=10, max_samples=min(1000, len(X)), random_state=0, n_jobs=-1).importances_mean


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


def load_from_csv(fpath: str, keycol: str, kids=None, sort_by=None) -> pd.DataFrame:
    df = pd.read_csv(fpath)
    if kids is not None:
        df = df[df[keycol].isin(kids)]
    if sort_by is not None:
        df = df.sort_values(by=sort_by)
    return df


def load_features(args: SimpleParser, kids=None, sort_by=None):
    features = load_from_csv(os.path.join(
        args.feature_dir, f'{args.ffile_prefix}.csv'), args.keycol, kids, None)
    # reqs = load_from_csv(args.req_src, args.keycol, kids, None)
    # features = features.merge(reqs, on=args.keycol)
    if sort_by is not None:
        features = features.sort_values(by=sort_by)
    return features


def load_labels(args: SimpleParser, kids=None):
    labels = load_from_csv(args.label_src, args.keycol, kids, None)
    assert labels.isna().sum().sum() == 0, f'labels contains NaN'
    return labels[[args.keycol, args.target]]


def nan_processing(df: pd.DataFrame, dropna):
    if dropna:
        df = df.dropna()
    else:
        df = df.fillna(0)
    assert df.isna().sum().sum() == 0
    return df


def _feature_dtype_inference(df: pd.DataFrame, keycol: str, target: str):
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
                if col.find('_') > 0 and col.split('_', 1)[0] in SUPPORTED_AGGS:
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


def _feature_ctype_inference(df: pd.DataFrame, keycol: str, target: str):
    agg_features = []
    nonagg_features = []
    for col in df.columns:
        if col == keycol or col == target:
            continue
        # if col starts with '{agg}_', where agg is in [count, avg, sum, var, std, min, max, median], it is an aggregated feature
        if col.find('_') > 0 and col.split('_', 1)[0] in SUPPORTED_AGGS:
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
    num_features, cat_features, dt_features = _feature_dtype_inference(
        df, keycol, target)
    typed_features['num_features'] = num_features
    typed_features['cat_features'] = cat_features
    typed_features['dt_features'] = dt_features
    agg_features, nonagg_features = _feature_ctype_inference(
        df, keycol, target)
    typed_features['agg_features'] = agg_features
    typed_features['nonagg_features'] = nonagg_features
    return typed_features


def build_pipeline(args: SimpleParser):
    assert args.sample == 0, f'sample={args.sample} must be in 0'

    features = load_features(args, sort_by=args.sort_by)
    features = nan_processing(features, dropna=True)
    typed_fnames = feature_type_inference(
        features, args.keycol, target=args.target)
    labels = load_labels(args, features[args.keycol].values.tolist())
    Xy = pd.merge(features, labels, on=args.keycol).sort_values(
        by=args.sort_by).drop(columns=typed_fnames['dt_features'])
    X_train, X_test, y_train, y_test, kids_train, kids_test = train_test_split(Xy.drop(
        columns=[args.target]), Xy[args.target], Xy[args.keycol],
        test_size=args.model_test_size, random_state=args.random_state, shuffle=args.split_shuffle)

    # save test X, y, kids
    save_features(X_test, args.outdir_base, 'test_X.csv')
    save_features(y_test, args.outdir_base, 'test_y.csv')
    save_features(kids_test, args.outdir_base, 'test_kids.csv')

    model = create_model(args)
    pipe = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), typed_fnames['num_features']),
            ('cat', OneHotEncoder(sparse_output=False,
             handle_unknown='ignore'), typed_fnames['cat_features']),
            ('drop', 'drop', args.keycol)
        ], remainder='passthrough', verbose=True, verbose_feature_names_out=False)),
        ('model', model),
    ])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, args.pipeline_fpath)

    fnames, fimps = get_feature_importance(args, pipe, X_train, y_train)
    # print(f'fnames = {fnames} \nfimps = {fimps}')
    important_fnames = get_importance_features(
        args, fnames, fimps, X_train.columns.tolist(), topk=10)
    print(f'selected importanct features: {important_fnames}')

    evals = []
    evals.append(evaluate_pipeline(args, pipe, X_train, y_train, 'train'))
    evals.append(evaluate_pipeline(args, pipe, X_test, y_test, 'test'))

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

    return pipe


def load_pipeline(fpath: str):
    return joblib.load(fpath)


if __name__ == '__main__':
    args = SimpleParser().parse_args()
    build_pipeline(args)
