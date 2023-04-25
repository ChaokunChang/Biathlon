from shared import *
set_config(transform_output='pandas')


def apx_value_estimation(df: pd.DataFrame, apx_cols: list, sampling_rate: float = None):
    if sampling_rate is None or sampling_rate == 0:
        return df
    count_names = [x for x in apx_cols if x.startswith('count_')]
    sum_names = [x for x in apx_cols if x.startswith('sum_')]
    df[count_names] = df[count_names].apply(
        lambda x: x/sampling_rate, axis=0)
    df[sum_names] = df[sum_names].apply(
        lambda x: x/sampling_rate, axis=0)
    return df


def _create_regressor(args: SimpleParser):
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
            'verbose': 0,
        }
        model = MLPRegressor(**mlp_params)
    else:
        raise ValueError("model name not supported")

    return model


def _create_classifier(args: SimpleParser):
    if args.model_name == 'lgbm':
        lgb_params = {
            'objective': 'multiclass' if args.multi_class else 'binary',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'random_state': args.random_state,
            'verbose': -1,
        }
        model = lgb.LGBMClassifier(**lgb_params)
    elif args.model_name == 'xgb':
        xgb_params = {
            'objective': 'multi:softmax' if args.multi_class else 'binary:logistic',
            'booster': 'gbtree',
            'learning_rate': 0.1,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = xgb.XGBClassifier(**xgb_params)
    elif args.model_name == 'rf':
        rf_params = {
            'n_estimators': 100,
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = RandomForestClassifier(**rf_params)
    elif args.model_name == 'lr':
        lr_params = {
            'random_state': args.random_state,
            'verbose': 1,
        }
        model = LogisticRegression(**lr_params)
    elif args.model_name == 'dt':
        dt_params = {
            'random_state': args.random_state,
        }
        model = DecisionTreeClassifier(**dt_params)
    elif args.model_name == 'svc':
        svc_params = {
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'auto',
            'coef0': 0.0,
            'tol': 0.001,
            'C': 1.0,
            'shrinking': True,
            'cache_size': 200,
            'verbose': 1,
            'max_iter': 1000,
        }
        model = SVC(**svc_params)
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
        model = KNeighborsClassifier(**knn_params)
    elif args.model_name == 'mlp':
        mlp_params = {
            'hidden_layer_sizes': (100, 100),
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 100,
            'random_state': args.random_state,
            'verbose': 0,
        }
        model = MLPClassifier(**mlp_params)
    else:
        raise ValueError("model name not supported")
    return model


def create_model(args: SimpleParser):
    if args.model_type == 'regressor':
        return _create_regressor(args)
    elif args.model_type == 'classifier':
        return _create_classifier(args)
    else:
        raise ValueError("model type not supported")


def compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True):
    # We can compute feature importance of model's feature, or pipeline's feature.
    if use_pipe_feature:
        return X.columns, permutation_importance(pipe, X, y, n_repeats=10, max_samples=min(1000, len(X)), random_state=random_state, n_jobs=-1).importances_mean
    else:
        X = pipe[:-1].transform(X)
        return X.columns, permutation_importance(pipe[-1], X, y, n_repeats=10, max_samples=min(1000, len(X)), random_state=0, n_jobs=-1).importances_mean


def _get_feature_importance(pipe: Pipeline, X, y):
    model = pipe[-1]
    if isinstance(model, (LGBMRegressor, LGBMClassifier)):
        return model.feature_name_, model.feature_importances_
    elif isinstance(model, (XGBRegressor, XGBClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (LinearRegression, LogisticRegression)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (SVC, SVR)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (KNeighborsRegressor, KNeighborsClassifier)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (MLPRegressor, MLPClassifier)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    else:
        raise ValueError("model name not supported")


def get_feature_importance(pipe: Pipeline, X:pd.DataFrame, y) -> pd.DataFrame:
    fnames, imps = _get_feature_importance(pipe, X, y)
    fimps = pd.DataFrame({'feature': fnames, 'importance': imps})

    fcols = X.columns.to_list()
    # note that the fnames may contain derived features, thus different with fcols
    def _get_fname(x):
        # we use this function to get the pipline input fname
        # given the model input fname
        fname = x['feature']
        if fname in fcols:
            return fname
        else:
            for fcol in fcols:
                if fname.startswith(fcol):
                    return fcol
        return 'other'

    """ in fimps,
        for feature in fcols, keep that row;
        for feature not in fcols, but starts with a col in fols,
        keep that row, and add a new row with feature name as the col name, and importance as the sum of all features that starts with that col name.
        for feature not in fcols, and not starts with a col in fcols,
        keep that row, and add a new row with feature name as 'other', and importance as the sum of all features that not starts with a col in fcols.
        end
    """

    fimps['fname'] = fimps.apply(_get_fname, axis=1)
    fimps = fimps.groupby('fname').agg(
        {'feature': lambda x: '+'.join(x), 'importance': 'sum'}).reset_index()
    fimps.sort_values('importance', ascending=False, inplace=True)
    return fimps


def baseline_expected_default(X_train, X_test, aggcols):
    default_values = X_train[aggcols].mean()
    exp_test = X_test.copy(deep=True)
    for col in aggcols:
        exp_test[col] = default_values[col]
    return exp_test


def _evaluate_regressor_pipeline(args: SimpleParser, pipe: Pipeline, X, y, tag, verbose=False):
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
    return pd.Series([tag, mse, mae, r2, expv, maxe], index=['tag', 'mse', 'mae', 'r2', 'expv', 'maxe'])


def _evaluate_classifier_pipeline(args: SimpleParser, pipe: Pipeline, X, y, tag, verbose=False):
    def __compute(y, y_pred, y_score, average):
        recall = metrics.recall_score(
            y, y_pred, average=average, zero_division=0)
        precision = metrics.precision_score(
            y, y_pred, average=average, zero_division=0)
        f1 = metrics.f1_score(y, y_pred, average=average, zero_division=0)
        if np.unique(y).shape[0] == (y_score.shape[1] if len(y_score.shape) == 2 else 2):
            roc = metrics.roc_auc_score(
                y, y_score, average=average, multi_class='ovr')
        else:
            roc = -1
        return recall, precision, f1, roc

    y_pred = pipe.predict(X)
    y_score = pipe.predict_proba(X)
    if not args.multi_class:
        # TODO: remove the dependency to args
        y_score = y_score[:, 1]

    acc = metrics.accuracy_score(y, y_pred)
    recall, precision, f1, roc = __compute(y, y_pred, y_score, 'macro')
    recall_micro, precision_micro, f1_micro, roc_micro = __compute(
        y, y_pred, y_score, 'micro')
    recall_weighted, precision_weighted, f1_weighted, roc_weighted = __compute(
        y, y_pred, y_score, 'weighted')

    if verbose:
        print(f'evaluate_pipeline: {tag} y_pred.shape={y_pred.shape}')
        print(f'ACC  of {tag} : ', acc)
        print(f'Recall of {tag} : ', recall)
        print(f'Precision of {tag} : ', precision)
        print(f'F1 of {tag} : ', f1)
        print(f'ROC of {tag} : ', roc)
        print(f'Recall Micro of {tag} : ', recall_micro)
        print(f'Precision Micro of {tag} : ', precision_micro)
        print(f'F1 Micro of {tag} : ', f1_micro)
        print(f'ROC Micro of {tag} : ', roc_micro)
        print(f'Recall Weighted of {tag} : ', recall_weighted)
        print(f'Precision Weighted of {tag} : ', precision_weighted)
        print(f'F1 Weighted of {tag} : ', f1_weighted)
        print(f'ROC Weighted of {tag} : ', roc_weighted)
        # evaluation of every class
        print(metrics.classification_report(y, y_pred, zero_division=0))
    return pd.Series([tag, acc, recall, precision, f1, roc, recall_micro, precision_micro, f1_micro, roc_micro, recall_weighted, precision_weighted, f1_weighted, roc_weighted], index=['tag', 'acc', 'recall', 'precision', 'f1', 'roc', 'recall_micro', 'precision_micro', 'f1_micro', 'roc_micro', 'recall_weighted', 'precision_weighted', 'f1_weighted', 'roc_weighted'])


def evaluate_pipeline(args: SimpleParser, pipe: Pipeline, X, y, tag, verbose=False):
    if args.model_type == 'regressor':
        return _evaluate_regressor_pipeline(args, pipe, X, y, tag, verbose)
    elif args.model_type == 'classifier':
        return _evaluate_classifier_pipeline(args, pipe, X, y, tag, verbose)
    else:
        raise ValueError(f'args.model_type={args.model_type} not supported')


def plot_hist_and_save(args: SimpleParser, data, fpath, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    n, bins, patches = ax.hist(data, bins=100)
    n, bins, patches = ax2.hist(
        data, cumulative=1, histtype='step', bins=100, color='tab:orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fpath)
    plt.close()


def nan_processing(df: pd.DataFrame, dropna=True) -> pd.DataFrame:
    if dropna:
        df = df.dropna()
    else:
        df = df.fillna(0)
    assert df.isna().sum().sum() == 0
    return df


def datetime_processing(df: pd.DataFrame, method='drop'):
    # find the datetime cols
    datetime_cols = [
        col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    datetime_cols += [col for col in df.columns if df[col].dtype ==
                      'object' and col.endswith('_datetime')]
    if len(datetime_cols) == 0:
        return df
    if method == 'drop':
        df = df.drop(columns=datetime_cols)
    else:
        raise ValueError(f'methods={method} not supported')
    return df


def load_features(args: SimpleParser, dropna=True, sort_by:str=None, kids=None, cols=None) -> pd.DataFrame:
    features = load_from_csv(args.feature_dir, f'{args.ffile_prefix}.csv')
    if kids is not None:
        features = features[features[args.keycol].isin(kids)]
    if sort_by is not None:
        features = features.sort_values(by=sort_by)
    if cols is not None:
        cols = [args.keycol] + cols if args.keycol not in cols else cols
        features = features[cols]
    features = nan_processing(features, dropna)
    return features


def load_labels(args: SimpleParser, kids=None) -> pd.DataFrame:
    labels = load_from_csv(args.task_dir, f'labels.csv')
    if kids is not None:
        labels = labels[labels[args.keycol].isin(kids)]
    assert labels.isna().sum().sum() == 0, f'labels contains NaN'
    return labels[[args.keycol, args.target]]


def feature_dtype_inference(df: pd.DataFrame, keycol: str, target: str):
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


def is_agg_feature(fname: str):
    return fname.find('_') > 0 and fname.split('_', 1)[0] in SUPPORTED_AGGS


def feature_ctype_inference(cols: list, keycol: str, target: str):
    agg_features = []
    nonagg_features = []
    for col in cols:
        if col == keycol or col == target:
            continue
        # if col starts with '{agg}_', where agg is in [count, avg, sum, var, std, min, max, median], it is an aggregated feature
        if is_agg_feature(col):
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
    num_features, cat_features, dt_features = feature_dtype_inference(
        df, keycol, target)
    typed_features['num_features'] = num_features
    typed_features['cat_features'] = cat_features
    typed_features['dt_features'] = dt_features
    agg_features, nonagg_features = feature_ctype_inference(
        df.columns.to_list(), keycol, target)
    typed_features['agg_features'] = agg_features
    typed_features['nonagg_features'] = nonagg_features
    return typed_features


def load_pipeline(pipe_dir: str, input_name: str = 'pipeline.pkl') -> Pipeline:
    fpath = os.path.join(pipe_dir, input_name)
    return joblib.load(fpath)


def save_pipeline(pipe: Pipeline, pipe_dir: str, output_name: str = 'pipeline.pkl'):
    joblib.dump(pipe, os.path.join(pipe_dir, output_name))
    return None


def build_pipeline(args: SimpleParser):
    assert args.apx_training or args.sample is None, 'either apx_training or sample must not be set'

    features = load_features(
        args, dropna=True, sort_by=args.sort_by, cols=args.fcols)
    labels = load_labels(args, features[args.keycol].values.tolist())

    Xy = pd.merge(features, labels, on=args.keycol, how='left')

    typed_fnames = feature_type_inference(Xy, args.keycol, target=args.target)

    Xy = apx_value_estimation(Xy, typed_fnames['agg_features'], args.sample)
    Xy.drop(columns=typed_fnames['dt_features'], inplace=True)

    X_train, X_test, y_train, y_test, kids_train, kids_test = train_test_split(Xy.drop(
        columns=[args.target]), Xy[args.target], Xy[args.keycol],
        test_size=args.model_test_size, random_state=args.random_state, shuffle=args.split_shuffle)

    # save test X, y, kids
    save_to_csv(X_test, args.pipelines_dir, 'test_X.csv')
    save_to_csv(y_test, args.pipelines_dir, 'test_y.csv')
    save_to_csv(kids_test, args.pipelines_dir, 'test_kids.csv')

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
    save_pipeline(pipe, args.pipelines_dir, 'pipeline.pkl')

    fimps = get_feature_importance(pipe, X_train, y_train)
    save_to_csv(fimps, args.pipelines_dir, 'feature_importance.csv')

    topk_fnames = fimps.sort_values(by='importance', ascending=False).head(
        args.topk_features)['fname'].values.tolist()
    print(f'topk importanct fnames: {topk_fnames}')

    evals = []
    evals.append(evaluate_pipeline(args, pipe, X_train, y_train, 'train'))
    evals.append(evaluate_pipeline(args, pipe, X_test, y_test, 'test'))

    # show evals as pandas dataframe
    evals_df = pd.DataFrame(evals)
    print(evals_df)
    save_to_csv(evals_df, args.pipelines_dir, 'evals.csv')

    plot_hist_and_save(args, y_train, os.path.join(
        args.experiment_dir, 'y_train.png'), 'y_train', 'y', 'count')
    plot_hist_and_save(args, y_test, os.path.join(
        args.experiment_dir, 'y_test.png'), 'y_test', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(X_train), os.path.join(
        args.pipelines_dir, 'y_train_pred.png'), 'y_train_pred', 'y', 'count')
    plot_hist_and_save(args, pipe.predict(X_test), os.path.join(
        args.pipelines_dir, 'y_test_pred.png'), 'y_test_pred', 'y', 'count')

    return pipe


if __name__ == '__main__':
    args = SimpleParser().parse_args()
    # print(args)
    pipe = build_pipeline(args)
    # print(pipe)
