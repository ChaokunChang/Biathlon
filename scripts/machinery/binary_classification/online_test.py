from pandarallel import pandarallel
from tap import Tap
from typing import Literal
import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import time
from sklearn import pipeline, metrics
from sklearn.pipeline import Pipeline, make_pipeline
import clickhouse_connect
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

pandarallel.initialize(progress_bar=True, nb_workers=1)
pandarallel.initialize(progress_bar=True)

DATA_HOME = "/home/ckchang/ApproxInfer/data"
RESULTS_HOME = "/home/ckchang/ApproxInfer/results"


class OnlineParser(Tap):
    database = 'machinery_more'
    segment_size = 50000

    # path to the task directory
    task = "binary_classification"

    model_name: str = 'xgb'  # model name
    model_type: Literal['regressor', 'classifier'] = 'classifier'  # model type
    multi_class: bool = False  # multi class classification

    sample_strategy: str = 'equal'  # sample strategy
    sample_budget_each: float = 0.1  # sample budget each in avg
    low_conf_threshold: float = 0.8  # low confidence threshold

    npoints_for_conf: int = 100  # number of points for confidence

    clear_cache: bool = False  # clear cache

    def process_args(self) -> None:
        self.job_dir: str = os.path.join(
            RESULTS_HOME, self.database, f'{self.task}_{self.model_name}')


class DBConnector:
    def __init__(self, host='localhost', port=0, username='default', passwd='') -> None:
        self.host = host
        self.port = port
        self.username = username
        self.passwd = passwd
        # get current process id for identifying the session
        self.thread_id = os.getpid()
        self.session_time = time.time()
        self.session_id = f'session_{self.thread_id}_{self.session_time}'
        self.client = clickhouse_connect.get_client(
            host=self.host, port=self.port,
            username=self.username, password=self.passwd,
            session_id=self.session_id)

    def execute(self, sql):
        # print(f'Executing sql: {sql}')
        return self.client.query_df(sql)


def _evaluate_regressor_pipeline(args: OnlineParser, pipe: Pipeline, X, y, tag, verbose=False):
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


def _evaluate_classifier_pipeline(args: OnlineParser, pipe: Pipeline, X, y, tag, verbose=False):
    def __compute(y, y_pred, y_score, average):
        recall = metrics.recall_score(
            y, y_pred, average=average, zero_division=0)
        precision = metrics.precision_score(
            y, y_pred, average=average, zero_division=0)
        f1 = metrics.f1_score(y, y_pred, average=average, zero_division=0)
        # print(f'for roc {np.unique(y).shape}, y.shape={y.shape} y_score.shape={y_score.shape}')
        if np.unique(y).shape[0] == (y_score.shape[1] if len(y_score.shape) == 2 else 2):
            roc = metrics.roc_auc_score(
                y, y_score, average=average, multi_class='ovr')
        else:
            roc = -1
        return recall, precision, f1, roc

    y_pred = pipe.predict(X)
    y_score = pipe.predict_proba(X)
    # print(f'y_score.shape={y_score.shape}')
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


def evaluate_pipeline(args: OnlineParser, pipe: Pipeline, X, y, tag, verbose=False):
    if args.model_type == 'regressor':
        return _evaluate_regressor_pipeline(args, pipe, X, y, tag, verbose)
    elif args.model_type == 'classifier':
        return _evaluate_classifier_pipeline(args, pipe, X, y, tag, verbose)
    else:
        raise ValueError(f'args.model_type={args.model_type} not supported')


def evaluate_feature(y, y_pred, tag, verbose=False):
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


def allocate_qsamples(sample_each_in_avg: float, qimps: list[float]):
    # allocate sampling rate according to query importance
    # count number of non-zero element in qimps
    sum_sample = sample_each_in_avg * \
        len([qimp for qimp in qimps if qimp >= 0])
    qsamples = sum_sample * np.array(qimps) / np.sum(qimps)
    # qsamples = [np.round(qsample * 10000) / 10000 for qsample in qsamples]
    return qsamples


def compute_apx_features(args: OnlineParser, job_dir: str, requests: pd.DataFrame, sample_budget_each: float, sample_strategy: str, sample_offset: float = 0) -> pd.DataFrame:
    assert sample_budget_each >= 0 and sample_budget_each <= 1
    segment_size = args.segment_size
    # add new column sample to request
    reqs = requests.copy()
    reqs.insert(0, 'nsample', sample_budget_each * segment_size)
    reqs.insert(0, 'noffset', sample_offset * segment_size)

    feature_importance_path = os.path.join(
        job_dir, 'feature_importances.csv')
    feature_importances = pd.read_csv(feature_importance_path)
    print(
        f'feature_importances={feature_importances[["fname", "importance"]]}')

    if sample_strategy.endswith('fimp'):
        qsamples = allocate_qsamples(
            sample_budget_each, feature_importances['importance'].tolist())
    else:
        qsamples = allocate_qsamples(sample_budget_each, [
                                     1.0 if imp > 0 else 0.0 for imp in feature_importances['importance'].tolist()])
    print(f'qsamples={qsamples} -> sum={np.sum(qsamples)}')

    # compute features for machinery dataset
    def machinery_compute(req, database, sensor_id):
        db_client = DBConnector().client
        sql = """
            SELECT avgOrDefault(sensor_{sensor_id}) as sensor_{sensor_id}_mean,
                    countOrDefault(sensor_{sensor_id}) as sensor_{sensor_id}_count,
                    varSampOrDefault(sensor_{sensor_id}) as sensor_{sensor_id}_var
            FROM {database}.sensors_sensor_{sensor_id}
            WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsample})
        """.format(**req, sensor_id=sensor_id, database=database)
        st = time.time()
        rows_df = db_client.query_df(sql)
        load_time = time.time() - st
        # print(f'rows={rows}')
        rows_df[f'load_time_{sensor_id}'] = load_time
        rows_df[f'compute_time_{sensor_id}'] = 0.0
        return rows_df.iloc[0]

    # extract features now
    fextraction_time = 0
    features_list = []
    for sensor_id in range(8):
        st = time.time()
        reqs['nsample'] = qsamples[sensor_id] * segment_size
        features = reqs.parallel_apply(
            lambda row: machinery_compute(row, args.database, sensor_id), axis=1)
        features_list.append(features)
        fextraction_time += time.time() - st
    print(f'feature extraction only takes {fextraction_time} seconds')
    features = pd.concat(features_list, axis=1)
    # load_time = load_time_0 + load_time_1 + load_time_2 + ... + load_time_7
    features['load_time'] = features[[
        f'load_time_{i}' for i in range(8)]].sum(axis=1)
    features['compute_time'] = features[[
        f'compute_time_{i}' for i in range(8)]].sum(axis=1)
    return features


def get_apx_features(args: OnlineParser, job_dir, requests, sample_budget_each, sample_strategy, clear_cache=False):
    feature_dir = os.path.join(
        job_dir, 'features', f'sample_{sample_budget_each}')
    os.makedirs(feature_dir, exist_ok=True)
    apx_feature_path = os.path.join(
        feature_dir, f'apx_features_{sample_strategy}.csv')
    if clear_cache or not os.path.exists(apx_feature_path):
        st = time.time()
        apx_features = compute_apx_features(args,
                                            job_dir, requests, sample_budget_each, sample_strategy)
        print(f'compute apx_features takes {time.time() - st} seconds')
        apx_features.to_csv(apx_feature_path, index=False)
    else:
        st = time.time()
        apx_features = pd.read_csv(apx_feature_path)
        print(f'load apx_features from disk takes {time.time() - st} seconds')
    return apx_features


def get_prediction_confidence(apx_f, apx_se=None, n_samples=100):
    """ compute the confidence interval of prediction according to the apx_f, apx_se
        each sensor feature is distributed following N(apx_feature, apx_se)
        For each request, we sample a set of random points (100x by default) according to the distribution of every sensor feature
        Then we compute the prediction of these random points and compute the confidence interval
        The confidence interval is computed as the 5th and 95th percentile of the prediction of these random points
        The confidence interval is used to compute the confidence of prediction
    """
    pass


def compute_approx_pconf(model, features, variances, cardinalities, n_samples=100, seed=7077):
    """ 
    feature: (m, p) = (m, kc)
    variance: (m, p) = (m, kc)
    cardinality: (m, p) = (m, kc)
    every feature follows normal distribution with feature as mean, and variance/cardinality as variance
    We sample n_samples points for each request, each points has p features
    """
    central_preds = model.predict(features)
    m, p = features.shape
    means = features
    cardinalities = np.where(cardinalities < 30, 1.0, cardinalities)
    scales = np.sqrt(np.where(cardinalities >= 250000,
                     0.0, variances) / cardinalities)
    # (n_samples, m, p)
    np.random.seed(seed)
    samples = np.random.normal(means, scales, size=(n_samples, m, p))
    # (n_samples * m, )
    spreds = model.predict(samples.reshape(-1, p)).reshape(n_samples, m)
    # (m, )
    # assert model.n_classes_ == 2
    # scores = np.mean(spreds, axis=0)
    # pconf = np.where(central_preds >= 0.5, scores, 1 - scores)
    pconf = np.count_nonzero(
        spreds.T == central_preds.reshape(-1, 1), axis=1) / n_samples
    return pconf


def run(args: OnlineParser):
    """ run online experiemnt with automatically sampled features.
    """

    # load the pipeline
    pipeline_path = os.path.join(args.job_dir, 'pipeline.pkl')
    pipe: Pipeline = joblib.load(pipeline_path)

    # load the workload
    requests = pd.read_csv(os.path.join(args.job_dir, 'test_requests.csv'))
    labels = pd.read_csv(os.path.join(args.job_dir, 'test_labels.csv'))

    # load exact features as oracle features
    exact_features = pd.read_csv(
        os.path.join(args.job_dir, 'test_features.csv'))
    feature_cols = exact_features.columns

    # get oracle prediction results and time
    st = time.time()
    exact_pred = pipe.predict(exact_features)
    pred_time = time.time() - st

    feature_dir = os.path.join(
        args.job_dir, 'features', f'sample_{args.sample_budget_each}')
    os.makedirs(feature_dir, exist_ok=True)

    # load approximate features
    apx_features = get_apx_features(args,
                                    args.job_dir, requests, args.sample_budget_each, args.sample_strategy, args.clear_cache)

    # get time measurements
    load_cpu_time = apx_features['load_time'].sum()
    compute_cpu_time = 0.0

    # compute approximate prediction
    apx_pred = pipe.predict(apx_features[feature_cols])

    # compute the standard error of the mean (i.e. approximate features)
    fea_cnt_cols = [f'sensor_{i}_count' for i in range(8)]
    fea_var_cols = [f'sensor_{i}_var' for i in range(8)]

    # compute prediction confidence with approximate features
    # apx_pred_conf = pipe.predict_proba(apx_features[feature_cols]).max(axis=1)
    st = time.time()
    apx_pred_conf = compute_approx_pconf(pipe,
                                         apx_features[feature_cols].values,
                                         apx_features[fea_var_cols].values,
                                         apx_features[fea_cnt_cols].values,
                                         n_samples=args.npoints_for_conf)
    pconf_time = time.time() - st

    if args.sample_strategy.startswith('online') and args.sample_budget_each < 1.0:
        # for those prediction with low probability, we increase the sample rate to 1.0 and re-compute these features
        # we only do this for online sampling strategy
        low_conf_threshold = args.low_conf_threshold
        apx_pred_conf_df = pd.DataFrame(
            {'pred': apx_pred, 'conf': apx_pred_conf})
        low_conf_df = apx_pred_conf_df[apx_pred_conf_df['conf']
                                       < low_conf_threshold]
        low_conf_df.to_csv(os.path.join(
            feature_dir, f'low_conf_{args.sample_strategy}.csv'), index=False)
        low_conf_requests = requests.iloc[low_conf_df.index]
        if len(low_conf_requests) > 0:
            low_conf_feature_path = os.path.join(
                feature_dir, f'low_conf_features_{args.sample_strategy}.csv')
            if os.path.exists(low_conf_feature_path):
                low_conf_features = pd.read_csv(low_conf_feature_path)
            else:
                low_conf_features = compute_apx_features(args,
                                                         args.job_dir, low_conf_requests, 1.0, args.sample_strategy)
                low_conf_features.to_csv(low_conf_feature_path, index=False)
            load_cpu_time += low_conf_features['load_time'].sum(
            ) - apx_features.iloc[low_conf_df.index]['load_time'].sum()
            compute_cpu_time += 0.0
            # - apx_features.iloc[low_conf_df.index]['compute_time'].sum()
            print(
                f'new load_cpu_time={load_cpu_time}, compute_cpu_time={compute_cpu_time}')
            apx_features.iloc[low_conf_df.index] = low_conf_features

            st = time.time()
            apx_pred = pipe.predict(apx_features[feature_cols])
            pred_time += time.time() - st

            # apx_pred_conf = pipe.predict_proba(apx_features[feature_cols]).max(axis=1)
            st = time.time()
            apx_pred_conf = compute_approx_pconf(
                pipe, apx_features[feature_cols].values, apx_features[fea_var_cols].values, apx_features[fea_cnt_cols].values)
            pconf_time += time.time() - st
        else:
            print('no low confidence requests')

    print(f'load_cpu_time    = {load_cpu_time}')
    print(f'compute_cpu_time = {compute_cpu_time}')
    print(f'model_pred_time  = {pred_time}')
    print(f'pcconf_time      = {pconf_time}')

    # save the prediction confidence
    is_same = (exact_pred == apx_pred)
    is_label = (exact_pred == labels['label'])
    apx_pred_conf_df = pd.DataFrame(
        {'pred': apx_pred, 'conf': apx_pred_conf, 'is_same': is_same, 'is_label': is_label})
    apx_pred_conf_df.to_csv(os.path.join(
        feature_dir, f'apx_pred_conf_{args.sample_strategy}.csv'), index=False)
    min_conf = apx_pred_conf_df['conf'].min()
    median_conf = apx_pred_conf_df['conf'].median()
    max_conf = apx_pred_conf_df['conf'].max()
    mean_conf = apx_pred_conf_df['conf'].mean()
    print(
        f'mean_conf={mean_conf}, min_conf={min_conf}, median_conf={median_conf}, max_conf={max_conf}')

    # plot the distribution of prediction confidence
    plt.figure()
    sns.histplot(apx_pred_conf_df['conf'], kde=True, alpha=.4)
    plt.savefig(os.path.join(
        feature_dir, f'apx_pred_conf_{args.sample_strategy}.png'))
    plt.close()

    # plot the distribution of prediction confidence for the same to label
    plt.figure()
    sns.histplot(apx_pred_conf_df[apx_pred_conf_df['is_label']]
                 ['conf'], kde=False, alpha=.4, label='is_label')
    sns.histplot(apx_pred_conf_df[~apx_pred_conf_df['is_label']]
                 ['conf'], kde=False, alpha=.4, label='different')
    plt.legend()
    plt.savefig(os.path.join(
        feature_dir, f'apx_pred_conf_same_diff_{args.sample_strategy}_to_label.png'))
    plt.close()

    # plot the distribution of prediction confidence for the same and different predictions
    plt.figure()
    sns.histplot(apx_pred_conf_df[apx_pred_conf_df['is_same']]
                 ['conf'], kde=False, alpha=.4, label='is_same')
    sns.histplot(apx_pred_conf_df[~apx_pred_conf_df['is_same']]
                 ['conf'], kde=False, alpha=.4, label='different')
    plt.legend()
    plt.savefig(os.path.join(
        feature_dir, f'apx_pred_conf_same_diff_{args.sample_strategy}.png'))
    plt.close()

    # evaluate the pipeline
    evals = []
    evals.append(evaluate_pipeline(
        args, pipe, exact_features, labels, 'extF-acc'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_features[feature_cols], labels, 'apxF-acc'))
    evals.append(evaluate_pipeline(
        args, pipe, apx_features[feature_cols], exact_pred, 'apxF-sim'))
    evals_df = pd.DataFrame(evals)
    evals_df['min_conf'] = min_conf
    evals_df['median_conf'] = median_conf
    evals_df['max_conf'] = max_conf
    evals_df['mean_conf'] = mean_conf
    evals_df['pred_time'] = pred_time
    evals_df['pconf_time'] = pconf_time
    evals_df['feature_time'] = load_cpu_time + compute_cpu_time
    print(evals_df)
    evals_df.to_csv(os.path.join(
        feature_dir, f'evals_{args.sample_strategy}.csv'), index=False)

    # evaluate the approximate features
    fevals = [evaluate_feature(exact_features[fname], apx_features[fname],
                               tag=f'{fname}') for fname in feature_cols]
    fevals_df = pd.DataFrame(fevals)
    fevals_df['load_cpu_time'] = -1.0
    fevals_df['compute_cpu_time'] = -1.0
    fevals_df['feature_time'] = -1.0

    # insert a new row for the overall evaluation
    overall_eval = fevals_df.mean(numeric_only=True)
    overall_eval['tag'] = 'overall'  # rename the tag as overall
    overall_eval['load_cpu_time'] = load_cpu_time
    overall_eval['compute_cpu_time'] = compute_cpu_time
    overall_eval['feature_time'] = load_cpu_time + compute_cpu_time
    fevals_df = pd.concat(
        [fevals_df, pd.DataFrame([overall_eval])], ignore_index=True)
    print(fevals_df)
    fevals_df.to_csv(os.path.join(
        feature_dir, f'fevals_{args.sample_strategy}.csv'), index=False)


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    run(args)
