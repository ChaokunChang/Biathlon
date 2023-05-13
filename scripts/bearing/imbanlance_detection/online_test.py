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


class OnlineParser(Tap):
    # path to the task directory
    task = "status_classification"

    model_name: str = 'xgb'  # model name
    model_type: Literal['regressor', 'classifier'] = 'classifier'  # model type
    multi_class: bool = True  # multi class classification

    sample_strategy: str = 'equal'  # sample strategy
    sample_budget_each: float = 0.1  # sample budget each in avg
    low_conf_threshold: float = 0.8  # low confidence threshold

    def process_args(self) -> None:
        self.job_dir: str = os.path.join(
            "/home/ckchang/ApproxInfer/data/bearing", self.task)


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
    sum_sample = sample_each_in_avg * len([qimp for qimp in qimps if qimp > 0])
    qsamples = sum_sample * np.array(qimps) / np.sum(qimps)
    qsamples = [np.round(qsample * 100) / 100 for qsample in qsamples]
    return qsamples


# Root Mean Squared Sum
def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return result

# extract peak-to-peak features


def calculate_p2p(df):
    return np.array(df.max().abs() + df.min().abs())

# extract shannon entropy (cut signals to 500 bins)


def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(stats.entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)
# extract clearence factor


def calculate_clearence(df):
    result = []
    for col in df:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
        result.append(r)
    return result


def compute_agg_function(rows: pd.DataFrame):
    st = time.time()
    mean_abs = rows.abs().mean().iloc[0]
    max_abs = rows.abs().max().iloc[0]
    rms = calculate_rms(rows)[0]
    crest = max_abs / rms
    shape = rms / mean_abs
    impulse = max_abs / mean_abs
    std = rows.std().iloc[0]
    skew = rows.skew().iloc[0]
    kurtosis = rows.kurtosis().iloc[0]
    entropy = calculate_entropy(rows)[0]
    p2p = calculate_p2p(rows)[0]
    clearence = calculate_clearence(rows)[0]
    compute_time = time.time() - st

    return pd.DataFrame([compute_time,
                         mean_abs, rms, max_abs, crest, shape, impulse,
                         std, skew, kurtosis, entropy, p2p, clearence],
                        index=['compute_time',
                               'B_mean', 'B_rms', 'B_max', 'B_crest', 'B_shape', 'B_impulse',
                               'B_std', 'B_skew', 'B_kurtosis', 'B_entropy', 'B_p2p', 'B_clearence']).T


def compute_agg_function_noentropy(rows: pd.DataFrame):
    st = time.time()
    mean_abs = rows.abs().mean().iloc[0]
    max_abs = rows.abs().max().iloc[0]
    rms = calculate_rms(rows)[0]
    crest = max_abs / rms
    shape = rms / mean_abs
    impulse = max_abs / mean_abs
    std = rows.std().iloc[0]
    skew = rows.skew().iloc[0]
    kurtosis = rows.kurtosis().iloc[0]
    p2p = calculate_p2p(rows)[0]
    clearence = calculate_clearence(rows)[0]
    compute_time = time.time() - st

    return pd.DataFrame([compute_time,
                         mean_abs, rms, max_abs, crest, shape, impulse,
                         std, skew, kurtosis, p2p, clearence],
                        index=['compute_time',
                               'B_mean', 'B_rms', 'B_max', 'B_crest', 'B_shape', 'B_impulse',
                               'B_std', 'B_skew', 'B_kurtosis', 'B_p2p', 'B_clearence']).T


def compute_agg_function_easy(rows: pd.DataFrame):
    st = time.time()
    mean_abs = rows.abs().mean().iloc[0]
    rms = calculate_rms(rows)[0]
    shape = rms / mean_abs
    std = rows.std().iloc[0]
    clearence = calculate_clearence(rows)[0]
    compute_time = time.time() - st

    return pd.DataFrame([compute_time,
                         mean_abs, rms, shape,
                         std, clearence],
                        index=['compute_time',
                               'B_mean', 'B_rms', 'B_shape',
                               'B_std', 'B_clearence']).T


def compute_apx_features(job_dir: str, requests: pd.DataFrame, sample_budget_each: float, sample_strategy: str, sample_offset: float = 0) -> pd.DataFrame:
    assert sample_budget_each >= 0 and sample_budget_each <= 1

    feature_importance_path = os.path.join(
        job_dir, 'feature_importances.csv')
    feature_importances = pd.read_csv(feature_importance_path)
    print(
        f'feature_importances={feature_importances[["fname", "importance"]]}')

    # extract features now
    fextraction_time = 0

    # add new column sample to request
    reqs = requests.copy()
    reqs.insert(2, 'sample', sample_budget_each)
    reqs.insert(3, 'sample_offset', sample_offset)

    if job_dir.endswith('easy'):
        aggfunc = compute_agg_function_easy
    elif job_dir.endswith('noentropy'):
        aggfunc = compute_agg_function_noentropy
    else:
        aggfunc = compute_agg_function

    # compute features for bearing dataset
    def bearing_compute(req, aggfunc):
        db_client = DBConnector().client
        sql = """
        WITH toInt32(100*{sample_offset}) AS min_pid, toInt32(100*{sample}) AS max_pid
        SELECT signal
        FROM bearing_online
        WHERE bid = {bid} AND pid >= min_pid AND pid < max_pid AND timestamp = toDateTime('{time}')
        """.format(**req)
        st = time.time()
        rows = db_client.query_df(sql)
        load_time = time.time() - st
        # print(f'rows={rows}')

        agg_feas = aggfunc(rows)
        agg_feas['load_time'] = load_time
        return agg_feas

    st = time.time()
    features = reqs.parallel_apply(
        lambda row: bearing_compute(row, aggfunc).iloc[0], axis=1)
    fextraction_time += time.time() - st
    print(
        f'load cpu time={features["load_time"].sum()}, compute cpu time={features["compute_time"].sum()}')
    print(f'feature extraction only takes {fextraction_time} seconds')
    return features


def get_apx_features(job_dir, requests, sample_budget_each, sample_strategy):
    feature_dir = os.path.join(
        job_dir, 'features', f'sample_{sample_budget_each}')
    os.makedirs(feature_dir, exist_ok=True)
    apx_feature_path = os.path.join(
        feature_dir, f'apx_features_{sample_strategy}.csv')
    if not os.path.exists(apx_feature_path):
        st = time.time()
        apx_features = compute_apx_features(
            job_dir, requests, sample_budget_each, sample_strategy)
        print(f'compute apx_features takes {time.time() - st} seconds')
        apx_features.to_csv(apx_feature_path, index=False)
    else:
        st = time.time()
        apx_features = pd.read_csv(apx_feature_path)
        print(f'load apx_features from disk takes {time.time() - st} seconds')
    return apx_features


def run(args: OnlineParser):
    """ run online experiemnt with automatically sampled features.
    """

    # load the pipeline
    pipeline_path = os.path.join(args.job_dir, 'pipeline.pkl')
    pipe: Pipeline = joblib.load(pipeline_path)

    # load the workload
    requests = pd.read_csv(os.path.join(args.job_dir, 'test_requests.csv'))
    labels = pd.read_csv(os.path.join(args.job_dir, 'test_labels.csv'))

    # load the queries and approximate queries
    # no need to load the exact queries
    selected_features = ['B_mean', 'B_rms', 'B_max', 'B_crest',
                         'B_shape', 'B_impulse', 'B_std', 'B_skew',
                         'B_kurtosis', 'B_entropy', 'B_p2p', 'B_clearence']

    # load exact features as oracle features
    exact_features = pd.read_csv(
        os.path.join(args.job_dir, 'test_features.csv'))
    feature_cols = exact_features.columns

    # get oracle prediction results and time
    st = time.time()
    exact_pred = pipe.predict(exact_features)
    pred_time = time.time() - st
    print(f'model prediction time={pred_time}')

    feature_dir = os.path.join(
        args.job_dir, 'features', f'sample_{args.sample_budget_each}')
    os.makedirs(feature_dir, exist_ok=True)

    # load approximate features
    apx_features = get_apx_features(
        args.job_dir, requests, args.sample_budget_each, args.sample_strategy)

    # get time measurements
    load_cpu_time = apx_features['load_time'].sum()
    compute_cpu_time = apx_features['compute_time'].sum()
    print(
        f'load_cpu_time={load_cpu_time}, compute_cpu_time={compute_cpu_time}')

    # sort the columns of apx_features to match the order of exact_features
    # apx_features = apx_features[exact_features.columns]

    # compute prediction confidence with approximate features
    apx_pred = pipe.predict(apx_features[feature_cols])
    apx_pred_proba = pipe.predict_proba(apx_features[feature_cols]).max(axis=1)

    if args.sample_strategy.startswith('online') and args.sample_budget_each < 1.0:
        # for those prediction with low probability, we increase the sample rate to 1.0 and re-compute these features
        # we only do this for online sampling strategy
        low_conf_threshold = args.low_conf_threshold
        apx_pred_conf_df = pd.DataFrame(
            {'pred': apx_pred, 'conf': apx_pred_proba})
        low_conf_df = apx_pred_conf_df[apx_pred_conf_df['conf']
                                       < low_conf_threshold]
        low_conf_df.to_csv(os.path.join(
            feature_dir, f'low_conf_{args.sample_strategy}.csv'), index=False)
        low_conf_requests = requests.iloc[low_conf_df.index]
        low_conf_feature_path = os.path.join(
            feature_dir, f'low_conf_features_{args.sample_strategy}.csv')
        if os.path.exists(low_conf_feature_path):
            low_conf_features = pd.read_csv(low_conf_feature_path)
        else:
            low_conf_features = compute_apx_features(
                args.job_dir, low_conf_requests, 1.0, args.sample_strategy)
            low_conf_features.to_csv(low_conf_feature_path, index=False)
        load_cpu_time += low_conf_features['load_time'].sum(
        ) - apx_features.iloc[low_conf_df.index]['load_time'].sum()
        compute_cpu_time += low_conf_features['compute_time'].sum()
        # - apx_features.iloc[low_conf_df.index]['compute_time'].sum()
        print(
            f'new load_cpu_time={load_cpu_time}, compute_cpu_time={compute_cpu_time}')
        apx_features.iloc[low_conf_df.index] = low_conf_features
        apx_pred = pipe.predict(apx_features[feature_cols])
        apx_pred_proba = pipe.predict_proba(
            apx_features[feature_cols]).max(axis=1)

    # save the prediction confidence
    is_same = (exact_pred == apx_pred)
    is_label = (exact_pred == labels['class'])
    apx_pred_conf_df = pd.DataFrame(
        {'pred': apx_pred, 'conf': apx_pred_proba, 'is_same': is_same, 'is_label': is_label})
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
                 ['conf'], kde=False, alpha=.4, label='islabel')
    sns.histplot(apx_pred_conf_df[~apx_pred_conf_df['is_label']]
                 ['conf'], kde=False, alpha=.4, label='different')
    plt.legend()
    plt.savefig(os.path.join(
        feature_dir, f'apx_pred_conf_same_diff_{args.sample_strategy}_to_label.png'))
    plt.close()

    # plot the distribution of prediction confidence for the same and different predictions
    plt.figure()
    sns.histplot(apx_pred_conf_df[apx_pred_conf_df['is_same']]
                 ['conf'], kde=False, alpha=.4, label='same')
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
    evals_df['pred_time'] = pred_time
    evals_df['min_conf'] = min_conf
    evals_df['median_conf'] = median_conf
    evals_df['max_conf'] = max_conf
    evals_df['mean_conf'] = mean_conf
    print(evals_df)
    evals_df.to_csv(os.path.join(
        feature_dir, f'evals_{args.sample_strategy}.csv'), index=False)

    # evaluate the approximate features
    fevals = [evaluate_feature(exact_features[fname], apx_features[fname],
                               tag=f'{fname}') for fname in feature_cols]
    fevals_df = pd.DataFrame(fevals)
    fevals_df['load_cpu_time'] = -1
    fevals_df['compute_cpu_time'] = -1
    fevals_df['cpu_time'] = -1

    # insert a new row for the overall evaluation
    overall_eval = fevals_df.mean(numeric_only=True)
    overall_eval['tag'] = 'overall'  # rename the tag as overall
    overall_eval['load_cpu_time'] = load_cpu_time
    overall_eval['compute_cpu_time'] = compute_cpu_time
    overall_eval['cpu_time'] = load_cpu_time + compute_cpu_time
    fevals_df = pd.concat(
        [fevals_df, pd.DataFrame([overall_eval])], ignore_index=True)
    print(fevals_df)
    fevals_df.to_csv(os.path.join(
        feature_dir, f'fevals_{args.sample_strategy}.csv'), index=False)


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    run(args)
