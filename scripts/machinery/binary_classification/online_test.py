from pandarallel import pandarallel
from tap import Tap
from typing import Literal, Tuple
import numpy as np
import pandas as pd
import os
import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
import clickhouse_connect
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names, but (\w+) was fitted with feature names",
)

pandarallel.initialize(progress_bar=False)
# pandarallel.initialize(progress_bar=False, nb_workers=1)

DATA_HOME = "/home/ckchang/ApproxInfer/data"
RESULTS_HOME = "/home/ckchang/ApproxInfer/results2"


class OnlineParser(Tap):
    database = "machinery_more"
    segment_size = 50000

    # path to the task directory
    task = "binary_classification"

    model_name: str = "mlp"  # model name
    model_type: Literal["regressor", "classifier"] = "classifier"  # model type
    multi_class: bool = False  # multi class classification

    num_agg_queries: int = 8  # number of aggregation queries
    max_sample_budget: float = 1.0  # max sample budget each in avg

    init_sample_budget: float = 0.01  # initial sample budget each in avg
    init_sample_policy: Literal["uniform", "fimp"] = "uniform"  # initial sample policy

    feature_estimator: Literal[
        "closed_form", "bootstrap"
    ] = "closed_form"  # feature estimator
    feature_estimation_nsamples: int = 1000  # number of points for feature estimation

    prediction_estimator: Literal[
        "joint_distribution", "independent_distribution", "auto"
    ] = "auto"  # prediction estimator
    prediction_estimator_thresh: float = 1.0  # prediction estimator threshold
    prediction_estimation_nsamples: int = (
        1000  # number of points for prediction estimation
    )

    feature_influence_estimator: Literal[
        "shap", "lime", "auto"
    ] = "auto"  # feature influence estimator
    feature_influence_estimator_thresh: float = (
        1.0  # feature influence estimator threshold
    )
    feature_influence_estimation_nsamples: int = (
        16000  # number of points for feature influence estimation
    )

    # policy to increase sample to budget
    sample_budget: float = 1.0  # sample budget each in avg
    sample_refine_max_niters: int = 0  # nax number of iters to refine the sample budget
    sample_refine_step_policy: Literal[
        "uniform", "exponential", "exponential_rev"
    ] = "uniform"  # sample refine step policy
    sample_allocation_policy: Literal[
        "uniform", "fimp", "finf", "auto"
    ] = "uniform"  # sample allocation policy

    seed = 7077  # random seed
    clear_cache: bool = False  # clear cache

    def process_args(self) -> None:
        self.job_dir: str = os.path.join(
            RESULTS_HOME, self.database, f"{self.task}_{self.model_name}"
        )
        self.feature_dir = os.path.join(self.job_dir, "features")
        self.init_feature_dir = os.path.join(
            self.feature_dir,
            f"init_sample_{self.init_sample_budget}_{self.init_sample_policy}",
            f"feature_estimator_{self.feature_estimator}_{self.feature_estimation_nsamples}",
        )
        self.online_feature_dir = os.path.join(
            self.init_feature_dir,
            f"sample_{self.sample_budget}_{self.sample_allocation_policy}",
            f"refine_{self.sample_refine_max_niters}_{self.sample_refine_step_policy}",
            f"feat_est_{self.feature_estimator}_{self.feature_estimation_nsamples}",
            f"pred_est_{self.prediction_estimator}_{self.prediction_estimator_thresh}_{self.prediction_estimation_nsamples}",
            f"finf_est_{self.feature_influence_estimator}_{self.feature_influence_estimator_thresh}_{self.feature_influence_estimation_nsamples}",
        )
        self.evals_dir = os.path.join(
            self.job_dir,
            "evals",
            f"init_sample_{self.init_sample_budget}_{self.init_sample_policy}",
            f"feature_estimator_{self.feature_estimator}_{self.feature_estimation_nsamples}",
            f"sample_{self.sample_budget}_{self.sample_allocation_policy}",
            f"refine_{self.sample_refine_max_niters}_{self.sample_refine_step_policy}",
            f"feat_est_{self.feature_estimator}_{self.feature_estimation_nsamples}",
            f"pred_est_{self.prediction_estimator}_{self.prediction_estimator_thresh}_{self.prediction_estimation_nsamples}",
            f"finf_est_{self.feature_influence_estimator}_{self.feature_influence_estimator_thresh}_{self.feature_influence_estimation_nsamples}",
        )
        os.makedirs(self.online_feature_dir, exist_ok=True)
        os.makedirs(self.evals_dir, exist_ok=True)

        feat_suffix = f"{self.feature_estimator}_{self.feature_estimation_nsamples}"
        self.init_feat_est_path = os.path.join(
            self.init_feature_dir, f"init_feat_est_{feat_suffix}.csv"
        )
        pred_suffix = f"{feat_suffix}_{self.prediction_estimator}_{self.prediction_estimation_nsamples}"
        self.init_pred_est_path = os.path.join(
            self.init_feature_dir, f"init_pred_est_{pred_suffix}.csv"
        )
        finf_suffix = f"{pred_suffix}_{self.feature_influence_estimator}_{self.feature_influence_estimation_nsamples}"
        self.init_finf_est_path = os.path.join(
            self.init_feature_dir, f"init_finf_est_{finf_suffix}.csv"
        )


class DBConnector:
    def __init__(self, host="localhost", port=0, username="default", passwd="") -> None:
        self.host = host
        self.port = port
        self.username = username
        self.passwd = passwd
        # get current process id for identifying the session
        self.thread_id = os.getpid()
        self.session_time = time.time()
        self.session_id = f"session_{self.thread_id}_{self.session_time}"
        self.client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.passwd,
            session_id=self.session_id,
        )

    def execute(self, sql):
        return self.client.query_df(sql)


def _evaluate_regressor_pipeline(
    args: OnlineParser, ppl: Pipeline, X, y, tag, verbose=False
):
    y_pred = ppl.predict(X)
    mse = metrics.mean_squared_error(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    expv = metrics.explained_variance_score(y, y_pred)
    maxe = metrics.max_error(y, y_pred)
    if verbose:
        print(f"evaluate_pipeline: {tag} y_pred.shape={y_pred.shape}")
        print(f"MSE  of {tag} : ", mse)
        print(f"MAE  of {tag} : ", mae)
        print(f"R2   of {tag} : ", r2)
        print(f"ExpV of {tag} : ", expv)
        print(f"MaxE of {tag} : ", maxe)
    return pd.Series(
        [tag, mse, mae, r2, expv, maxe],
        index=["tag", "mse", "mae", "r2", "expv", "maxe"],
    )


def _evaluate_classifier_pipeline(
    args: OnlineParser, ppl: Pipeline, X, y, tag, verbose=False
):
    def __compute(y, y_pred, y_score, average):
        recall = metrics.recall_score(y, y_pred, average=average, zero_division=0)
        precision = metrics.precision_score(y, y_pred, average=average, zero_division=0)
        f1 = metrics.f1_score(y, y_pred, average=average, zero_division=0)
        # print(f'for roc {np.unique(y).shape}, y.shape={y.shape} y_score.shape={y_score.shape}')
        if np.unique(y).shape[0] == (
            y_score.shape[1] if len(y_score.shape) == 2 else 2
        ):
            roc = metrics.roc_auc_score(y, y_score, average=average, multi_class="ovr")
        else:
            roc = -1
        return recall, precision, f1, roc

    y_pred = ppl.predict(X)
    y_score = ppl.predict_proba(X)
    # print(f'y_score.shape={y_score.shape}')
    if not args.multi_class:
        # TODO: remove the dependency to args
        y_score = y_score[:, 1]

    acc = metrics.accuracy_score(y, y_pred)
    recall, precision, f1, roc = __compute(y, y_pred, y_score, "macro")
    recall_micro, precision_micro, f1_micro, roc_micro = __compute(
        y, y_pred, y_score, "micro"
    )
    recall_weighted, precision_weighted, f1_weighted, roc_weighted = __compute(
        y, y_pred, y_score, "weighted"
    )

    if verbose:
        print(f"evaluate_pipeline: {tag} y_pred.shape={y_pred.shape}")
        print(f"ACC  of {tag} : ", acc)
        print(f"Recall of {tag} : ", recall)
        print(f"Precision of {tag} : ", precision)
        print(f"F1 of {tag} : ", f1)
        print(f"ROC of {tag} : ", roc)
        print(f"Recall Micro of {tag} : ", recall_micro)
        print(f"Precision Micro of {tag} : ", precision_micro)
        print(f"F1 Micro of {tag} : ", f1_micro)
        print(f"ROC Micro of {tag} : ", roc_micro)
        print(f"Recall Weighted of {tag} : ", recall_weighted)
        print(f"Precision Weighted of {tag} : ", precision_weighted)
        print(f"F1 Weighted of {tag} : ", f1_weighted)
        print(f"ROC Weighted of {tag} : ", roc_weighted)
        # evaluation of every class
        print(metrics.classification_report(y, y_pred, zero_division=0))
    return pd.Series(
        [
            tag,
            acc,
            recall,
            precision,
            f1,
            roc,
            recall_micro,
            precision_micro,
            f1_micro,
            roc_micro,
            recall_weighted,
            precision_weighted,
            f1_weighted,
            roc_weighted,
        ],
        index=[
            "tag",
            "acc",
            "recall",
            "precision",
            "f1",
            "roc",
            "recall_micro",
            "precision_micro",
            "f1_micro",
            "roc_micro",
            "recall_weighted",
            "precision_weighted",
            "f1_weighted",
            "roc_weighted",
        ],
    )


def evaluate_pipeline(args: OnlineParser, ppl: Pipeline, X, y, tag, verbose=False):
    if args.model_type == "regressor":
        return _evaluate_regressor_pipeline(args, ppl, X, y, tag, verbose)
    elif args.model_type == "classifier":
        return _evaluate_classifier_pipeline(args, ppl, X, y, tag, verbose)
    else:
        raise ValueError(f"args.model_type={args.model_type} not supported")


def evaluate_feature(y, y_pred, tag, verbose=False):
    mse = metrics.mean_squared_error(y, y_pred)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    expv = metrics.explained_variance_score(y, y_pred)
    maxe = metrics.max_error(y, y_pred)
    if verbose:
        print(f"evaluate_pipeline: {tag} y_pred.shape={y_pred.shape}")
        print(f"MSE  of {tag} : ", mse)
        print(f"MAE  of {tag} : ", mae)
        print(f"R2   of {tag} : ", r2)
        print(f"ExpV of {tag} : ", expv)
        print(f"MaxE of {tag} : ", maxe)
    return pd.Series(
        [tag, mse, mae, r2, expv, maxe],
        index=["tag", "mse", "mae", "r2", "expv", "maxe"],
    )


def allocate_by_weight(
    weights: np.array,
    total_budget: float,
    max_budget_each: float = 1.0,
    allocated: np.array = None,
) -> np.array:
    if allocated is None:
        allocation = np.zeros(len(weights))
    else:
        allocation = allocated.copy()
    allocation_order = np.argsort(weights)[::-1]
    budget = total_budget
    wts = weights.copy()
    for i in allocation_order:
        if np.sum(wts) == 0 or budget <= 0:
            break
        sample = budget * wts[i] / np.sum(wts)
        allocated_sample = min(sample, max_budget_each - allocation[i])
        allocation[i] += allocated_sample
        budget -= allocated_sample
        wts[i] = 0
    if budget > 0:
        # print(
        #     f"Warning: total budget = {total_budget} is not fully allocated with weights={weights} on allocated={allocated}, {budget} is left."
        # )
        # allocate remaining to id that is less than max_budget_each
        for i in allocation_order:
            if budget <= 0:
                break
            if allocation[i] < max_budget_each:
                allocated_sample = min(budget, max_budget_each - allocation[i])
                allocation[i] += allocated_sample
                budget -= allocated_sample
        assert (
            budget <= 1e-9
        ), f"budget={budget} is not fully allocated, allocation={allocation}"
    return allocation


def init_sample_allocatation(args: OnlineParser, requests: pd.DataFrame) -> np.array:
    num_reqs = requests.shape[0]
    num_agg_queries = args.num_agg_queries
    max_budget_each = args.max_sample_budget
    total_budget = args.init_sample_budget * num_agg_queries
    policy = args.init_sample_policy

    if policy == "uniform":
        allocation = (
            np.ones((num_reqs, num_agg_queries)) * total_budget / num_agg_queries
        )
    elif policy == "fimp":
        feature_importance_path = os.path.join(args.job_dir, "feature_importances.csv")
        feature_importances = pd.read_csv(feature_importance_path)
        assert feature_importances.shape[0] == num_agg_queries
        allocation_per_req = allocate_by_weight(
            feature_importances["importance"].values, total_budget, max_budget_each
        )
        allocation = np.ones((num_reqs, num_agg_queries)) * allocation_per_req
    else:
        raise ValueError(f"args.init_sample_policy={policy} not supported")

    print(
        f"init_sample_allocatation: {args.init_sample_policy} allocation[:3]={allocation[:3]}"
    )
    return allocation


def extract_feature(rows: pd.DataFrame) -> np.array:
    # compute mean and return
    return rows.mean().values[0]


def estimate_feature_closed_form(rows: pd.DataFrame) -> np.array:
    # estimate the feature distribution, i.e. compute scale and return
    cnt = rows.shape[0]
    var = rows.var(ddof=1).values
    cnt = np.where(cnt < 1, 1.0, cnt)
    scale = np.sqrt(np.where(cnt >= 50000, 0.0, var) / cnt)
    # if cnt is too small, set scale as big number
    scale = np.where(cnt < 30, 1e9, scale)
    return scale[0]


def estimate_feature(
    rows: pd.DataFrame, feature_estimator: str, feature_estimation_nsamples: int
):
    if feature_estimator == "closed_form":
        feature = extract_feature(rows)
        scale = estimate_feature_closed_form(rows)
        return feature, scale
    elif feature_estimator == "bootstrap":
        feature_samples = []
        for i in range(feature_estimation_nsamples):
            sample = rows.sample(frac=1.0, replace=True)
            feature_samples.append(extract_feature(sample))
        feature_samples = np.array(feature_samples)
        feature = feature_samples.mean()
        scale = feature_samples.std(ddof=1)
        return feature, scale
    else:
        raise ValueError(f"feature_estimator={feature_estimator} not supported")


def machinery_compute(req, feature_estimator: str, feature_estimation_nsamples: int):
    db_client = DBConnector().client
    sql = """
        SELECT sensor_{sensor_id} AS sensor_{sensor_id}
        FROM {database}.sensors_shuffle_sensor_{sensor_id}
        WHERE bid={bid} AND pid >= {noffset} AND pid < ({noffset}+{nsample})
    """.format(
        **req
    )
    st = time.time()
    rows_df = db_client.query_df(sql)
    feature_loading_time = time.time() - st
    feature_loading_nrows = int(rows_df.shape[0])
    # os.system('echo 3 | sudo tee /proc/sys/vm/drop_caches')

    sensor_id = req["sensor_id"]

    # estimate the feature
    st = time.time()
    feature, scale = estimate_feature(
        rows_df, feature_estimator, feature_estimation_nsamples
    )
    feature_estimation_time = time.time() - st

    # return feature, feature_estimation, and feature_loading_time, feature_estimation_time as a pd.Series
    ret = pd.Series(
        [
            feature,
            int(feature_loading_nrows),
            scale,
            feature_loading_time,
            feature_estimation_time,
        ],
        index=[
            f"sensor_{sensor_id}_mean",
            f"sensor_{sensor_id}_feature_loading_nrows",
            f"sensor_{sensor_id}_feature_estimation",
            f"sensor_{sensor_id}_feature_loading_time",
            f"sensor_{sensor_id}_feature_estimation_time",
        ],
    )

    return ret


def compute_apx_features(
    args: OnlineParser, requests: pd.DataFrame, allocation: np.array
) -> pd.DataFrame:
    """iterate the requests and compute the features and estimate the feature distribution for each query
    Remember to allocate samples for each query before executing machinery_compute
    """
    total_chunks = 50000
    num_reqs = requests.shape[0]
    # iterate requests and compute features
    features_list = []  # store extracted features for each request
    for i in tqdm(range(num_reqs), desc="compute_apx_features"):
        req = requests.iloc[i]
        # split into 8 requests, one for each sensor
        reqs = pd.concat([req] * 8, axis=1).T
        # assign database, sensor_id
        reqs["database"] = args.database
        reqs["sensor_id"] = np.arange(8)
        # allocate samples for each request
        reqs["nsample"] = (total_chunks * allocation[i]).astype(int)
        reqs["noffset"] = 0  # TODO add offset for incremental computation
        # compute features
        sensor_features = reqs.parallel_apply(
            lambda row: machinery_compute(
                row, args.feature_estimator, args.feature_estimation_nsamples
            ),
            axis=1,
        )
        # concat features for all sensors
        features = sensor_features.sum(axis=0)
        # add allocation columns for each sensor
        features = pd.concat(
            [
                features,
                pd.Series(
                    allocation[i],
                    index=[f"sensor_{sensor_id}_allocation" for sensor_id in range(8)],
                ),
            ],
            axis=0,
        )

        # add new info for total loading rows, loading time, and estimation time
        features["feature_loading_nrows"] = features[
            [f"sensor_{sensor_id}_feature_loading_nrows" for sensor_id in range(8)]
        ].sum()
        features["feature_loading_time"] = features[
            [f"sensor_{sensor_id}_feature_loading_time" for sensor_id in range(8)]
        ].sum()
        features["feature_estimation_time"] = features[
            [f"sensor_{sensor_id}_feature_estimation_time" for sensor_id in range(8)]
        ].sum()

        features_list.append(features)

    # convert the list of pd.Seriese into pd.DataFrame
    features_df = pd.concat(features_list, axis=1).T
    return features_df


def get_init_apx_features(args: OnlineParser, requests: pd.DataFrame):
    apx_feature_path = args.init_feat_est_path
    if args.clear_cache or not os.path.exists(apx_feature_path):
        # allocate samples for each request
        st = time.time()
        init_allocation = init_sample_allocatation(args, requests)
        init_allocation_time = time.time() - st
        print(f"init_allocation takes {init_allocation_time} seconds")

        # compute features
        st = time.time()
        apx_features = compute_apx_features(args, requests, init_allocation)
        compute_apx_features_time = time.time() - st
        print(f"compute apx_features takes {compute_apx_features_time} seconds")
        apx_features.to_csv(apx_feature_path, index=False)
    else:
        st = time.time()
        apx_features = pd.read_csv(apx_feature_path)
        print(f"load apx_features from disk takes {time.time() - st} seconds")
    return apx_features


def estimate_apx_prediction(
    args: OnlineParser,
    ppl: Pipeline,
    apx_features_w_estimation: pd.DataFrame,
    verbose=False,
) -> pd.DataFrame:
    """
    ppl: pipeline contains the model
    apx_features_w_estimation: apx_features + feature_estimation
    apx_features: all extracted features
    feature_estimation: estimation of features

    every feature follows normal distribution with feature as mean, and variance/cardinality as scale
    We sample n_samples points for each request, each points has p features
    """
    fnames = ppl.feature_names_in_
    means = apx_features_w_estimation[fnames].values
    scales = apx_features_w_estimation[
        [name.replace("_mean", "_feature_estimation") for name in fnames]
    ].values

    np.random.seed(args.seed)
    m, p = means.shape
    n_samples = args.prediction_estimation_nsamples

    # samples \in (n_samples, m, p)
    if args.prediction_estimator == "joint_distribution":
        samples = np.random.normal(means, scales, size=(n_samples, m, p))
    elif args.prediction_estimator == "independent_distribution":
        # copy means n_samlpes times
        samples = np.repeat(means[np.newaxis, :, :], n_samples, axis=0)
        dim_samples = n_samples // p
        for i in range(p):
            start = i * dim_samples
            end = (i + 1) * dim_samples
            samples[start:end, :, i] = np.random.normal(
                means[:, i], scales[:, i], size=(dim_samples, m)
            )
    elif args.prediction_estimator == "auto":
        # half for joint, half for independent
        samples_joint = np.random.normal(means, scales, size=(n_samples // 2, m, p))
        samples_ind = np.repeat(means[np.newaxis, :, :], n_samples // 2, axis=0)
        dim_samples = (n_samples - n_samples // 2) // p
        for i in range(p):
            start = i * dim_samples
            end = (i + 1) * dim_samples
            samples_ind[start:end, :, i] = np.random.normal(
                means[:, i], scales[:, i], size=(dim_samples, m)
            )
        samples = np.concatenate([samples_joint, samples_ind], axis=0)
    else:
        raise ValueError(
            f"prediction_estimator {args.prediction_estimator} not supported"
        )

    # spreds \in (m, n_samples)
    spreds = ppl.predict(samples.reshape(-1, p)).reshape(n_samples, m).T

    # compute the prediction of each request, and the estimation of the prediction
    preds = np.zeros(m)
    preds_estimation = np.zeros(m)
    for i in range(m):
        pred, pred_count = np.unique(spreds[i], return_counts=True)
        preds[i] = pred[np.argmax(pred_count)]
        preds_estimation[i] = np.max(pred_count) / n_samples

    # return prediction with estimation as DataFrame
    return pd.DataFrame({"pred": preds, "pred_estimation": preds_estimation})


def estimate_apx_feature_influence(
    args: OnlineParser,
    ppl: Pipeline,
    apx_features_w_estimation: pd.DataFrame,
    prediction_w_estimation: pd.DataFrame,
    verbose=False,
) -> pd.DataFrame:
    fnames = ppl.feature_names_in_

    means = apx_features_w_estimation[fnames].values
    scales = apx_features_w_estimation[
        [name.replace("_mean", "_feature_estimation") for name in fnames]
    ].values
    apx_preds = prediction_w_estimation["pred"].values

    # (n_samples, m, p)
    np.random.seed(args.seed)
    m, p = means.shape
    n_samples = args.feature_influence_estimation_nsamples

    influences = np.zeros((m, p))
    dim_samples = n_samples // p
    for fid in range(p):
        fid_scales = np.zeros((m, p))
        fid_scales[:, fid] = scales[:, fid]
        samples = np.random.normal(means, scales, size=(dim_samples, m, p))
        # (dim_samples * m, p)
        spreds = ppl.predict(samples.reshape(-1, p)).reshape(dim_samples, m)
        # (m, )
        pconf = (
            np.count_nonzero(spreds.T == apx_preds.reshape(-1, 1), axis=1) / dim_samples
        )
        influences[:, fid] = 1.0 - pconf

    return pd.DataFrame(
        influences, columns=[name.replace("_mean", "_finf") for name in fnames]
    )


def online_sample_allocation(
    args: OnlineParser,
    requests: pd.DataFrame,
    cur_allocation: np.array,
    finfs: pd.DataFrame,
    to_budget: float,
) -> np.array:
    """allocate the sample budget to each request"""
    num_reqs = requests.shape[0]
    num_agg_queries = args.num_agg_queries
    max_budget_each = args.max_sample_budget
    total_budget = to_budget * num_agg_queries
    policy = args.sample_allocation_policy

    if policy == "uniform":
        allocation = np.zeros((num_reqs, num_agg_queries))
        for i in range(num_reqs):
            allocation[i] = allocate_by_weight(
                np.ones(num_agg_queries),
                total_budget - cur_allocation[i].sum(),
                max_budget_each,
                cur_allocation[i],
            )
    elif policy == "fimp":
        feature_importance_path = os.path.join(args.job_dir, "feature_importances.csv")
        feature_importances = pd.read_csv(feature_importance_path)
        assert feature_importances.shape[0] == num_agg_queries
        allocation = np.zeros((num_reqs, num_agg_queries))
        for i in range(num_reqs):
            allocation[i] = allocate_by_weight(
                feature_importances["importance"].values,
                total_budget - cur_allocation[i].sum(),
                max_budget_each,
                cur_allocation[i],
            )
    elif policy == "finf":
        allocation = np.zeros((num_reqs, num_agg_queries))
        for i in range(num_reqs):
            allocation[i] = allocate_by_weight(
                finfs.values[i],
                total_budget - cur_allocation[i].sum(),
                max_budget_each,
                cur_allocation[i],
            )
    else:
        raise ValueError(f"policy={policy} not supported")

    print(f"online_sample_allocation: {policy} allocation[:3]={allocation[:3]}")
    return allocation


def get_online_apx_feature(
    args: OnlineParser,
    requests: pd.DataFrame,
    finfs: pd.DataFrame,
    current_allocation: np.array,
    niters: int,
    iter_id: int,
    to_budget: float,
) -> pd.DataFrame:
    """get the apx_features for each request"""
    feature_dir = args.online_feature_dir
    apx_feature_path = os.path.join(feature_dir, f"online_apx_features_{iter_id}.csv")
    if args.clear_cache or not os.path.exists(apx_feature_path):
        # allocate samples for each request
        st = time.time()
        new_allocation = online_sample_allocation(
            args, requests, current_allocation, finfs, to_budget
        )
        # new_allocation = current_allocation + allocation_inc
        online_allocation_time = time.time() - st
        print(
            f"Iter-{iter_id} online allocation takes {online_allocation_time} seconds"
        )

        # compute features
        st = time.time()
        apx_features = compute_apx_features(args, requests, new_allocation)
        compute_apx_features_time = time.time() - st
        print(
            f"Iter-{iter_id} compute apx_features takes {compute_apx_features_time} seconds"
        )
        apx_features.to_csv(apx_feature_path, index=False)
    else:
        st = time.time()
        apx_features = pd.read_csv(apx_feature_path)
        print(f"load apx_features from disk takes {time.time() - st} seconds")
    return apx_features


def online_refinement(
    args: OnlineParser,
    ppl: Pipeline,
    requests: pd.DataFrame,
    apx_features_w_estimation: pd.DataFrame,
    apx_preds_w_estimation: pd.DataFrame,
    apx_feature_influence: pd.DataFrame,
    verbose=False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float, float]:
    """refine the apx_features_w_estimation, apx_preds_w_estimation, apx_feature_influence"""
    if args.sample_budget <= args.init_sample_budget:
        return (
            apx_features_w_estimation,
            apx_preds_w_estimation,
            apx_feature_influence,
            0.0,
            0.0,
            0.0,
        )

    cur_allocated_budget = args.init_sample_budget
    refinement_budget = args.sample_budget - args.init_sample_budget
    niters = args.sample_refine_max_niters
    sample_refine_step_policy = args.sample_refine_step_policy
    prediction_estimator_thresh = args.prediction_estimator_thresh

    feature_time = 0.0
    prediction_estimation_time = 0.0
    feature_influence_time = 0.0
    iter_id = 0
    to_budget = cur_allocated_budget
    while iter_id < niters and to_budget < args.sample_budget:
        uncertain_idxs = apx_preds_w_estimation[
            apx_preds_w_estimation["pred_estimation"] < prediction_estimator_thresh
        ].index
        if len(uncertain_idxs) == 0:
            print(f"no more uncertain samples, end at iter-{iter_id}/{niters}")
            break

        uncertain_reqs = requests.iloc[uncertain_idxs]
        current_allocation = apx_features_w_estimation.iloc[uncertain_idxs][
            [f"sensor_{sid}_allocation" for sid in range(8)]
        ].values
        uncertain_finfs = apx_feature_influence.iloc[uncertain_idxs][
            [f"sensor_{sid}_finf" for sid in range(8)]
        ]

        # allocate budget for each request
        if iter_id == niters - 1:
            to_budget = args.sample_budget
        else:
            if sample_refine_step_policy == "uniform":
                to_budget += refinement_budget / niters
            elif sample_refine_step_policy == "exponential":
                # allocate exponentially, more on first iterations
                factor = 2
                base = (factor**niters) - 1
                to_budget += (
                    refinement_budget * (factor ** (niters - iter_id - 1)) / base
                )
            elif sample_refine_step_policy == "exponential_rev":
                # allocate exponentially, more on last iterations
                factor = 2
                base = (factor**niters) - 1
                to_budget += refinement_budget * (factor**iter_id) / base
            else:
                raise NotImplementedError()
        print(f"iter-{iter_id} allocate until budget: {to_budget}")

        st = time.time()
        iter_apx_features_w_estimation = get_online_apx_feature(
            args,
            uncertain_reqs,
            uncertain_finfs,
            current_allocation,
            niters,
            iter_id,
            to_budget,
        )
        feature_time += time.time() - st

        st = time.time()
        iter_apx_preds_w_estimation = estimate_apx_prediction(
            args, ppl, iter_apx_features_w_estimation
        )
        prediction_estimation_time += time.time() - st

        st = time.time()
        iter_apx_feature_influence = estimate_apx_feature_influence(
            args, ppl, iter_apx_features_w_estimation, iter_apx_preds_w_estimation
        )
        feature_influence_time += time.time() - st

        print(f"uncertain_idxs X{uncertain_idxs.shape[0]}: {uncertain_idxs}")
        # print(f"uncertain_requests: {uncertain_reqs}")
        # print(f"uncertain_features: {iter_apx_features_w_estimation}")
        # print(f"uncertain_preds: {iter_apx_preds_w_estimation}")
        # print(f"uncertain_finf: {iter_apx_feature_influence}")

        apx_features_w_estimation.iloc[uncertain_idxs] = iter_apx_features_w_estimation
        apx_preds_w_estimation.iloc[uncertain_idxs] = iter_apx_preds_w_estimation
        apx_feature_influence.iloc[uncertain_idxs] = iter_apx_feature_influence

        iter_id += 1

    return (
        apx_features_w_estimation,
        apx_preds_w_estimation,
        apx_feature_influence,
        feature_time,
        prediction_estimation_time,
        feature_influence_time,
    )


def run(args: OnlineParser) -> Tuple[dict, dict]:
    """run online experiemnt with automatically sampled features."""
    time_eval_path = os.path.join(args.evals_dir, "time.csv")
    if os.path.exists(time_eval_path) and not args.clear_cache:
        print(f"skip {args.evals_dir}")
        time_df = pd.read_csv(os.path.join(args.evals_dir, "time.csv"))
        print(time_df)
        fevals_df = pd.read_csv(os.path.join(args.evals_dir, "fevals.csv"))
        print(fevals_df)
        ppl_evals_df = pd.read_csv(os.path.join(args.evals_dir, "ppl_evals.csv"))
        print(ppl_evals_df)
        return {
            "feat_est": pd.read_csv(os.path.join(args.evals_dir, "features.csv")),
            "pred_est": pd.read_csv(os.path.join(args.evals_dir, "preds.csv")),
            "finf_est": pd.read_csv(
                os.path.join(args.evals_dir, "feature_influence.csv")
            ),
        }, {"time_eval": time_df, "feat_eval": fevals_df, "ppl_eval": ppl_evals_df}

    # load the pipeline
    pipeline_path = os.path.join(args.job_dir, "pipeline.pkl")
    ppl: Pipeline = joblib.load(pipeline_path)

    # load the workload
    requests = pd.read_csv(os.path.join(args.job_dir, "test_requests.csv"))
    labels = pd.read_csv(os.path.join(args.job_dir, "test_labels.csv"))

    # load exact features as oracle features
    exact_features = pd.read_csv(os.path.join(args.job_dir, "test_features.csv"))
    feature_cols = exact_features.columns

    # get oracle prediction results and time
    st = time.time()
    exact_pred = ppl.predict(exact_features)
    exact_pred_time = time.time() - st

    # load approximate features with initial allocation
    st = time.time()
    apx_features_w_estimation = get_init_apx_features(args, requests)
    total_feature_time = time.time() - st

    # compute approximate prediction and prediction estimation
    st = time.time()
    apx_preds_w_estimation = estimate_apx_prediction(
        args, ppl, apx_features_w_estimation, verbose=True
    )
    total_prediction_estimation_time = time.time() - st
    # save the prediction results
    apx_preds_w_estimation.to_csv(args.init_pred_est_path, index=False)

    # compute the influence of each feature in each request
    st = time.time()
    apx_feature_influence = estimate_apx_feature_influence(
        args, ppl, apx_features_w_estimation, apx_preds_w_estimation, verbose=True
    )
    total_feature_influence_time = time.time() - st
    # save the feature influence results
    apx_feature_influence.to_csv(args.init_finf_est_path, index=False)

    # refine stage
    (
        apx_features_w_estimation,
        apx_preds_w_estimation,
        apx_feature_influence,
        feature_time,
        prediction_estimation_time,
        feature_influence_time,
    ) = online_refinement(
        args,
        ppl,
        requests,
        apx_features_w_estimation,
        apx_preds_w_estimation,
        apx_feature_influence,
    )
    total_feature_time += feature_time
    total_prediction_estimation_time += prediction_estimation_time
    total_feature_influence_time += feature_influence_time

    total_feature_loading_nrows = apx_features_w_estimation[
        "feature_loading_nrows"
    ].sum()
    total_feature_loading_time = apx_features_w_estimation["feature_loading_time"].sum()
    total_feature_estimation_time = apx_features_w_estimation[
        "feature_estimation_time"
    ].sum()

    # print the measurements
    print(f"exact prediction time:             {exact_pred_time}")
    print(f"total loading nrows:               {total_feature_loading_nrows}")
    print(f"total loading time:                {total_feature_loading_time}")
    print(f"total feature time:                {total_feature_time}")
    print(f"total feature estimation time:     {total_feature_estimation_time}")
    print(f"total prediction estimation time:  {total_prediction_estimation_time}")
    print(f"total feature influence time:      {total_feature_influence_time}")

    # save the results
    evals_dir = args.evals_dir
    os.makedirs(evals_dir, exist_ok=True)

    # save final features, predictions, and feature influence
    apx_features_w_estimation.to_csv(
        os.path.join(evals_dir, "features.csv"), index=False
    )
    apx_preds_w_estimation.to_csv(os.path.join(evals_dir, "preds.csv"), index=False)
    apx_feature_influence.to_csv(
        os.path.join(evals_dir, "feature_influence.csv"), index=False
    )

    # save time measures as dataframe
    total_all_nrows = len(requests) * args.num_agg_queries * 50000
    total_feature_loading_frac = total_feature_loading_nrows / total_all_nrows
    time_df = pd.DataFrame(
        [
            {
                "exact_pred_time": exact_pred_time,
                "total_feature_loading_nrows": total_feature_loading_nrows,
                "total_feature_loading_frac": total_feature_loading_frac,
                "total_feature_loading_time": total_feature_loading_time,
                "total_feature_time": total_feature_time,
                "total_feature_estimation_time": total_feature_estimation_time,
                "total_prediction_estimation_time": total_prediction_estimation_time,
                "total_feature_influence_time": total_feature_influence_time,
            }
        ]
    )
    time_df.to_csv(os.path.join(evals_dir, "time.csv"), index=False)
    print(time_df)

    # evaluate the approximate features and save evals
    fevals = [
        evaluate_feature(
            exact_features[fname], apx_features_w_estimation[fname], tag=f"{fname}"
        )
        for fname in feature_cols
    ]
    fevals_df = pd.DataFrame(fevals)
    # insert a new row for the overall evaluation
    overall_eval = fevals_df.mean(numeric_only=True)
    overall_eval["tag"] = "overall"  # rename the tag as overall
    fevals_df = pd.concat([fevals_df, pd.DataFrame([overall_eval])], ignore_index=True)
    fevals_df.to_csv(os.path.join(evals_dir, "fevals.csv"), index=False)
    print(fevals_df)

    # evaluate the pipeline and save evals
    ppl_evals = []
    ppl_evals.append(evaluate_pipeline(args, ppl, exact_features, labels, "extF-acc"))
    ppl_evals.append(
        evaluate_pipeline(
            args, ppl, apx_features_w_estimation[feature_cols], labels, "apxF-acc"
        )
    )
    ppl_evals.append(
        evaluate_pipeline(
            args, ppl, apx_features_w_estimation[feature_cols], exact_pred, "apxF-sim"
        )
    )
    ppl_evals_df = pd.DataFrame(ppl_evals)
    ppl_evals_df.to_csv(os.path.join(evals_dir, "ppl_evals.csv"), index=False)
    print(ppl_evals_df)

    return {
        "feat_est": apx_features_w_estimation,
        "pred_est": apx_preds_w_estimation,
        "finf_est": apx_feature_influence,
    }, {"time_eval": time_df, "feat_eval": fevals_df, "ppl_eval": ppl_evals_df}


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    run(args)
