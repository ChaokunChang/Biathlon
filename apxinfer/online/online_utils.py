"""
This module is never imported by online_stage.py
it will be used by user to help them run the online stage
"""
import os

import numpy as np
from tqdm import tqdm
from tap import Tap
from typing import Tuple, Literal, List, Callable
from sklearn.pipeline import Pipeline
import logging
import warnings
import pandas as pd
import joblib

import apxinfer.utils as xutils
from apxinfer.online_stage import OnlineExecutor, XIPQuery, FeatureExtractor

EXP_HOME = '/home/ckchang/.cache/apxinf'

logging.basicConfig(level=logging.INFO)
online_logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


class OnlineStageArgs(Tap):
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation
    multi_class: bool = False  # whether the task is multi-class classification
    all_features: bool = False  # whether to use all features

    pest_nsamples: int = 1000  # number of samples for prediction estimation
    pest: Literal['monte_carlo'] = 'monte_carlo'  # prediction estimation method
    allocator: Literal['budget_pconf_delta', 'no_budget', 'uniform'] = 'budget_pconf_delta'  # allocator type
    alloc_factor: float = 1.0  # allocation factor

    num_requests: int = 0  # number of test requests
    max_round: int = 10  # maximum round
    target_conf: float = 1.0  # minimum confidence
    target_bound: float = 0.1  # maximum bound
    time_budget: float = 60  # time budget in seconds

    verbose_execution: bool = False  # whether to print execution details


def get_exp_dir(task: str, args: OnlineStageArgs) -> str:
    task_dir = os.path.join(EXP_HOME, task)
    model_dir = os.path.join(task_dir, args.model)
    exp_dir = os.path.join(model_dir, f'seed-{args.seed}')
    return exp_dir


class FEstimator:
    min_cnt = 30

    def estimate_any(data: np.ndarray, p: float, func: Callable, nsamples: int = 100) -> Tuple[np.ndarray, list]:
        if p >= 1.0:
            features = func(data)
            return features, [('norm', features[i], 0.0) for i in range(features.shape[0])]
        cnt = data.shape[0]
        feas = []
        for _ in range(nsamples):
            sample = data[np.random.choice(cnt, size=cnt, replace=True)]
            feas.append(func(sample))
        features = np.mean(feas, axis=0)
        if cnt < FEstimator.min_cnt:
            scales = 1e9 * np.ones_like(features)
        else:
            scales = np.std(feas, axis=0, ddof=1)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]
        return features, fests

    def estimate_min(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.min(x, axis=0))
        return features, fests

    def estimate_max(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.max(x, axis=0))
        return features, fests

    def estimate_median(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.median(x, axis=0))
        return features, fests

    def estimate_stdPop(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))
        return features, fests

    def estimate_stdSamp(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.std(x, axis=0, ddof=0))
        return features, fests

    def estimate_unique(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features, fests = FEstimator.estimate_any(data, p, lambda x : np.array([len(np.unique(x[:, i])) for i in range(x.shape[1])]))
        return features, fests

    def compute_dvars(data: np.ndarray):
        cnt = data.shape[0]
        if cnt < FEstimator.min_cnt:
            # if cnt is too small, set scale as big number
            return 1e9 * np.ones_like(data[0])
        else:
            return np.var(data, axis=0, ddof=1)

    def compute_closed_form_scale(features: np.ndarray, cnt: int, dvars: np.ndarray, p: float) -> np.ndarray:
        cnt = np.where(cnt < 1, 1.0, cnt)
        scales = np.sqrt(np.where(p >= 1.0, 0.0, dvars) / cnt)
        return scales

    def estimate_avg(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        cnt = data.shape[0]
        features = np.mean(data, axis=0)
        dvars = FEstimator.compute_dvars(data)
        scales = FEstimator.compute_closed_form_scale(features, cnt, dvars, p)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]
        return features, fests

    def estimate_count(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        cnt = data.shape[0]
        features = np.array([cnt / p])
        scales = FEstimator.compute_closed_form_scale(features, cnt, np.array([cnt * (1 - p) * p]), p)
        fests = [('norm', features[0], scales[0])]
        return features, fests

    def estimate_sum(data: np.ndarray, p: float) -> Tuple[np.ndarray, list]:
        features = np.sum(data, axis=0) / p
        cnt = data.shape[0]
        dvars = FEstimator.compute_dvars(data)
        scales = FEstimator.compute_closed_form_scale(features, cnt, cnt * cnt * dvars, p)
        fests = [('norm', features[i], scales[i]) for i in range(features.shape[0])]

        return features, fests

    def merge_ffests(ffests: list) -> list:
        # ffests: list of tuple[np.ndarray, list[tuple]]
        # return: tuple(np.ndarray, list[tuple])
        features = np.concatenate([ffest[0] for ffest in ffests], axis=0)
        fests = []
        for _, fest in ffests:
            fests.extend(fest)
        return features, fests


def get_default_cfgs(chunks=10) -> List[dict]:
    granularity = 1.0 / chunks
    cfgs = [
        {'sample': granularity * i, 'cost': granularity * i}
        for i in range(1, chunks + 1)
    ]
    return cfgs


def evaluate_classifier_online_results(online_results: list, verbose=False) -> dict:
    # qcosts in total
    qcosts = np.array([result['qcosts'] for result in online_results])

    # evaluate the online stage results
    apx_pred = [result['prediction']['pred_value'] for result in online_results]
    apx_bound = [result['prediction']['pred_bound'] for result in online_results]
    apx_conf = [result['prediction']['pred_conf'] for result in online_results]
    labels = [result['label'] for result in online_results]
    ext_preds = [result['ext_pred'] for result in online_results]

    # compare apx features and ext features
    ext_features = np.array([result['ext_features'] for result in online_results])
    apx_features = np.array([result['features']['features'] for result in online_results])
    fevals = xutils.evaluate_features(ext_features, apx_features)

    # compute accuracy and similarity
    metrics_to_label = xutils.evaluate_classifier(labels, apx_pred)
    metrics_to_ext = xutils.evaluate_classifier(ext_preds, apx_pred)

    # measure estimations
    avg_bound = np.mean(apx_bound)
    avg_conf = np.mean(apx_conf)

    # measure costs
    avg_qcosts = np.mean(qcosts, axis=0)
    avg_qcosts_total = np.mean(qcosts)

    if verbose:
        print(f'Online stage accuracy   : {metrics_to_label["acc"]}')
        print(f'Online stage similarity : {metrics_to_ext["acc"]}')
        print(f'Average feature mae     : {fevals["mae"]}')
        print(f'Average bound per req   : {avg_bound}')
        print(f'Average conf per req    : {avg_conf}')
        print(f'Average qcosts per req  : {avg_qcosts} => {avg_qcosts_total}')

    return {
        'metrics_to_label': metrics_to_label,
        'metrics_to_ext': metrics_to_ext,
        'avg_bound': avg_bound,
        'avg_conf': avg_conf,
        'avg_qcosts': avg_qcosts.tolist(),
        'avg_qcosts_total': avg_qcosts_total,
        'fevals': fevals
    }


def evaluate_regressor_online_results(online_results: list, verbose=False) -> dict:
    # qcosts in total
    qcosts = np.array([result['qcosts'] for result in online_results])

    # evaluate the online stage results
    apx_pred = [result['prediction']['pred_value'] for result in online_results]
    apx_bound = [result['prediction']['pred_bound'] for result in online_results]
    apx_conf = [result['prediction']['pred_conf'] for result in online_results]
    labels = [result['label'] for result in online_results]
    ext_preds = [result['ext_pred'] for result in online_results]

    # compare apx features and ext features
    ext_features = np.array([result['ext_features'] for result in online_results])
    apx_features = np.array([result['features']['features'] for result in online_results])
    fevals = xutils.evaluate_features(ext_features, apx_features)

    # compute accuracy and similarity
    metrics_to_label = xutils.evaluate_regressor(labels, apx_pred)
    metrics_to_ext = xutils.evaluate_regressor(ext_preds, apx_pred)

    # measure estimations
    avg_bound = np.mean(apx_bound)
    avg_conf = np.mean(apx_conf)

    # measure costs
    avg_qcosts = np.mean(qcosts, axis=0)
    avg_qcosts_total = np.mean(qcosts)

    if verbose:
        print(f'Online stage mse2label  : {metrics_to_label["mae"]}')
        print(f'Online stage mse2ext    : {metrics_to_ext["mae"]}')
        print(f'Average feature mae     : {fevals["mae"]}')
        print(f'Average bound per req   : {avg_bound}')
        print(f'Average conf per req    : {avg_conf}')
        print(f'Average qcosts per req  : {avg_qcosts} => {avg_qcosts_total}')

    return {
        'metrics_to_label': metrics_to_label,
        'metrics_to_ext': metrics_to_ext,
        'avg_bound': avg_bound,
        'avg_conf': avg_conf,
        'avg_qcosts': avg_qcosts.tolist(),
        'avg_qcosts_total': avg_qcosts_total,
        'fevals': fevals
    }


def run_online_executor(online_executor: OnlineExecutor,
                        requests: list, features: np.ndarray,
                        labels: np.ndarray, preds: np.ndarray,
                        exact_version=False) -> Tuple[List[dict], dict]:
    # run the online stage
    online_results = []
    for request, feature, label, ext_pred in tqdm(zip(requests, features, labels, preds),
                                                  desc="serving",
                                                  total=len(requests),
                                                  disable=(online_logger.level == logging.DEBUG)):
        online_logger.debug(f"request: {request}")
        if exact_version:
            online_result = online_executor.serve_exact(request)
        else:
            online_result = online_executor.serve(request)
        online_result['label'] = label
        online_result['ext_pred'] = ext_pred
        online_result['ext_features'] = feature.tolist()
        online_results.append(online_result)
        online_logger.debug(f"label={label}, ext_pred={ext_pred}")
        online_logger.debug(f"online pred     : {online_result['prediction']}")
        online_logger.debug(f"ext    features : {online_result['ext_features']}")
        online_logger.debug(f"online features : {online_result['features']['features']}")

    # evalute the online results
    if xutils.get_model_type(online_executor.ppl) == 'classifier':
        evals = evaluate_classifier_online_results(online_results, verbose=True)
    elif xutils.get_model_type(online_executor.ppl) == 'regressor':
        evals = evaluate_regressor_online_results(online_results, verbose=True)
    else:
        raise ValueError(f"Unknown model type: {xutils.get_model_type(online_executor.ppl)}")

    return online_results, evals


def run_online_stage(args: OnlineStageArgs, queries: List[XIPQuery], exp_dir: str) -> Tuple[List[dict], dict]:
    """
    1. load the model
    2. load offline stage results
    3. load test data
    4. for each request serve with the OnlineExecutor.serve
    """

    prepare_dir = os.path.join(exp_dir, 'prepare')
    # offline_dir = os.path.join(exp_dir, 'offline')

    # load pipeline
    ppl_path = os.path.join(prepare_dir, 'pipeline.pkl')
    ppl: Pipeline = joblib.load(ppl_path)

    # load offline stage results
    # offline_results = joblib.load(os.path.join(offline_dir, 'offline_results.pkl'))

    # load test requests, test features, test labels
    test_set = pd.read_csv(os.path.join(prepare_dir, 'test_set.csv'))
    if args.num_requests > 0:
        test_set = test_set.sample(args.num_requests, random_state=0)
        # debug level
        online_logger.setLevel(logging.DEBUG)

    test_columns = test_set.columns
    requests = test_set[[col for col in test_columns if col.startswith('request_')]].to_dict(orient='records')
    labels = test_set[['request_label']].to_numpy()
    features = test_set[[col for col in test_columns if col.startswith('f_')]].to_numpy()
    preds = test_set[['ppl_pred']].to_numpy()

    # create a feature extractor for this task
    fextractor = FeatureExtractor(queries)

    # initialize the online executor
    allocator_params = {'factor': args.alloc_factor}
    online_executor = OnlineExecutor(fextractor, ppl,
                                     target_bound=args.target_bound, target_conf=args.target_conf,
                                     time_budget=args.time_budget, max_round=args.max_round,
                                     seed=args.seed, pest=args.pest, pest_nsamples=args.pest_nsamples,
                                     allocator=args.allocator,
                                     allocator_params=allocator_params,
                                     logging_level=logging.DEBUG if args.verbose_execution else logging.INFO)

    # run the online executor
    online_results, evals = run_online_executor(online_executor, requests, features,
                                                labels, preds,
                                                exact_version=args.max_round == 0)

    return online_results, evals
