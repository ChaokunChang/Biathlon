"""
This module is never imported by online_stage.py
it will be used by user to help them run the online stage
"""
import os

import numpy as np
from tqdm import tqdm
from tap import Tap
from typing import Tuple, Literal, List
from sklearn import metrics
from sklearn.pipeline import Pipeline
import logging
import warnings
import pandas as pd
import joblib

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

    pest_nsamples: int = 1000  # number of samples for prediction estimation
    pest: Literal['monte_carlo'] = 'monte_carlo'  # prediction estimation method
    allocator: Literal['budget_pconf_delta', 'no_budget'] = 'budget_pconf_delta'  # allocator type

    num_requests: int = 0  # number of test requests
    max_round: int = 10  # maximum round
    target_conf: float = 1.0  # minimum confidence
    target_bound: float = 0.1  # maximum bound
    time_budget: float = 60  # time budget in seconds


def get_exp_dir(task: str, args: OnlineStageArgs) -> str:
    task_dir = os.path.join(EXP_HOME, task)
    model_dir = os.path.join(task_dir, args.model)
    exp_dir = os.path.join(model_dir, f'seed-{args.seed}')
    return exp_dir


def evaluate_online_results(online_results: list, verbose=False) -> dict:
    # qcosts in total
    qcosts = np.array([result['qcosts'] for result in online_results])

    # evaluate the online stage results
    apx_pred = [result['prediction']['pred_value'] for result in online_results]
    apx_bound = [result['prediction']['pred_bound'] for result in online_results]
    apx_conf = [result['prediction']['pred_conf'] for result in online_results]
    labels = [result['label'] for result in online_results]
    ext_preds = [result['ext_pred'] for result in online_results]

    # compute accuracy and similarity
    acc = metrics.accuracy_score(labels, apx_pred)
    similarity = metrics.accuracy_score(ext_preds, apx_pred)

    # measure estimations
    avg_bound = np.mean(apx_bound)
    avg_conf = np.mean(apx_conf)

    # measure costs
    avg_qcosts = np.mean(qcosts, axis=0)
    avg_qcosts_total = np.mean(qcosts)

    if verbose:
        print(f'Online stage accuracy   : {acc}')
        print(f'Online stage similarity : {similarity}')
        print(f'Average bound per req   : {avg_bound}')
        print(f'Average conf per req    : {avg_conf}')
        print(f'Average qcosts per req  : {avg_qcosts} => {avg_qcosts_total}')

    return {
        'acc': acc,
        'similarity': similarity,
        'avg_bound': avg_bound,
        'avg_conf': avg_conf,
        'avg_qcosts': avg_qcosts,
        'avg_qcosts_total': avg_qcosts_total,
    }


def run_online_executor(online_executor: OnlineExecutor,
                        requests: list, features: np.array,
                        labels: np.array, preds: np.array,
                        exact_version=False) -> Tuple[list, dict]:
    # run the online stage
    online_results = []
    for request, feature, label, ext_pred in tqdm(zip(requests, features, labels, preds), desc="serving"):
        online_logger.debug(f"request: {request}")
        if exact_version:
            online_result = online_executor.serve_exact(request)
        else:
            online_result = online_executor.serve(request)
        online_result['label'] = label
        online_result['ext_pred'] = ext_pred
        online_results.append(online_result)

    # evalute the online results
    evals = evaluate_online_results(online_results, verbose=True)

    return online_results, evals


def run_online_stage(args: OnlineStageArgs, queries: List[XIPQuery], exp_dir: str) -> Tuple[list, dict]:
    """
    1. load the model
    2. load offline stage results
    3. load test data
    4. for each request serve with the OnlineExecutor.serve
    """

    prepare_dir = os.path.join(exp_dir, 'prepare')
    # offline_dir = os.path.join(exp_dir, 'offline')
    online_dir = os.path.join(exp_dir, 'online')
    os.makedirs(online_dir, exist_ok=True)

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
    features = test_set[[col for col in test_columns if col.startswith('feature_')]].to_numpy()
    preds = test_set[['ppl_pred']].to_numpy()

    # create a feature extractor for this task
    fextractor = FeatureExtractor(queries)

    # initialize the online executor
    online_executor = OnlineExecutor(fextractor, ppl,
                                     target_bound=args.target_bound, target_conf=args.target_conf,
                                     time_budget=args.time_budget, max_round=args.max_round,
                                     seed=args.seed, pest=args.pest, pest_nsamples=args.pest_nsamples,
                                     allocator=args.allocator)

    # run the online executor
    online_results, evals = run_online_executor(online_executor, requests, features,
                                                labels, preds,
                                                exact_version=args.max_round == 0)

    return online_results, evals
