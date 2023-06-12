from pandarallel import pandarallel
from tap import Tap
from typing import Literal, Tuple, List, Dict, Union, Callable
import numpy as np
import pandas as pd
import scipy.stats
import os
import time
from sklearn import metrics
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
import clickhouse_connect
import joblib
from tqdm import tqdm
import itertools
import warnings
import logging
from joblib import Memory

from apxinfer.online_stage import XIPQuery, FeatureExtractor, OnlineExecutor

EXP_HOME = '/home/ckchang/.cache/apxinf'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# warnings.filterwarnings('ignore')

joblib_memory = Memory('/tmp/apxinf', verbose=0)
joblib_memory.clear()


class OnlineStageArgs(Tap):
    task: str = 'test'  # task name
    model: str = 'lgbm'  # model name

    max_round: int = 10  # maximum round
    target_conf: float = 1.0  # minimum confidence
    target_bound: float = 0.1  # maximum bound
    time_budget: float = 60  # time budget in seconds

    seed: int = 0  # seed for prediction estimation
    pest_nsamples: int = 1000  # number of samples for prediction estimation

    num_test: int = 0  # number of test requests

    def process_args(self) -> None:
        self.task_dir = os.path.join(EXP_HOME, self.task)
        self.model_dir = os.path.join(self.task_dir, self.model)
        self.exp_dir = os.path.join(self.model_dir, f'seed-{self.seed}')


def get_fextractor() -> FeatureExtractor:

    def test_helper(means, rate):
        nof = len(means)
        scales = [0.1] * nof

        max_nsamples = 1000
        samples = np.random.normal(means, scales, (int(max_nsamples * rate), nof))
        apxf = np.mean(samples, axis=0)
        apxf_std = np.std(samples, axis=0, ddof=1)

        if rate >= 1.0:
            apxf = means
            apxf_std = [0.0] * nof

        return apxf, [('norm', apxf[i], apxf_std[i]) for i in range(nof)]

    @joblib_memory.cache
    def test_executor_q1(request: dict, cfg: dict):
        # reqid = request['request_id']
        req_f1 = request['request_f1']
        # req_f2 = request['request_f2']
        # req_f3 = request['request_f3']
        means = [req_f1]
        return test_helper(means, cfg['sample'])

    @joblib_memory.cache
    def test_executor_q2(request: dict, cfg: dict):
        # reqid = request['request_id']
        # req_f1 = request['request_f1']
        req_f2 = request['request_f2']
        # req_f3 = request['request_f3']
        means = [req_f2]
        return test_helper(means, cfg['sample'])

    @joblib_memory.cache
    def test_executor_q3(request: dict, cfg: dict):
        # reqid = request['request_id']
        # req_f1 = request['request_f1']
        # req_f2 = request['request_f2']
        req_f3 = request['request_f3']
        means = [req_f3]
        return test_helper(means, cfg['sample'])

    cfgs = [
        {'sample': 0.1 * i, 'cost': 0.1 * i}
        for i in range(1, 10 + 1)
    ]
    # print(f'cfgs: {cfgs}')
    # queries = [XIPQuery(key='q', fnames=['feature_1', 'feature_2', 'feature_3'], cfgs=cfgs, executor=test_executor)]
    q1 = XIPQuery(key='q1', fnames=['feature_1'], cfgs=cfgs, executor=test_executor_q1)
    q2 = XIPQuery(key='q2', fnames=['feature_2'], cfgs=cfgs, executor=test_executor_q2)
    q3 = XIPQuery(key='q3', fnames=['feature_3'], cfgs=cfgs, executor=test_executor_q3)
    queries = [q1, q2, q3]
    fextractor = FeatureExtractor(queries)

    return fextractor


def run(args: OnlineStageArgs) -> List[dict]:
    """
    1. load the model
    2. load offline stage results
    3. load test data
    4. for each request serve with the OnlineExecutor.serve
    """

    prepare_dir = os.path.join(args.exp_dir, 'prepare')
    # offline_dir = os.path.join(args.exp_dir, 'offline')
    online_dir = os.path.join(args.exp_dir, 'online')
    os.makedirs(online_dir, exist_ok=True)

    # load pipeline
    ppl_path = os.path.join(prepare_dir, 'pipeline.pkl')
    ppl: Pipeline = joblib.load(ppl_path)

    # load offline stage results
    # offline_results = joblib.load(os.path.join(offline_dir, 'offline_results.pkl'))

    # load test requests, test features, test labels
    test_set = pd.read_csv(os.path.join(prepare_dir, 'test_set.csv'))
    if args.num_test > 0:
        test_set = test_set.sample(args.num_test, random_state=0)
        # debug level
        logging.getLogger().setLevel(logging.DEBUG)

    test_columns = test_set.columns
    requests = test_set[[col for col in test_columns if col.startswith('request_')]].to_dict(orient='records')
    labels = test_set[['request_label']].to_numpy()
    features = test_set[[col for col in test_columns if col.startswith('feature_')]].to_numpy()
    preds = test_set[['ppl_pred']].to_numpy()

    # create a feature extractor for this task
    fextractor = get_fextractor()

    # initialize the online executor
    online_executor = OnlineExecutor(fextractor, ppl,
                                     target_bound=args.target_bound, target_conf=args.target_conf,
                                     time_budget=args.time_budget, max_round=args.max_round,
                                     seed=args.seed)

    # run the online stage
    online_results = []
    for request, feature, label, ext_pred in tqdm(zip(requests, features, labels, preds), desc="serving"):
        logger.debug(f"request: {request}")
        online_result = online_executor.serve(request)
        online_result['label'] = label
        online_result['ext_pred'] = ext_pred
        online_results.append(online_result)

    # save the online stage results
    joblib.dump(online_results, os.path.join(online_dir, 'online_results.pkl'))

    # qcosts in total
    qcosts = np.array([result['qcosts'] for result in online_results])

    # evaluate the online stage results
    apx_pred = [result['prediction']['pred_value'] for result in online_results]
    apx_bound = [result['prediction']['pred_bound'] for result in online_results]
    apx_conf = [result['prediction']['pred_conf'] for result in online_results]

    # compute accuracy of apx_pred to label
    acc = metrics.accuracy_score(labels, apx_pred)

    # compute accuracy of apx_pred to ext_pred
    similarity = metrics.accuracy_score(preds, apx_pred)

    print(f'Online stage accuracy   : {acc}')
    print(f'Online stage similarity : {similarity}')
    print(f'average qcosts per req  : {np.mean(qcosts, axis=0)}')

    return online_results


if __name__ == "__main__":
    args: OnlineStageArgs = OnlineStageArgs().parse_args()
    run(args)
