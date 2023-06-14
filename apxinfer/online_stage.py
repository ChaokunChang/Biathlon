from typing import List, Callable
import numpy as np
import time
from sklearn.pipeline import Pipeline
import itertools
import logging
from apxinfer.utils import get_model_type

logging.basicConfig(level=logging.INFO)


class XIPQuery:
    def __init__(self, key: str, fnames: list, cfgs: list, executor: Callable) -> None:
        """
        executor: a function that takes a request and a configuration, return a list of features and a list of features' estimation
        """
        self.key: str = key
        self.fnames: list = fnames
        self.cfgs_pool: list = cfgs
        self.cfgs_costs: list = [cfg['cost'] for cfg in cfgs]
        self.executor: Callable = executor

    def execute(self, request: dict, cfgid: int) -> dict:
        """ Execute the query given the request and the configuration id
        return a dict of excution results
        """
        st = time.time()
        features, fests = self.executor(request, self.cfgs_pool[cfgid])
        et = time.time()
        return {'fnames': self.fnames, 'features': features, 'fests': fests, 'qtime': et - st}


class FeatureExtractor:
    def __init__(self, queries: List[XIPQuery]) -> None:
        self.queries: List[XIPQuery] = queries
        self.num_queries = len(self.queries)  # number of queries
        self.max_num_cfgs = max([len(qry.cfgs_pool) for qry in self.queries])

        self.all_fnames = list(itertools.chain.from_iterable([qry.fnames for qry in self.queries]))
        # check if there are duplicate feature names
        assert len(self.all_fnames) == len(set(self.all_fnames)), f'Found duplicate feature names: {self.all_fnames}'
        self.num_features = len(self.all_fnames)  # number of features
        self._cache: dict = {}

    def extract(self, request: dict, qcfgs: np.array) -> dict:
        # qcfgs: (num_queries, )
        # return: dict of results

        fnames = []
        features = []
        fests = []
        qtime = 0.0
        for qry_j, cfg_k in enumerate(qcfgs):
            qry_res = self.queries[qry_j].execute(request, cfg_k)
            fnames.extend(qry_res['fnames'])
            features.extend(qry_res['features'])
            fests.extend(qry_res['fests'])
            qtime += qry_res['qtime']

        return {'fnames': fnames, 'features': features, 'fests': fests, 'qtime': qtime}


class OnlineExecutor:
    def __init__(self, fextractor: FeatureExtractor, ppl: Pipeline,
                 target_bound: float, target_conf: float,
                 time_budget: float, max_round: int,
                 seed: int = 0,
                 pest: str = 'monte_carlo',
                 pest_nsamples: int = 1000,
                 allocator: str = 'budget_pconf_delta',
                 logging_level: int = logging.INFO) -> None:
        self.fextractor: FeatureExtractor = fextractor
        self.ppl: Pipeline = ppl

        # for easy usage
        self.target_bound: float = target_bound
        self.target_conf: float = target_conf
        self.time_budget: float = time_budget
        self.max_round: int = max_round

        self.seed = seed
        self.pest = pest
        self.pest_nsamples = pest_nsamples
        self.allocator = allocator

        self.fnames = [list(itertools.chain.from_iterable(qry.fnames)) for qry in self.fextractor.queries]
        self.queries: List[XIPQuery] = self.fextractor.queries
        self.num_queries: int = len(self.queries)  # number of queries
        self.queries_cfgs_pools = [qry.cfgs_pool for qry in self.queries]

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def estimate_features(self, request: dict, qcfgs: np.array) -> dict:
        features_estimation = self.fextractor.extract(request, qcfgs)
        return features_estimation

    def _monte_carlo(self, fmeans: np.array, fscales: np.array) -> dict:
        p = len(fmeans)
        seed = self.seed
        n_samples = self.pest_nsamples

        np.random.seed(seed)

        st = time.time()
        samples = np.random.normal(fmeans, fscales, size=(n_samples, p))
        pred_values = self.ppl.predict(samples)

        if get_model_type(self.ppl) == "regressor":
            pred_value = np.mean(pred_values)
            pred_bound = np.std(pred_values)
            # compute the confidence
            pred_conf = np.mean(np.abs(pred_values - pred_value) <= self.target_bound)
        elif get_model_type(self.ppl) == "classifier":
            pred_value = np.argmax(np.bincount(pred_values))
            pred_bound = 0.0
            pred_conf = np.mean(pred_values == pred_value)
        else:
            raise NotImplementedError
        et = time.time()
        return {'pred_value': pred_value, 'pred_bound': pred_bound, 'pred_conf': pred_conf, 'ptime': et - st}

    def estimate_prediction(self, features_estimation: dict) -> dict:
        features = features_estimation['features']
        fests = features_estimation['fests']
        fscales = np.array([fest[-1] for fest in fests])
        return self._monte_carlo(np.array(features), fscales)

    def get_next_qcfgs_v1(self, qcfgs: np.array, features_estimation: dict, prediction_estimation: dict) -> np.array:
        # determine the next query configuration according to the current query configuration and the estimation of features and prediction
        # qcfgs is the cfg id of each query, the higher id means higher cost. We can only increase cfg id i.e. next_qcfgs >= qcfgs

        # get the cost until now
        prev_cost = time.time() - self.serve_start_time  # option 1: use time
        prev_cost = np.sum(qcfgs)  # option 2: use cfg id
        prev_cost = np.sum([self.queries[qid].cfgs_costs[cfg_id] for qid, cfg_id in enumerate(qcfgs)])  # option 3: use cfg cost

        least_required_budget = 0.0
        for qid, cfg_id in enumerate(qcfgs):
            this_budget = self.queries[qid].cfgs_costs[cfg_id + 1] if cfg_id + 1 < len(self.queries[qid].cfgs_costs) else np.inf
            least_required_budget = min(least_required_budget, this_budget)

        # estimate budget of this round
        factor = 1.0
        new_budget = factor * prev_cost  # exponentially
        # make sure that new_budget is not too large or too small
        assert least_required_budget <= self.time_budget - prev_cost
        new_budget = min(new_budget, self.time_budget - prev_cost)
        new_budget = max(new_budget, least_required_budget)

        # now let's allocate new budget to each query
        # we want to allocate more budget to queries whose uncertainty has higher infuence on the uncertainty of prediction
        # our strategy is to estimate how much the uncertainty of prediction will be reduced if we allocate more budget to a query
        # we use the following formula to estimate the reduction of uncertainty of prediction
        # reduction = new_pred_conf - pred_conf
        # where pred_conf is the uncertainty of prediction before allocating new budget
        # and new_pred_conf is the uncertainty of prediction after allocating new budget
        # we want to allocate more budget to queries whose reduction is higher

        # get the features and fests
        features = features_estimation['features']
        fests = features_estimation['fests']
        fscales = np.array([fest[-1] for fest in fests])

        # get the prediction
        # pred_value = prediction_estimation['pred_value']
        # pred_bound = prediction_estimation['pred_bound']
        pred_conf = prediction_estimation['pred_conf']

        # compute the reduction of uncertainty of prediction if we allocate more budget to each query
        reductions = []
        from_fid = 0
        for qid, cfg_id in enumerate(qcfgs):
            if cfg_id == len(self.queries[qid].cfgs_costs) - 1:
                # this query has already been set as the most expensive configuration
                reductions.append(0.0)
                continue

            f_cnt = len(self.queries[qid].fnames)

            # assume that this query is set as certain, how much the uncertainty of prediction will be reduced
            test_fscales = fscales.copy()
            test_fscales[from_fid:from_fid + f_cnt] = 0.0
            test_pest = self._monte_carlo(features, test_fscales)
            test_pred_conf = test_pest['pred_conf']
            reduction = pred_conf - test_pred_conf
            reductions.append(reduction)

            from_fid += f_cnt

        # allocate the budget to queries according to the reduction of uncertainty of prediction
        # the higher reduction means the higher priority
        # the higher priority means the higher budget
        # the higher budget means the higher cfg id
        # the higher cfg id means the higher cost

        # sort the queries according to the reduction of uncertainty of prediction
        sorted_qids = np.argsort(reductions)[::-1]

        # allocate the budget to queries according to the sorted queries
        new_qcfgs = qcfgs.copy()
        remaining_budget = new_budget
        for qid in sorted_qids:
            while (new_qcfgs[qid] < len(self.queries[qid].cfgs_costs) - 1):
                if self.queries[qid].cfgs_costs[new_qcfgs[qid]] > remaining_budget:
                    break
                new_qcfgs[qid] += 1
            if new_qcfgs[qid] > qcfgs[qid]:
                remaining_budget -= self.queries[qid].cfgs_costs[new_qcfgs[qid]]

        return new_qcfgs

    def get_next_qcfgs_no_budget(self, qcfgs: np.array, features_estimation: dict, prediction_estimation: dict) -> np.array:
        # determine the next query configuration according to the current query configuration and the estimation of features and prediction
        # qcfgs is the cfg id of each query, the higher id means higher cost. We can only increase cfg id i.e. next_qcfgs >= qcfgs
        # In this method, we will choose one query to increase its cfg id according to the reduction of uncertainty of prediction

        # get the features and fests
        features = features_estimation['features']
        fests = features_estimation['fests']
        fscales = np.array([fest[-1] for fest in fests])

        # get the prediction
        # pred_value = prediction_estimation['pred_value']
        # pred_bound = prediction_estimation['pred_bound']
        pred_conf = prediction_estimation['pred_conf']

        # compute the reduction of uncertainty of prediction if we allocate more budget to each query
        reductions = []
        from_fid = 0
        for qid, cfg_id in enumerate(qcfgs):
            if cfg_id == len(self.queries[qid].cfgs_costs) - 1:
                # this query has already been set as the most expensive configuration
                reductions.append(0.0)
                continue

            f_cnt = len(self.queries[qid].fnames)

            # assume that this query is set as certain, how much the uncertainty of prediction will be reduced
            test_fscales = fscales.copy()
            test_fscales[from_fid:from_fid + f_cnt] = 0.0
            test_pest = self._monte_carlo(features, test_fscales)
            test_pred_conf = test_pest['pred_conf']
            reduction = pred_conf - test_pred_conf
            reductions.append(reduction)

            from_fid += f_cnt

        # sort the queries according to the reduction of uncertainty of prediction
        sorted_qids = np.argsort(reductions)[::-1]

        # allocate the budget to queries according to the sorted queries
        next_qcfgs = qcfgs.copy()
        for qid in sorted_qids:
            if (next_qcfgs[qid] < len(self.queries_cfgs_pools[qid]) - 1):
                next_qcfgs[qid] += 1
                break
        return next_qcfgs

    def get_next_qcfgs(self, qcfgs: np.array, features_estimation: dict, prediction_estimation: dict) -> np.array:
        if self.allocator == 'no_budget':
            return self.get_next_qcfgs_no_budget(qcfgs, features_estimation, prediction_estimation)
        elif self.allocator == 'budget_pconf_delta':
            return self.get_next_qcfgs_v1(qcfgs, features_estimation, prediction_estimation)

    def serve(self, request: dict) -> dict:
        self.serve_start_time = time.time()
        qcfgs_pool_size = [len(pool) for pool in self.queries_cfgs_pools]
        init_qcfgs = np.zeros(len(self.queries), dtype=np.int32)

        qcfgs = init_qcfgs.copy()
        pred_bound, pred_conf = 1e9, 0
        for round_id in range(self.max_round - 1):
            features_estimation = self.estimate_features(request, qcfgs)
            prediction_estimation: dict = self.estimate_prediction(features_estimation)
            pred_value, pred_bound, pred_conf = prediction_estimation['pred_value'], prediction_estimation['pred_bound'], prediction_estimation['pred_conf']

            self.logger.debug(f'Round {round_id}: qcfgs={qcfgs}, pred_value={pred_value}, pred_bound={pred_bound}, pred_conf={pred_conf}')

            if np.sum(qcfgs) >= np.sum(qcfgs_pool_size):
                # all features are certain -> prediction is certain.
                assert pred_bound == 0 and pred_conf == 1.0, f'Bug here! pred_bound={pred_bound} != 0 or pred_conf={pred_conf} != 1.0 with exact features'
                break

            if pred_conf >= self.target_conf and pred_bound <= self.target_bound:
                break
            qcfgs = self.get_next_qcfgs(qcfgs, features_estimation, prediction_estimation)

        if not (pred_conf >= self.target_conf and pred_bound <= self.target_bound):
            qcfgs = np.array([len(pool) - 1 for pool in self.queries_cfgs_pools])
            self.logger.debug(f'Refinement Round: qcfgs={qcfgs}')
            features_estimation = self.estimate_features(request, qcfgs)
            prediction_estimation: dict = self.estimate_prediction(features_estimation)
            pred_value, pred_bound, pred_conf = prediction_estimation['pred_value'], prediction_estimation['pred_bound'], prediction_estimation['pred_conf']
            assert pred_bound == 0 and pred_conf == 1.0, f'Bug here! pred_bound={pred_bound} != 0 or pred_conf={pred_conf} != 1.0 with exact features'

        qcosts = np.array([self.queries[qid].cfgs_costs[cfg_id] for qid, cfg_id in enumerate(qcfgs)])
        ret = {"prediction": prediction_estimation, 'features': features_estimation, 'qcfgs': qcfgs, 'qcosts': qcosts}
        return ret

    def serve_exact(self, request: dict) -> dict:
        self.serve_start_time = time.time()
        qcfgs = np.array([len(pool) - 1 for pool in self.queries_cfgs_pools])
        feature_estimations = self.fextractor.extract(request, qcfgs)
        features = feature_estimations['features']
        prediction_estimation = {"pred_value": self.ppl.predict([features])[0],
                                 "pred_bound": 0.,
                                 "pred_conf": 1.0}
        qcosts = np.array([self.queries[qid].cfgs_costs[cfg_id] for qid, cfg_id in enumerate(qcfgs)])
        ret = {"prediction": prediction_estimation, 'features': feature_estimations, 'qcfgs': qcfgs, 'qcosts': qcosts}
        return ret
