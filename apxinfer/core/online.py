import numpy as np
import pandas as pd
import json
import pickle
from typing import List
import logging
from tqdm import tqdm

# from apxinfer.core.utils import XIPRequest, XIPPredEstimation
from apxinfer.core.feature import evaluate_features
from apxinfer.core.model import evaluate_model
from apxinfer.core.pipeline import XIPPipeline


class OnlineExecutor:
    def __init__(self, ppl: XIPPipeline,
                 working_dir: str,
                 verbose: bool = False) -> None:
        self.ppl = ppl
        self.working_dir = working_dir

        self.logger = logging.getLogger('OnlineExecutor')
        self.verbose = verbose
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

    def preprocess(self, dataset: pd.DataFrame) -> dict:
        cols = dataset.columns.tolist()
        req_cols = [col for col in cols if col.startswith('req_')]
        fcols = [col for col in cols if col.startswith('f_')]
        label_col = 'label'

        requests = dataset[req_cols].to_dict(orient='records')
        labels = dataset[label_col].to_numpy()
        ext_features = dataset[fcols].to_numpy()
        ext_preds = dataset[['ppl_pred']].to_numpy()

        self.requests = requests
        self.labels = labels
        self.ext_features = ext_features
        self.ext_preds = ext_preds

        return {
            'requests': requests,
            'labels': labels,
            'ext_features': ext_features,
            'ext_preds': ext_preds
        }

    def collect_preds(self, requests: List[dict], exact: bool) -> dict:
        preds = []

        qcfgs_list = []
        qcosts_list = []
        nrounds_list = []

        cum_qtimes = []
        cum_pred_times = []
        cum_scheduler_times = []

        for i, request in tqdm(enumerate(requests),
                               desc='Serving requests',
                               total=len(requests),
                               disable=(self.logger.level == logging.DEBUG)):
            self.logger.debug(f'request[{i}]      = {request}')

            xip_pred = self.ppl.serve(request, ret_fvec=True, exact=exact)
            nrounds = len(self.ppl.scheduler.history)
            last_qcfgs = self.ppl.scheduler.history[-1]['qcfgs']
            last_qcosts = self.ppl.scheduler.history[-1]['qcosts']
            preds.append(xip_pred)
            qcfgs_list.append(last_qcfgs)
            qcosts_list.append(last_qcosts)
            nrounds_list.append(nrounds)

            # logging for debugging
            self.logger.debug(f'label[{i}]        = {self.labels[i]}')
            self.logger.debug(f'ext_features[{i}] = {self.ext_features[i]}')
            self.logger.debug(f'ext_pred[{i}]     = {self.ext_preds[i]}')
            self.logger.debug(f'xip_features[{i}] = {xip_pred["fvec"]["fvals"]}')
            self.logger.debug(f'pred[{i}]         = {xip_pred["pred_value"]}')
            self.logger.debug(f'pred[{i}] error   = {xip_pred["pred_error"]}')
            self.logger.debug(f'pred[{i}] conf    = {xip_pred["pred_conf"]}')
            self.logger.debug(f'last_qcfgs[{i}]   = {last_qcfgs}')
            self.logger.debug(f'last_qcosts[{i}]  = {last_qcosts}')
            self.logger.debug(f'nrounds[{i}]      = {nrounds}')

            if not exact:
                cum_qtimes.append(self.ppl.cumulative_qtimes)
                cum_pred_times.append(self.ppl.cumulative_pred_time)
                cum_scheduler_times.append(self.ppl.cumulative_scheduler_time)
                self.logger.debug('Cumulative query time: {}'.format(cum_qtimes[-1]))
                self.logger.debug('Cumulative prediction time: {}'.format(cum_pred_times[-1]))
                self.logger.debug('Cumulative scheduler time: {}'.format(cum_scheduler_times[-1]))

        return {
            'xip_preds': preds,
            'qcfgs_list': qcfgs_list,
            'qcosts_list': qcosts_list,
            'nrounds_list': nrounds_list,
            'cum_qtimes': cum_qtimes,
            'cum_pred_times': cum_pred_times,
            'cum_scheduler_times': cum_scheduler_times
        }

    def evaluate(self, results: dict) -> dict:
        labels = results['labels']
        ext_preds = results['ext_preds']
        xip_preds = results['xip_preds']
        ext_features = results['ext_features']
        xip_features = np.array([pred['fvec']['fvals'] for pred in xip_preds])
        # xip_pred_vals = [pred['pred_val'] for pred in xip_preds]
        xip_pred_errors = [pred['pred_error'] for pred in xip_preds]
        xip_pred_confs = [pred['pred_conf'] for pred in xip_preds]

        # compare xip features against ext features
        fevals = evaluate_features(ext_features, xip_features)
        self.logger.debug(f'fevals = {fevals}')

        # compare xip predictions against ext predictions
        evals_to_ext = evaluate_model(self.ppl.model, xip_features, ext_preds)
        self.logger.debug(f'evals_to_ext = {evals_to_ext}')

        # compare xip predictions against ground truth
        evals_to_gt = evaluate_model(self.ppl.model, xip_features, labels)
        self.logger.debug(f'evals_to_gt = {evals_to_gt}')

        # average error and conf
        avg_error = np.mean(xip_pred_errors)
        avg_conf = np.mean(xip_pred_confs)
        self.logger.debug(f'avg_error = {avg_error}')
        self.logger.debug(f'avg_conf  = {avg_conf}')

        # averge number of rounds
        nrounds_list = results['nrounds_list']
        avg_nrounds = np.mean(nrounds_list)
        self.logger.debug(f'avg_nrounds = {avg_nrounds}')

        # average qsamples and qcosts
        if avg_nrounds == 0:
            avg_sample_each_qry = np.zeros(self.ppl.fextractor.num_queries)
            avg_cost_each_qry = np.zeros(self.ppl.fextractor.num_queries)
            avg_sample = 0
            avg_cost = 0
        else:
            qcfgs_list = results['qcfgs_list']
            qcosts_list = results['qcosts_list']
            avg_sample_each_qry = np.mean([[qcfg['qsample'] for qcfg in qcfgs] for qcfgs in qcfgs_list], axis=0)
            avg_cost_each_qry = np.mean([[qcost['time'] for qcost in qcosts] for qcosts in qcosts_list], axis=0)
            self.logger.debug(f'avg_sample_each_qry = {avg_sample_each_qry}')
            self.logger.debug(f'avg_cost_each_qry   = {avg_cost_each_qry}')
            avg_sample = np.mean(avg_sample_each_qry)
            avg_cost = np.mean(avg_cost_each_qry)
            self.logger.debug(f'avg_sample = {avg_sample}')
            self.logger.debug(f'avg_cost   = {avg_cost}')

        # cumulative times
        cum_qtimes = results['cum_qtimes']
        cum_pred_times = results['cum_pred_times']
        cum_scheduler_times = results['cum_scheduler_times']
        if len(cum_qtimes) > 0:
            avg_cum_qtimes = np.mean(cum_qtimes, axis=0)
            avg_cum_pred_time = np.mean(cum_pred_times)
            avg_cum_scheduler_time = np.mean(cum_scheduler_times)
            self.logger.debug(f'avg_cum_qtime = {avg_cum_qtimes}')
            self.logger.debug(f'avg_cum_pred_time = {avg_cum_pred_time}')
            self.logger.debug(f'avg_cum_scheduler_time = {avg_cum_scheduler_time}')
        else:
            avg_cum_qtimes = np.zeros(self.ppl.fextractor.num_queries)
            avg_cum_pred_time = 0
            avg_cum_scheduler_time = 0

        evals = {
            'evals_to_ext': evals_to_ext,
            'evals_to_gt': evals_to_gt,
            'avg_error': avg_error,
            'avg_conf': avg_conf,
            'avg_nrounds': avg_nrounds,
            'avg_sample': avg_sample,
            'avg_cost': avg_cost,
            'avg_sample_query': avg_sample_each_qry.tolist(),
            'avg_cost_query': avg_cost_each_qry.tolist(),
            'avg_cum_qtimes': avg_cum_qtimes.tolist(),
            'avg_cum_pred_time': avg_cum_pred_time,
            'avg_cum_scheduler_time': avg_cum_scheduler_time,
            'fevals': fevals
        }

        return evals

    def run(self, dataset: pd.DataFrame, exact: bool = False) -> dict:
        self.logger.info('Running online executor')
        results = self.preprocess(dataset)

        collected = self.collect_preds(results['requests'], exact=exact)
        results.update(collected)

        evals = self.evaluate(results)
        results['evals'] = evals

        # save evals to json
        with open(f'{self.working_dir}/evals.json', 'w') as f:
            json.dump(evals, f, indent=4)

        # save results
        with open(f'{self.working_dir}/results.pkl', 'wb') as f:
            pickle.dump(results, f)

        # duplicate to tagged file
        if exact:
            tag = 'exact'
        else:
            if self.verbose:
                tag = 'debug'
            else:
                tag = self.ppl.settings.__str__()
        with open(f'{self.working_dir}/evals_{tag}.json', 'w') as f:
            json.dump(evals, f, indent=4)

        return results
