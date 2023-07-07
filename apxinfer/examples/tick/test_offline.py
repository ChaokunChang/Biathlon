import os
import pandas as pd
import joblib
import pickle
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable
from tqdm import tqdm
import logging

import matplotlib.pyplot as plt

from apxinfer.core.config import BaseXIPArgs, DIRHelper
from apxinfer.core.utils import (
    XIPRequest,
    XIPPredEstimation,
    XIPQueryConfig,
    XIPExecutionProfile,
)
from apxinfer.core.query import XIPQuery, XIPQType
from apxinfer.core.feature import FEstimatorHelper, XIPFeatureExtractor
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import MCPredictionEstimator
from apxinfer.core.qinfluence import XIPQInfEstimator, XIPQInfEstimatorByFInfs
from apxinfer.core.qcost import XIPQCostModel

from apxinfer.examples.tick.feature import get_fextractor


def load_model(args: BaseXIPArgs) -> XIPModel:
    model_path = DIRHelper.get_model_path(args)
    model = joblib.load(model_path)
    return model


def load_dataset(args: BaseXIPArgs, name: str, num_requests: int = 0) -> pd.DataFrame:
    dataset_dir = DIRHelper.get_dataset_dir(args)
    ds_path = os.path.join(dataset_dir, f"{name}_set.csv")
    dataset = pd.read_csv(ds_path)
    if num_requests > 0:
        dataset = dataset[:num_requests]
    return dataset


class OfflineArgs(BaseXIPArgs):
    num_requests: int = 0  # number of test requests
    n_cfgs: int = 10  # number of query configurations
    verbose: bool = False


class TickOfflineArgs(OfflineArgs):
    plus: bool = False
    multiclass: bool = False


def get_offline_dir(args: OfflineArgs) -> str:
    working_dir = DIRHelper.get_working_dir(args)
    offline_dir = os.path.join(working_dir, "offline", args.model)
    offline_dir = os.path.join(offline_dir, f"nreqs-{args.num_requests}")
    offline_dir = os.path.join(offline_dir, f"ncfgs-{args.ncfgs}")
    os.makedirs(offline_dir, exist_ok=True)
    return offline_dir


class OfflineExecutor:
    def __init__(
        self,
        working_dir: str,
        fextractor: XIPFeatureExtractor,
        nparts: int,
        ncfgs: int,
        verbose: bool = False,
    ) -> None:
        self.fextractor: XIPFeatureExtractor = fextractor
        self.nparts = nparts
        self.ncfgs = ncfgs
        self.verbose = verbose
        self.working_dir = working_dir

        self.logger = logging.getLogger("OfflineExecutor")

    def preprocess(self, dataset: pd.DataFrame) -> dict:
        cols = dataset.columns.tolist()
        req_cols = [col for col in cols if col.startswith("req_")]
        fcols = [col for col in cols if col.startswith("f_")]
        label_col = "label"

        requests = dataset[req_cols].to_dict(orient="records")
        labels = dataset[label_col].to_numpy()
        ext_features = dataset[fcols].to_numpy()

        # self.requests = requests
        # self.labels = labels
        # self.ext_features = ext_features

        return {
            "requests": requests,
            "labels": labels,
            "ext_features": ext_features,
        }

    def collect(self, requests: pd.DataFrame) -> List[List[XIPExecutionProfile]]:
        records = []
        for i, request in tqdm(
            enumerate(requests),
            desc="Serving requests",
            total=len(requests),
            disable=self.verbose,
        ):
            self.logger.debug(f"request[{i}]      = {request}")
            profiles = []
            for cfg_id in tqdm(
                range(self.ncfgs),
                desc="Serving configurations",
                leave=False,
                disable=self.verbose,
            ):
                qcfgs = []
                for qid, qry in enumerate(self.fextractor.queries):
                    if qry.qtype == XIPQType.AGG:
                        sample = cfg_id / self.ncfgs
                    else:
                        sample = 1.0
                    qcfgs.append(
                        qry.get_qcfg(self, cfg_id=cfg_id, sample=sample, offset=0.0)
                    )
                fvec, qcosts = self.fextractor.extract(request, qcfgs)
                profiles.append(
                    XIPExecutionProfile(
                        request=request, qcfgs=qcfgs, fvec=fvec, qcosts=qcosts
                    )
                )
            records.append(profiles)
        return records

    def run(self, dataset: pd.DataFrame) -> dict:
        self.logger.info("Running online executor")
        results = self.preprocess(dataset)
        if os.path.exists(f"{self.working_dir}/records.pkl"):
            with open(f"{self.working_dir}/records.pkl", "rb") as f:
                records = pickle.load(f)
        else:
            records = self.collect(requests=results["requests"])
            with open(f"{self.working_dir}/records.pkl", "wb") as f:
                pickle.dump(results, f)

        agg_qids = []
        agg_fids = []
        cnt = 0
        for qid, qry in enumerate(self.fextractor.queries):
            if qry.qtype == XIPQType.AGG:
                agg_qids.append(qid)
                for i in range(len(qry.fnames)):
                    agg_fids.append(cnt + i)
            cnt += len(qry.fnames)

        fig, ax = plt.subplots(1, len(agg_fids), figsize=(8, 6 * len(agg_fids)))
        for i, profiles in enumerate(records):
            for fid in agg_fids:
                samples = []
                scales = []
                for profile in profiles:
                    samples.append(profile["qcfgs"][agg_qids[0]]["qsample"])
                    scales.append(profile["fvec"]["fests"][fid])
                ax[fid].plot(samples, scales)
        plt.savefig("./tmp.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = TickOfflineArgs().parse_args()

    # load test data
    test_set = load_dataset(args, "valid", args.num_requests)
    verbose = args.verbose and len(test_set) <= 10

    # load xip model
    model = load_model(args)

    # create a feature extractor for this task
    fextractor = get_fextractor(
        args.nparts,
        args.seed,
        disable_sample_cache=False,
        disable_query_cache=False,
        plus=args.plus,
    )
