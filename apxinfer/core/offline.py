import os
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
import logging
import pickle
import joblib
import json

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from apxinfer.core.utils import XIPExecutionProfile, XIPQType
from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.qcost import QueryCostModel, XIPQCostModel


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
        self.model_dir = os.path.join(self.working_dir, "model")
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = logging.getLogger("OfflineExecutor")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.agg_qids = []
        self.agg_fids = []
        self.agg_fnames = []
        self.agg_fid2qid = {}
        cnt = 0
        for qid, qry in enumerate(self.fextractor.queries):
            if qry.qtype == XIPQType.AGG:
                self.agg_qids.append(qid)
                for i in range(len(qry.fnames)):
                    self.agg_fids.append(cnt + i)
                    self.agg_fnames.append(qry.fnames[i])
                    self.agg_fid2qid[cnt + i] = qid
            cnt += len(qry.fnames)

    def preprocess(self, dataset: pd.DataFrame) -> dict:
        cols = dataset.columns.tolist()
        req_cols = [col for col in cols if col.startswith("req_")]
        fcols = [col for col in cols if col.startswith("f_")]
        label_col = "label"

        requests = dataset[req_cols].to_dict(orient="records")
        labels = dataset[label_col].to_numpy()
        ext_features = dataset[fcols].to_numpy()

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
            self.logger.debug(f"request[{i}] = {request}")
            profiles: List[XIPExecutionProfile] = []
            for cfg_id in tqdm(
                range(1, self.ncfgs + 1, 1),
                desc="Serving configurations",
                leave=False,
                disable=self.verbose,
            ):
                qcfgs = []
                for qid, qry in enumerate(self.fextractor.queries):
                    if qry.qtype == XIPQType.AGG:
                        sample = cfg_id * 1.0 / self.ncfgs
                    else:
                        sample = 1.0
                    qcfgs.append(qry.get_qcfg(cfg_id=cfg_id, sample=sample))
                fvec, qcosts = self.fextractor.extract(request, qcfgs)
                if len(profiles) > 0:
                    for i in range(len(qcosts)):
                        qcosts[i]["time"] += profiles[-1]["qcosts"][i]["time"]
                profiles.append(
                    XIPExecutionProfile(
                        request=request, qcfgs=qcfgs, fvec=fvec, qcosts=qcosts
                    )
                )
            records.append(profiles)
        return records

    def plot_fvars(self, qsamples: np.ndarray, fvars: np.ndarray) -> None:
        fig, axes = plt.subplots(
            len(self.agg_fids), 1, figsize=(8, 6 * len(self.agg_fids))
        )
        if len(self.agg_fids) == 1:
            axes = [axes]
        for k in range(len(self.agg_fids)):
            ax: plt.Axes = axes[k]
            fname = self.agg_fnames[k]
            for i in range(len(fvars)):
                ax.plot(qsamples[i, :, k], np.log(fvars[i, :, k]))
            ax.set_title(fname)
        plt.savefig(f"{self.working_dir}/fvars.pdf", bbox_inches="tight")

    def plot_qcost(self, features: pd.DataFrame):
        fig, axes = plt.subplots(
            len(self.agg_qids), 1, figsize=(8, 6 * len(self.agg_fids))
        )
        if len(self.agg_qids) == 1:
            axes = [axes]
        for k, qid in enumerate(self.agg_qids):
            ax: plt.Axes = axes[k]
            qname = self.fextractor.qnames[qid]
            gf = features[features["qid_idx"] == qid]
            ax.scatter(gf["qsample"], gf["tcost"], alpha=0.7)
            ax.set_title(qname)
            ax.set_xlabel("qsample")
            ax.set_ylabel("tcost")
        plt.savefig(f"{self.working_dir}/qtcosts.pdf", bbox_inches="tight")

    def build_cost_model(self, records: List[List[XIPExecutionProfile]]):
        models = []
        for _, qid in enumerate(self.agg_qids):
            qname = self.fextractor.queries[qid].qname
            qcfgs = []
            qcosts = []
            for i, profiles in enumerate(records):
                for j, profile in enumerate(profiles):
                    qcfgs.append(profile["qcfgs"][qid])
                    qcosts.append(profile["qcosts"][qid])
            np.random.seed(0)
            model = QueryCostModel(qname)
            X_train, X_test, y_train, y_test = train_test_split(qcfgs, qcosts)
            model.fit(None, X_train, y_train)
            evals = model.evaluate(None, X_test, y_test)
            slope = model.get_weight()
            evals["slope"] = slope
            print(
                f"q-{qid} {qname}, slope={slope}, mae={evals['mae']}, mape={evals['mape']}"
            )
            model_tag = f"q-{qname}"
            joblib.dump(model, os.path.join(self.model_dir, f"{model_tag}.pkl"))
            with open(
                os.path.join(self.model_dir, f"{model_tag}_evals.json"), "w"
            ) as f:
                json.dump(evals, f, indent=4)
            models.append(model)
        xip_qcm = XIPQCostModel(models)
        joblib.dump(xip_qcm, os.path.join(self.model_dir, "xip_qcm.pkl"))

    def run(self, dataset: pd.DataFrame, clear_cache: bool = False) -> dict:
        self.logger.info("Running offline executor")
        results = self.preprocess(dataset)
        records_path = f"{self.working_dir}/records.pkl"
        if os.path.exists(records_path) and not clear_cache:
            with open(records_path, "rb") as f:
                records = pickle.load(f)
        else:
            records = self.collect(requests=results["requests"])
            with open(records_path, "wb") as f:
                pickle.dump(records, f)

        qsamples = np.zeros((len(records), self.ncfgs, len(self.agg_fids)))
        fvars = np.zeros((len(records), self.ncfgs, len(self.agg_fids)))
        tcosts = np.zeros((len(records), self.ncfgs, len(self.agg_qids)))
        for i, profiles in enumerate(records):
            for j, profile in enumerate(profiles):
                for k, fid in enumerate(self.agg_fids):
                    qsample = profile["qcfgs"][self.agg_fid2qid[fid]]["qsample"]
                    fvar = profile["fvec"]["fests"][fid]
                    qsamples[i][j][k] = qsample
                    fvars[i][j][k] = fvar
                for k, qid in enumerate(self.agg_qids):
                    tcosts[i][j][k] = profile["qcosts"][qid]["time"]
        self.logger.debug(f"qsamples={qsamples[0]}")
        self.logger.debug(f"fvars={fvars[0]}")
        # plot the fvars
        self.plot_fvars(qsamples, fvars)

        self.build_cost_model(records)

        # Convert the array to a pandas DataFrame
        df = pd.DataFrame(
            tcosts.reshape(-1, 1),
            columns=["tcost"],
        )
        # Add two new columns with the cfg and qid indices
        df["req_idx"] = np.repeat(
            np.arange(tcosts.shape[0]), tcosts.shape[1] * tcosts.shape[2]
        )
        df["cfg_idx"] = np.tile(
            np.repeat(np.arange(tcosts.shape[1]), tcosts.shape[2]), tcosts.shape[0]
        )
        df["qid_idx"] = np.tile(
            np.array(self.agg_qids), tcosts.shape[0] * tcosts.shape[1]
        )
        df["qsample"] = df["cfg_idx"] * 1.0 / self.ncfgs
        df.to_csv(os.path.join(self.model_dir, "features.csv"), index=False)

        self.plot_qcost(df)

        return None
