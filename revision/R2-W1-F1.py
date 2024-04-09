import os
import sys
import numpy as np
import json
import math
import time
from typing import List, Tuple
from tap import Tap
from tqdm import tqdm
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Bbox
import seaborn as sns
from sklearn import metrics
from scipy import stats
import warnings
import gc

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from apxinfer.core.config import OnlineArgs
from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.utils import is_same_float, XIPQType, XIPFeatureVec
from apxinfer.core.data import DBHelper
from apxinfer.core.prediction import BiathlonPredictionEstimator
from apxinfer.core.pipeline import XIPPipeline
from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS
from apxinfer.examples.run import get_ppl

from apxinfer.simulation import utils as simutils


def get_tag(args: OnlineArgs, max_seed: int):
    task_home, task_name = args.task.split("/")
    tag = "_".join([task_name, f"{args.nreqs_offset}", f"{args.nreqs}", f"{max_seed}"])
    return tag


def run(args: OnlineArgs, seeds_list: np.ndarray, save_dir: str, logfile: str) -> dict:
    task_home, task_name = args.task.split("/")
    assert task_home == "final"

    res_dir = os.path.join(save_dir, "results", "2.1.1")
    os.makedirs(res_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    res_path = os.path.join(res_dir, f"res_{tag}.pkl")
    if os.path.exists(res_path):
        print(f"Load results from {res_path}")
        return joblib.load(res_path)

    test_set: pd.DataFrame = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    req_cols = [col for col in test_set.columns if col.startswith("req_")]

    ppl: XIPPipeline = get_ppl(task_name, args, test_set, verbose=False)
    if args.verbose:
        log_dir = os.path.join(save_dir, "log")
        simutils.set_logger(logging.DEBUG, os.path.join(log_dir, logfile))

    online_dir = DIRHelper.get_online_dir(args)
    tag = ppl.settings.__str__()
    df_path = os.path.join(online_dir, f"final_df_{tag}.csv")
    assert os.path.exists(df_path), f"File not found: {df_path}"
    print(f"Loading {df_path}")
    df = pd.read_csv(df_path)
    df = df.iloc[: args.nreqs]

    fnames = ppl.fextractor.fnames
    noq = ppl.fextractor.num_queries
    qsample_cols = [f"qsamples_{i}" for i in range(noq)]

    fvecs_list = []
    oracle_fvec_list = []
    moments_list = []
    requests = test_set[req_cols].to_dict(orient="records")
    for rid, request in tqdm(
        enumerate(requests), total=len(requests), desc="Processing requests"
    ):
        qsamples = df.loc[rid, qsample_cols].to_numpy()
        qnparts = (qsamples * 100).astype(int)

        # run exact, make ncores = 0 and loading_mode=1 to be faster
        ppl.fextractor.ncores = 0
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 1
            qry.set_enable_qcache()
            qry.set_enable_dcache()
            qry.profiles = []

        oracle_pred = ppl.serve(request=request, exact=True)
        oracle_fvec_list.append(oracle_pred["fvec"])

        rrdatas = []
        warnings.simplefilter("ignore", RuntimeWarning)
        for qid, qry in enumerate(ppl.fextractor.queries):
            if qry.qtype == XIPQType.AGG:
                cached_rrd: np.ndarray = qry._dcache["cached_rrd"]
                rrdatas.append(cached_rrd)
                # print(f"{rid}-{qid}: {cached_rrd.shape}, {np.mean(cached_rrd)}")
                if cached_rrd is not None:
                    moments_list.append(
                        {
                            "rid": rid,
                            "qid": qid,
                            "size": len(cached_rrd),
                            "mean": np.mean(cached_rrd),
                            "std": np.std(cached_rrd),
                            "skew": stats.skew(cached_rrd),
                            "kurtosis": stats.kurtosis(cached_rrd),
                        }
                    )
                else:
                    moments_list.append(
                        {
                            "rid": rid,
                            "qid": qid,
                            "mean": None,
                            "std": None,
                            "skew": None,
                            "kurtosis": None,
                        }
                    )
            else:
                rrdatas.append(None)

        ppl.fextractor.ncores = 1
        for qry in ppl.fextractor.queries:
            qry.loading_mode = 0

        fvecs = []
        for sid, seed in enumerate(seeds_list):
            qcfgs = ppl.scheduler.get_final_qcfgs(request)
            for qid, qry in enumerate(ppl.fextractor.queries):
                if qry.qtype == XIPQType.AGG:
                    qry.set_enable_qcache()
                    # qry.profiles = qry.profiles[-1:]
                    qry.profiles = []
                    qrng = np.random.default_rng(seed)
                    all_rrd: np.ndarray = rrdatas[qid]
                    if all_rrd is not None:
                        total_n = all_rrd.shape[0]
                        size = int(total_n * qsamples[qid])
                        srrd = all_rrd[qrng.choice(total_n, size=size, replace=False)]
                    else:
                        srrd = None
                    qry._dcache["cached_rrd"] = srrd
                    qry._dcache["cached_nparts"] = qnparts[qid]

                    qcfgs[qid]["qsample"] = qsamples[qid]
            fvec, qcosts = ppl.fextractor.extract(request, qcfgs)
            fvecs.append(fvec)
            # qtimes = np.array([qcost["time"] for qcost in qcosts])
            # print(f"{rid}-{sid}: {qtimes}")
            # print(f"{rid}-{sid}: {fvec['fvals']}")

        fvecs_list.append(fvecs)
        gc.collect()
    # return oracle_fvec_list, fvecs_list
    res = {
        "oracle_fvec_list": oracle_fvec_list,
        "fvecs_list": fvecs_list,
        "moments_list": moments_list,
    }
    joblib.dump(res, res_path)
    print(f"Save results to {res_path}")
    return res


def plot_error_distribution(
    args: OnlineArgs, seeds_list: np.ndarray, res: dict, save_dir: str
):
    print(f"plottig error distribution for {args.nreqs} requests")
    oracle_fvec_list = res["oracle_fvec_list"]
    fvecs_list = res["fvecs_list"]

    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]
    ncols = 4
    nrows = nof // ncols + (nof % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    oracle_fvals_list = np.array([fvec["fvals"] for fvec in oracle_fvec_list])
    for rid in range(nreqs):
        oracle_fvals = oracle_fvals_list[rid]
        fvecs = fvecs_list[rid]
        for fid in range(nof):
            ax = axes[fid]
            fvals = np.array([fvec["fvals"][fid] for fvec in fvecs])
            errors = fvals - oracle_fvals[fid]
            shapiro_test = stats.shapiro(errors)
            pvalue = shapiro_test.pvalue
            sns.histplot(errors, kde=True, ax=ax, label=f"Req {rid} ({pvalue:.3f})")

    for fid in range(nof):
        ax = axes[fid]
        ax.set_title(f"Feature {fid}")
        ax.set_xlabel("Error Value")
        ax.set_ylabel("Density")
        if nreqs <= 10:
            ax.legend()

    plt.tight_layout()
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    fig_path = os.path.join(fig_dir, f"error_distribution_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/error_distribution.png")
    print(f"Save figure to {fig_path}")


def get_pvalues(
    args: OnlineArgs,
    res: dict,
    method: str = "shapiro",
) -> np.ndarray:
    oracle_fvec_list = res["oracle_fvec_list"]
    fvecs_list = res["fvecs_list"]

    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]

    oracle_fvals_list = np.array([fvec["fvals"] for fvec in oracle_fvec_list])
    p_values = np.zeros((nreqs, nof))
    for rid in range(nreqs):
        oracle_fvals = oracle_fvals_list[rid]
        fvecs = fvecs_list[rid]
        for fid in range(nof):
            fvals = np.array([fvec["fvals"][fid] for fvec in fvecs])
            errors = fvals - oracle_fvals[fid]
            if method == "shapiro":
                shapiro_test = stats.shapiro(errors)
            elif method == "normaltest":
                shapiro_test = stats.normaltest(errors)
            elif method == "kstest":
                shapiro_test = stats.kstest(errors, "norm")
            elif method == "jarque_bera":
                shapiro_test = stats.jarque_bera(errors)
            elif method == "kurtosistest":
                shapiro_test = stats.kurtosistest(errors)
            elif method == "skewtest":
                shapiro_test = stats.skewtest(errors)
            else:
                raise ValueError(f"Invalid method: {method}")
            pvalue = shapiro_test.pvalue
            p_values[rid, fid] = pvalue
    print(f"p_values shape: {p_values.shape}")

    return p_values


def plot_normality_test(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    res: dict,
    save_dir: str,
    method: str = "shapiro",
    significance_level: float = 0.05,
):
    print(f"plottig normality test for {args.nreqs} requests")
    fvecs_list = res["fvecs_list"]

    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]
    ncols = 4
    nrows = nof // ncols + (nof % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    pvalues = get_pvalues(args, res, method)
    for fid in range(nof):
        ax = axes[fid]
        percentage = np.sum(pvalues[:, fid] >= significance_level) / nreqs
        sns.histplot(
            pvalues[:, fid],
            bins=20,
            kde=True,
            ax=ax,
            label=f"Feature {fid} ({percentage:.3f})",
        )
        ax.set_title(f"Feature {fid}")
        ax.set_xlabel("P-value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    fig_path = os.path.join(fig_dir, f"normality_test_{method}_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/normality_test.png")
    print(f"Save figure to {fig_path}")

    return pvalues


def plot_nonnormal_cases(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    res: dict,
    save_dir: str,
    method: str = "shapiro",
    significance_level: float = 0.05,
):
    print(f"plottig non-normal cases for {args.nreqs} requests")
    oracle_fvec_list = res["oracle_fvec_list"]
    fvecs_list = res["fvecs_list"]

    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]
    ncols = 4
    nrows = nof // ncols + (nof % ncols > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()
    pvalues = get_pvalues(args, res, method)

    is_normal = pvalues >= significance_level

    oracle_fvals_list = np.array([fvec["fvals"] for fvec in oracle_fvec_list])
    for rid in range(nreqs):
        oracle_fvals = oracle_fvals_list[rid]
        fvecs = fvecs_list[rid]
        for fid in range(nof):
            if not is_normal[rid, fid]:
                ax = axes[fid]
                fvals = np.array([fvec["fvals"][fid] for fvec in fvecs])
                errors = fvals - oracle_fvals[fid]
                shapiro_test = stats.shapiro(errors)
                pvalue = shapiro_test.pvalue
                sns.histplot(errors, kde=True, ax=ax, label=f"Req {rid} ({pvalue:.3f})")

    for fid in range(nof):
        ax = axes[fid]
        num_cases = np.sum(~is_normal[:, fid])
        ax.set_title(f"Feature {fid} ({num_cases})")
        ax.set_xlabel("Error Value")
        ax.set_ylabel("Density")
        print(f"Feature {fid}: {num_cases} non-normal cases")
        if num_cases <= 10:
            ax.legend()

    plt.tight_layout()
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    fig_path = os.path.join(fig_dir, f"non-normal-cases_{method}_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/non-normal-cases.png")
    print(f"Save figure to {fig_path}")


def plot_inference_uncertainty(
    args: OnlineArgs,
    seeds_list: np.ndarray,
    res: dict,
    save_dir: str,
    qmc_pred: bool = False,
):
    print(
        f"plottig Inference Uncertainty with qmc={qmc_pred} for {args.nreqs} requests"
    )
    oracle_fvec_list = res["oracle_fvec_list"]
    fvecs_list = res["fvecs_list"]

    task_home, task_name = args.task.split("/")
    nreqs = args.nreqs
    nof = fvecs_list[0][0]["fvals"].shape[0]
    nseeds = len(seeds_list)

    model = LoadingHelper.load_model(args)
    oracle_preds = [model.predict([fvec["fvals"]])[0] for fvec in oracle_fvec_list]

    res_dir = os.path.join(save_dir, "results", "2.1.1")
    os.makedirs(res_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    qmcpreds_path = os.path.join(res_dir, f"qmcpreds_{tag}.pkl")
    if os.path.exists(qmcpreds_path):
        print(f"Load results from {qmcpreds_path}")
        preds_list = joblib.load(qmcpreds_path)
    else:
        if qmc_pred:
            preds_list = []
            ppl = get_ppl(task_name, args, None, verbose=False)
            for i in tqdm(range(nreqs), desc="QMC Prediction Requests", leave=False):
                for j in tqdm(range(nseeds), desc="QMC Prediction Seeds", leave=False):
                    fvec = fvecs_list[i][j]
                    preds = ppl.pred_estimator.estimate(ppl.model, fvec)
                    preds_list.append(preds["pred_value"])
            preds_list = np.array(preds_list).reshape(nreqs, nseeds)
        else:
            preds_list = np.array(
                [
                    [model.predict([fvec["fvals"]])[0] for fvec in fvecs]
                    for fvecs in fvecs_list
                ]
            )
        joblib.dump(preds_list, qmcpreds_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    errors_list = []
    conf_list = []
    for rid in range(nreqs):
        oracle_pred = oracle_preds[rid]
        preds = preds_list[rid]
        errors = preds - oracle_pred
        errors_list.append(errors)
        conf = np.sum(np.abs(errors) <= args.max_error) / len(errors)
        conf_list.append(conf)

    ax = axes[0]
    for rid in range(nreqs):
        shapiro_test = stats.shapiro(errors_list[rid])
        pvalue = shapiro_test.pvalue
        sns.histplot(
            errors_list[rid], kde=True, ax=ax, label=f"Req {rid} ({pvalue:.3f})"
        )

    ax.set_title("Inference Uncertainty")
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Density")
    if nreqs <= 10:
        ax.legend()

    ax = axes[1]
    sns.histplot(conf_list, kde=True, ax=ax)
    meet_frac = np.sum(np.array(conf_list) >= args.min_conf) / nreqs
    ax.set_title(f"Confidence ({int(meet_frac*100)}% above default conf level)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")

    plt.tight_layout()
    fig_dir = os.path.join(save_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    tag = get_tag(args, seeds_list[-1])
    if qmc_pred:
        tag = f"qmc_{tag}"
    fig_path = os.path.join(fig_dir, f"inference_uncertainty_{tag}.pdf")
    plt.savefig(fig_path)
    plt.savefig("./cache/inference_uncertainty.png")
    print(f"Save figure to {fig_path}")


if __name__ == "__main__":
    args = OnlineArgs().parse_args()
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache"
    logfile: str = "debug.log"
    seeds_list = np.arange(1000)
    res = run(args, seeds_list, save_dir, logfile)

    if args.nreqs <= 20:
        plot_error_distribution(args, seeds_list, res, save_dir)

    method = "shapiro"
    significance_level = 0.05
    plot_normality_test(args, seeds_list, res, save_dir, method, significance_level)
    plot_nonnormal_cases(args, seeds_list, res, save_dir, method, significance_level)

    plot_inference_uncertainty(args, seeds_list, res, save_dir)
    plot_inference_uncertainty(args, seeds_list, res, save_dir, qmc_pred=True)
