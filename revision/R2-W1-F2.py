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
import statsmodels.api as sm

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


class R2W1F2Args(Tap):
    dsize: int = 10_000_000 + 1
    # dsize: int = 1_000_000 + 1
    # dsize: int = 100_000 + 1
    sfrac: float = 0.05
    nseeds: int = 1000
    bs_seed: int = 0
    nresamples: int = 100

    nocache: bool = False
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache/results/2.1.2"

    def process_args(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def get_tag(self):
        tag = "-".join(
            [
                f"dsize={self.dsize}",
                f"sfrac={self.sfrac}",
                f"nseeds={self.nseeds}",
                f"bs_seed={self.bs_seed}",
                f"nresamples={self.nresamples}",
            ]
        )
        return tag


def run(args: R2W1F2Args) -> dict:
    tag = args.get_tag()
    res_path = os.path.join(args.save_dir, f"res-{tag}.pkl")
    if (not args.nocache) and os.path.exists(res_path):
        print(f"Loading from cache: {res_path}")
        res = joblib.load(res_path)
        return res

    dsize = args.dsize
    sampling_frac = args.sfrac
    nseeds = args.nseeds
    bs_seed = args.bs_seed
    nresamples = args.nresamples

    # smethod = 'binomial'
    smethod = "choice"

    rng = np.random.default_rng(0)
    datas = []
    p1size = int(0.5 * dsize)
    p2size = dsize - p1size
    dstd = dsize // 100_000
    dloc = dstd * 10
    params_list = [(-dloc, dstd, p1size), (dloc, dstd, p2size)]
    print(f"params_list: {params_list}")
    for params in params_list:
        loc, scale, part_size = params
        part_samples = rng.normal(loc=loc, scale=scale, size=part_size)
        datas.append(part_samples)
    population = np.concatenate(datas)
    real_median = np.median(population)

    print(f"Population size      : {len(population)}")
    print(f"Population Median    : {real_median}")

    rng = np.random.default_rng(0)
    if smethod == "binomial":
        samples = population[
            rng.binomial(1, sampling_frac, len(population)).astype(bool)
        ]
    elif smethod == "choice":
        samples = rng.choice(
            population, size=int(sampling_frac * len(population)), replace=False
        )
    sample_size = len(samples)
    sample_median = np.median(samples)
    print(f"Sampling Fraction : {sampling_frac}")
    print(f"Sampling Size     : {sample_size}")
    print(f"Sample Median     : {sample_median}")

    error_list = []
    for seed in tqdm(range(nseeds), desc="Generating Error List"):
        rng = np.random.default_rng(seed)
        if smethod == "binomial":
            tmp = population[
                rng.binomial(1, sampling_frac, len(population)).astype(bool)
            ]
        elif smethod == "choice":
            tmp = rng.choice(
                population, size=int(sampling_frac * len(population)), replace=False
            )
        error_list.append(np.median(tmp) - real_median)
    error_list = np.array(error_list)

    # test normality of error_list
    _, pval = stats.shapiro(error_list)
    print(f"error_list normality pval={pval}")

    resample_medians = []
    rng = np.random.default_rng(bs_seed)
    for i in tqdm(range(nresamples), desc="Bootstrap Resampling"):
        resample = rng.choice(samples, sample_size, replace=True)
        resample_median = np.median(resample)
        resample_medians.append(resample_median)
    resample_medians = np.array(resample_medians)

    # bias correction
    mean_resample_medians = np.mean(resample_medians)
    bias = mean_resample_medians - sample_median
    bs_median_estimation = sample_median - bias

    print(f"bias                 : {bias}")
    print(f"Resample AVG Median  : {mean_resample_medians}")
    print(f"Bootstrap Median     : {bs_median_estimation}")

    # resample_errors = resample_medians - sample_median
    # resample_errors = resample_medians - mean_resample_medians
    # resample_errors = resample_medians - bs_median_estimation
    resample_errors = resample_medians - real_median

    normal_bs_loc = 0
    normal_bs_std = np.std(resample_medians, ddof=1)
    normal_bs_errors = stats.norm.rvs(
        loc=normal_bs_loc, scale=normal_bs_std, size=10000
    )

    res = {
        "real_median": real_median,
        # "population": population if len(population) <= 100_000 else population[np.random.default_rng(0).choice(len(population), 100_000)],
        "population": population,
        "real_error": sample_median - real_median,
        "error_list": error_list,
        "resample_errors": resample_errors,
        "normal_bs_errors": normal_bs_errors,
    }

    joblib.dump(res, res_path)
    return res


def plot_median_error(res: dict):
    print("Plotting Median Error Distribution")
    error_list = res["error_list"]
    resample_errors = res["resample_errors"]

    # plot distribution of error_list
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    ax = axes[0]
    # plot distribution of population
    population = res["population"]
    population = np.random.default_rng(0).choice(population, 100_000)
    sns.histplot(
        population,
        # kde=True,
        label="Data Distribution",
        color="blue",
        alpha=0.5,
        bins=100,
        # stat="density",
        ax=ax,
    )
    ax.axvline(
        res["real_median"],
        color="black",
        linestyle="--",
        label="Median of Data",
    )

    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Data")

    ax = axes[1]
    sns.kdeplot(
        error_list,
        # cumulative=True,
        label="Real Distribution",
        color="blue",
        alpha=0.5,
        ax=ax,
    )
    sns.kdeplot(
        resample_errors,
        # cumulative=True,
        label="Bootstrap Distribution",
        color="red",
        alpha=0.5,
        ax=ax,
    )
    # ax.axvline(
    #     res["real_error"],
    #     color="black",
    #     linestyle="--",
    #     label="Real Error",
    # )

    ax.legend()
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Error")

    print(f"normality pval of real error     : {stats.shapiro(error_list)[1]}")
    print(f"normality pval of bootstrap error: {stats.shapiro(resample_errors)[1]}")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(error_list, resample_errors)
    print(f"KS statistic: {ks_statistic}, p-value: {p_value}")

    fig_dir = "/home/ckchang/ApproxInfer/revision/cache/figs"
    save_dir = os.path.join(fig_dir, "2.1.2")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "median_error_dist.pdf")
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.savefig("./cache/median_error_dist.png")


if __name__ == "__main__":
    args = R2W1F2Args().parse_args()
    res = run(args)
    plot_median_error(res)
