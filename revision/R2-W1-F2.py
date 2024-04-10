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


def run(dsize: int = 1000_000,
        sampling_frac: float = 0.01,
        nseeds: int = 1000,
        nresamples: int = 100) -> dict:
    population = simutils.generate_synthetic_data(
        100, "median", dsize=dsize, seed=0, ddist="zipf", arg=1.05
    )

    # dsize = 2000_000 + 1
    # datas = []
    # rng = np.random.default_rng(dbseed)
    # nmodals = 2
    # for i in range(nmodals):
    #     loc = rng.uniform(-10000, 10000)
    #     scale = rng.uniform(1, 10000)
    #     part_samples = rng.normal(loc=loc, scale=scale, size=dsize // nmodals)
    #     # a = rng.uniform(0.5, 2)
    #     # b = rng.uniform(3, 10)
    #     # part_samples = rng.beta(a, b, size=dsize // nmodals)
    #     datas.append(part_samples)
    # population = np.concatenate(datas)

    sample_size = int(sampling_frac * len(population))
    print(f"Population size: {len(population)}, Sample size: {sample_size}")

    median_list = []
    for seed in range(nseeds):
        rng = np.random.default_rng(seed)
        samples = rng.choice(population, sample_size, replace=False)
        median_list.append(np.median(samples))
    median_list = np.array(median_list)
    error_list = median_list - np.median(population)
    abserror_list = np.abs(error_list)

    print(f"Median of population: {np.median(population)}")
    print(f"Mean of medians: {np.mean(median_list)}")
    print(f"Std of medians: {np.std(median_list)}")

    # test normality of error_list
    _, pval = stats.shapiro(error_list)
    print(f"error_list normality pval={pval}")

    rng = np.random.default_rng(0)
    samples = rng.choice(population, sample_size, replace=False)
    sample_median = np.median(samples)

    resamples = []
    bs_seed = 0
    rng = np.random.default_rng(bs_seed)
    for i in range(nresamples):
        resamples.append(rng.choice(samples, sample_size, replace=True))
    resamples = np.array(resamples)

    resample_medians = np.median(resamples, axis=1)
    resample_errors = resample_medians - sample_median
    resample_abserrors = np.abs(resample_errors)

    # bias correction
    mean_resample_medians = np.mean(resample_medians)
    bias = mean_resample_medians - sample_median
    bs_median_estimation = sample_median + bias

    print(f'bias                 : {bias}')
    print(f"Median of sample     : {sample_median}")
    print(f"bs_median_estimation : {bs_median_estimation}")

    normal_bs_loc = 0
    normal_bs_std = np.std(resample_medians, ddof=1)
    normal_bs_errors = stats.norm.rvs(loc=normal_bs_loc, scale=normal_bs_std, size=10000)

    # rng = np.random.default_rng(bs_seed)
    # tmp_data = (samples, )
    # scipy_bs_result = stats.bootstrap(tmp_data, np.median, n_resamples=nresamples, random_state=rng)
    # scipy_bs_dist = scipy_bs_result.bootstrap_distribution
    # scipy_bs_errors = scipy_bs_dist - sample_median

    return {
        "population": population,
        "error_list": error_list,
        "resample_errors": resample_errors,
        "normal_bs_errors": normal_bs_errors,
        # "scipy_bs_errors": scipy_bs_errors,
    }


def plot_median_error(res: dict):
    population = res["population"]
    error_list = res["error_list"]
    resample_errors = res["resample_errors"]
    normal_bs_errors = res["normal_bs_errors"]
    # scipy_bs_errors = res["scipy_bs_errors"]

    # plot distribution of error_list
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    ax = axes[0]
    # plot distribution of population
    sns.histplot(
        population,
        kde=True,
        label="Population Distribution",
        color="blue",
        alpha=0.5,
        bins=50,
        stat="density",
        ax=ax,
    )
    ax.axvline(np.median(population), color="black", linestyle="--", label="Population Median")
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Population")

    ax = axes[1]
    nbins = 20
    sns.histplot(
        error_list,
        kde=True,
        label="Real Distribution",
        color="blue",
        alpha=0.5,
        bins=nbins,
        stat="density",
        ax=ax,
    )
    sns.histplot(
        resample_errors,
        kde=True,
        label="Bootstrap Distribution",
        color="red",
        alpha=0.5,
        bins=nbins,
        stat="density",
        ax=ax,
    )
    sns.histplot(
        normal_bs_errors,
        kde=True,
        label="Normal Distribution",
        color="green",
        alpha=0.5,
        bins=nbins,
        stat="density",
        ax=ax,
    )
    # sns.histplot(
    #     scipy_bs_errors,
    #     kde=True,
    #     label="Scipy Bootstrap Distribution",
    #     color="purple",
    #     alpha=0.5,
    #     bins=nbins,
    #     stat="density",
    #     ax=ax,
    # )
    ax.axvline(0, color="black", linestyle="--", label="Zero Error")

    ax.legend()
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Error Values")
    print(f"normality pval of real error     : {stats.shapiro(error_list)[1]}")
    print(f"normality pval of bootstrap error: {stats.shapiro(resample_errors)[1]}")
    # print(f"normality pval of scipy bs  error: {stats.shapiro(scipy_bs_errors)[1]}")

    ax = axes[2]
    # plot Q-Q plot of error_list and resample_errors
    sm.qqplot_2samples(error_list, resample_errors, line="45", ax=ax)
    ax.set_title("Q-Q Plot of Error Values")
    ax.set_xlabel("Real Distribution")
    ax.set_ylabel("Bootstrap Distribution")

    ax = axes[3]
    # plot Q-Q plot of error_list and normal_bs_errors
    sm.qqplot_2samples(error_list, normal_bs_errors, line='45', ax=ax)
    ax.set_title("Q-Q Plot of Error Values")
    ax.set_xlabel("Real Distribution")
    ax.set_ylabel("Normal Bootstrap Distribution")

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(error_list, resample_errors)
    print(f"KS statistic: {ks_statistic}, p-value: {p_value}")
    ks_statistic, p_value = stats.ks_2samp(error_list, normal_bs_errors)
    print(f"KS statistic: {ks_statistic}, p-value: {p_value}")
    # ks_statistic, p_value = stats.ks_2samp(error_list, scipy_bs_errors)
    # print(f"KS statistic: {ks_statistic}, p-value: {p_value}")

    fig_dir = "/home/ckchang/ApproxInfer/revision/cache/figs"
    save_dir = os.path.join(fig_dir, "2.1.2")
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "median_error_dist.pdf")
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.savefig("./cache/median_error_dist.png")


def plot_median_error_simple(res: dict):
    population = res["population"]
    error_list = res["error_list"]
    resample_errors = res["resample_errors"]

    # plot distribution of error_list
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    ax = axes[0]
    # plot distribution of population
    sns.histplot(
        population,
        kde=True,
        label="Data Distribution",
        color="blue",
        alpha=0.5,
        bins=100,
        stat="density",
        ax=ax,
    )
    ax.axvline(np.median(population), color="black", linestyle="--", label="True (Median) Feature")
    ax.legend()
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Data")

    ax = axes[1]
    nbins = 50
    # sns.histplot(
    #     error_list,
    #     kde=True,
    #     label="Real Distribution",
    #     color="blue",
    #     alpha=0.5,
    #     bins=nbins,
    #     stat="density",
    #     ax=ax,
    # )
    # sns.histplot(
    #     resample_errors,
    #     kde=True,
    #     label="Bootstrap Distribution",
    #     color="red",
    #     alpha=0.5,
    #     bins=nbins,
    #     stat="density",
    #     ax=ax,
    # )
    sns.kdeplot(
        error_list,
        label="Real Distribution",
        color="blue",
        alpha=0.5,
        ax=ax,
    )
    sns.kdeplot(
        resample_errors,
        label="Bootstrap Distribution",
        color="red",
        alpha=0.5,
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", label="Zero")

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
    res = run()
    # plot_median_error(res)
    plot_median_error_simple(res)