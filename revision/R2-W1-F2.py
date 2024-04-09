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


if __name__ == "__main__":
    x = 100
    operator = 'median'
    dsize = 1000_000
    dbseed = 0
    ddist = 'zipf'
    dist_arg = 1.05
    population = simutils.generate_synthetic_data(x, operator, dsize=dsize, seed=dbseed, ddist=ddist, arg=dist_arg)

    # datas = []
    # rng = np.random.default_rng(dbseed)
    # for i in range(10):
    #     loc = rng.uniform(-10, 10)
    #     scale = rng.uniform(1, 5)
    #     mode_samples = np.random.normal(loc=loc, scale=scale, size=dsize // 10)
    #     datas.append(mode_samples)
    # population = np.concatenate(datas)

    sr = 0.01
    sample_size = int(sr * len(population))
    print(f'Population size: {len(population)}, Sample size: {sample_size}')

    median_list = []
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        samples = rng.choice(population, sample_size, replace=False)
        median_list.append(np.median(samples))
    median_list = np.array(median_list)
    error_list = median_list - np.median(population)
    abserror_list = np.abs(error_list)

    print(f'Median of population: {np.median(population)}')
    print(f'Mean of medians: {np.mean(median_list)}')
    print(f'Std of medians: {np.std(median_list)}')

    # test normality of error_list
    _, pval = stats.shapiro(error_list)
    print(f'pval={pval}')

    bs_seed = 0
    rng = np.random.default_rng(bs_seed)
    samples = rng.choice(population, sample_size, replace=False)
    sample_median = np.median(samples)
    resamples = []
    for i in range(1000):
        resamples.append(rng.choice(samples, sample_size, replace=True))
    resamples = np.array(resamples)

    resample_medians = np.median(resamples, axis=1)
    # bias correction
    bias = np.mean(resample_medians - sample_median)
    resample_medians -= bias

    resample_errors = resample_medians - np.median(population)
    resample_abserrors = np.abs(resample_errors)

    # plot distribution of error_list
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()
    ax = axes[0]
    sns.histplot(error_list, kde=True, label='Real Distribution', color='blue', alpha=0.5, bins=50, stat='density', ax=ax)
    sns.histplot(resample_errors, kde=True, label='Bootstrap Distribution', color='red', alpha=0.5, bins=50, stat='density', ax=ax)
    ax.legend()
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Error Values')
    print(f'normality pval of real error     : {stats.shapiro(error_list)[1]}')
    print(f'normality pval of bootstrap error: {stats.shapiro(resample_errors)[1]}')

    ax = axes[1]
    # plot Q-Q plot of error_list and resample_errors
    import statsmodels.api as sm
    sm.qqplot_2samples(error_list, resample_errors, line='45', ax=ax)
    ax.set_title('Q-Q Plot of Error Values')
    ax.set_xlabel('Real Distribution')
    ax.set_ylabel('Bootstrap Distribution')

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(error_list, resample_errors)
    print(f'KS statistic: {ks_statistic}, p-value: {p_value}')

    fig_dir = "/home/ckchang/ApproxInfer/revision/cache/figs"
    save_dir = os.path.join(fig_dir, "2.1.1")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'error_dist.png'))
    plt.savefig('./cache/median_error_dist.png')
