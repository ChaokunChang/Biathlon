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

    sr = 0.001
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

    # plot distribution of error_list
    plt.figure(figsize=(10, 5))
    sns.histplot(error_list, kde=True)
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Error Values (pval={pval})')
    plt.savefig('./median_error_dist.png')
    # plt.show()