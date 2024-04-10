import os
import sys
import numpy as np
import json
import math
import time
from typing import List
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

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from apxinfer.core.config import OnlineArgs
from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.utils import is_same_float, XIPQType
from apxinfer.core.data import DBHelper
from apxinfer.core.pipeline import XIPPipeline
from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS
from apxinfer.examples.run import get_ppl

from apxinfer.simulation import utils as simutils


class R2W1F3Args(Tap):
    tasks: List[str] = ['machineryralf', 'machineryralfsimmedian0']

    oracle_type: str = 'exact'
    metric: str = 'acc'

    fig_dir: str = '/home/ckchang/ApproxInfer/revision/cache/figs/2.1.3'


def collect_data(args: R2W1F3Args) -> pd.DataFrame:
    data = []
    for task_name in args.tasks:
        sim_args = simutils.SimulationArgs().parse_args()
        sim_args.task_name = task_name
        if task_name == 'machineryralf':
            sim_args.bs_type = 'fstd'
        ol_args = simutils.get_online_args(sim_args)

        online_dir = DIRHelper.get_online_dir(ol_args)
        evals_tag = DIRHelper.get_eval_tag(ol_args)
        evals_path = os.path.join(online_dir, f"evals_{evals_tag}.json")
        with open(evals_path, 'r') as f:
            evals = json.load(f)
        latency = evals['avg_ppl_time']
        if args.oracle_type == 'exact':
            accuracy = evals['evals_to_ext'][args.metric]
        else:
            accuracy = evals['evals_to_gt'][args.metric]
        data.append({
            'task_name': task_name,
            'latency': latency,
            'accuracy': accuracy
        })
    df = pd.DataFrame(data)
    return df


def plot_data(args: R2W1F3Args, df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0] # compare latency of each task
    sns.barplot(x='task_name', y='latency', data=df, ax=ax)
    ax.set_title('Latency Comparison')
    ax.set_ylabel('Latency (s)')
    ax.set_xlabel('Task Name')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax = axes[1] # compare accuracy of each task
    sns.barplot(x='task_name', y='accuracy', data=df, ax=ax)
    ax.set_title('Accuracy Comparison')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Task Name')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    fig_dir = args.fig_dir
    os.makedirs(fig_dir, exist_ok=True)
    tag = "_".join(args.tasks + [
        args.oracle_type,
        args.metric
    ])
    fig_path = os.path.join(fig_dir, f'avg_vs_median_{tag}.pdf')
    plt.savefig(fig_path)
    plt.savefig("./cache/avg_vs_median.pdf")
    print(f'Save figure to {fig_path}')


if __name__ == '__main__':
    args = R2W1F3Args().parse_args()
    df = collect_data(args)
    plot_data(args, df)