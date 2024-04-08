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


class R3W2F2Args(simutils.SimulationArgs):
    save_dir: str = "/home/ckchang/ApproxInfer/revision/cache"


def plot_multireqs(
    args: R3W2F2Args, rid_list: np.ndarray, p_list: np.ndarray, res_list: List[dict]
):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = args.selected_qid
    imbalance_ratio = (0.5 - p_list) / (0.5 + p_list)
    print(imbalance_ratio)
    x_values = np.arange(len(p_list))

    # nround_list = np.array([len(res['history']) for res in res_list])
    # correct_list = np.array([is_same_float(res['xip_pred']['pred_value'], res['y_exact']) for res in res_list])
    qsamples_list = np.array(
        [[qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]] for res in res_list]
    )
    non_agg_ids = np.setdiff1d(np.arange(nops), agg_ids)
    qsamples_list[:, non_agg_ids] = 0.0
    avg_samples_list = np.array(
        [np.mean(qsamples[agg_ids]) for qsamples in qsamples_list]
    )
    perror_list = np.array(
        [abs(res["xip_pred"]["pred_value"] - res["y_exact"]) for res in res_list]
    )
    ferrors_list = np.array(
        [np.abs(res["xip_pred"]["fvec"]["fvals"] - res["feature"]) for res in res_list]
    )

    avg_samples_list = np.mean(
        avg_samples_list.reshape(len(rid_list), len(p_list)), axis=0
    )
    qsamples_list = np.mean(
        qsamples_list.reshape(len(rid_list), len(p_list), nops), axis=0
    )
    perror_list = np.mean(perror_list.reshape(len(rid_list), len(p_list)), axis=0)
    ferrors_list = np.mean(
        ferrors_list.reshape(len(rid_list), len(p_list), nops), axis=0
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].plot(x_values, avg_samples_list)
    axes[0].set_title("Average Fraction of Samples")
    axes[0].set_ylabel("Average Fraction of Samples")

    for qid in range(nops):
        axes[1].plot(x_values, qsamples_list[:, qid], label=f"q{qid}")
    axes[1].set_title("Fraction of Samples of Each Feature")
    axes[1].set_ylabel("Fraction of Samples of Each Feature")
    axes[1].legend()

    axes[2].plot(x_values, perror_list)
    axes[2].set_title("Prediction Error")
    axes[2].set_ylabel("Absolute Prediction Error")

    for qid in range(nops):
        axes[3].plot(x_values, ferrors_list[:, qid], label=f"f{qid}")
    axes[3].set_title("Error of Each Feature")
    axes[3].set_ylabel("Absolute Feature Error")
    axes[3].legend()

    for i, ax in enumerate(axes):
        ax.set_xlabel("Reverse Imbalance Ratio")
        # ax.set_xscale('log')
        ax.set_xticks(x_values)
        # ax.set_xticklabels([f'{alpha}' for alpha in x_values], rotation=45)
        ax.set_xticklabels([f"{alpha:.4g}" for alpha in imbalance_ratio], rotation=45)
        ax.grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "imbalance_multi")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"imbalance_{tag}.pdf")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "imbalance.png"))
    print(f"Saved imbalance plot to {save_path}")


def plot_multireqs_v2(
    args: R3W2F2Args, rid_list: np.ndarray, p_list: np.ndarray, res_list: List[dict]
):
    task_name = args.task_name
    nops = simutils.task_meta[task_name]["nops"]
    agg_ids = simutils.task_meta[task_name]["agg_ids"]
    selected_qid = args.selected_qid
    imbalance_ratio = (0.5 - p_list) / (0.5 + p_list)
    print(imbalance_ratio)
    x_values = np.arange(len(p_list))

    # nround_list = np.array([len(res['history']) for res in res_list])
    # correct_list = np.array([is_same_float(res['xip_pred']['pred_value'], res['y_exact']) for res in res_list])
    qsamples_list = np.array(
        [[qcfg["qsample"] for qcfg in res["history"][-1]["qcfgs"]] for res in res_list]
    )
    non_agg_ids = np.setdiff1d(np.arange(nops), agg_ids)
    qsamples_list[:, non_agg_ids] = 0.0
    avg_samples_list = np.array(
        [np.mean(qsamples[agg_ids]) for qsamples in qsamples_list]
    )
    perror_list = np.array(
        [abs(res["xip_pred"]["pred_value"] - res["y_exact"]) for res in res_list]
    )
    ferrors_list = np.array(
        [np.abs(res["xip_pred"]["fvec"]["fvals"] - res["feature"]) for res in res_list]
    )

    avg_samples_list = avg_samples_list.reshape(len(rid_list), len(p_list))
    qsamples_list = qsamples_list.reshape(len(rid_list), len(p_list), nops)
    perror_list = perror_list.reshape(len(rid_list), len(p_list))
    ferrors_list = ferrors_list.reshape(len(rid_list), len(p_list), nops)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    nrids = len(rid_list)
    for i, rid in enumerate(nrids):
        axes[0].plot(x_values, avg_samples_list[i, :], label=f"rid={rid}")
        axes[1].plot(x_values, perror_list[i, :], label=f"rid={rid}")

    if nrids <= 10:
        axes[0].legend()
        axes[1].legend()
    axes[0].set_title("Average Fraction of Samples")
    axes[0].set_ylabel("Average Fraction of Samples")
    axes[1].set_title("Prediction Error")
    axes[1].set_ylabel("Absolute Prediction Error")

    for i, ax in enumerate(axes):
        ax.set_xlabel("Reverse Imbalance Ratio")
        # ax.set_xscale('log')
        ax.set_xticks(x_values)
        # ax.set_xticklabels([f'{alpha}' for alpha in x_values], rotation=45)
        ax.set_xticklabels([f"{alpha:.4g}" for alpha in imbalance_ratio], rotation=45)
        ax.grid()

    plt.tight_layout()
    # save the plot
    save_dir = os.path.join(args.save_dir, "imbalance_multi")
    os.makedirs(save_dir, exist_ok=True)
    tag = args.get_tag()
    save_path = os.path.join(save_dir, f"imbalancev2_{tag}.pdf")
    plt.savefig(save_path)
    plt.savefig(os.path.join(save_dir, "imbalancev2.png"))
    print(f"Saved imbalancev2 plot to {save_path}")


if __name__ == "__main__":
    rid_list = np.arange(100)
    sim_args = R3W2F2Args().parse_args()
    if sim_args.task_name == 'machineryralf':
        rid_list = np.arange(338)

    imbalance_ratio = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9999,
            0.99999,
        ]
    )
    p_list = 0.5 * (1.0 - imbalance_ratio) / (1.0 + imbalance_ratio)
    p_list = np.maximum(p_list, 1e-5)
    res_list = []
    for rid in tqdm(rid_list, desc="rid", leave=False):
        for p in tqdm(p_list, desc="p", leave=False):
            sim_args = R3W2F2Args().parse_args()
            sim_args.rid = rid
            sim_args.dist_param = p
            res = simutils.run_default(sim_args)
            res_list.append(res)

    sim_args = R3W2F2Args().parse_args()
    plot_multireqs(sim_args, rid_list, p_list, res_list)
    plot_multireqs_v2(sim_args, rid_list, p_list, res_list)
