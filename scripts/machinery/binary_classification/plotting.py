from pandarallel import pandarallel

from tap import Tap

from typing import List, Dict, Any

# import numpy as np
import pandas as pd
import os
import json

# import time
# from sklearn import metrics
# from sklearn.pipeline import Pipeline
# import clickhouse_connect
# import joblib
# from tqdm import tqdm
import warnings
import copy
import matplotlib.pyplot as plt
import itertools

from online_test import OnlineParser
from online_test import run as run_online_test

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X does not have valid feature names, but (\w+) was fitted with feature names",
)

pandarallel.initialize(progress_bar=False)
# pandarallel.initialize(progress_bar=False, nb_workers=1)

DATA_HOME = "/home/ckchang/ApproxInfer/data"
RESULTS_HOME = "/home/ckchang/ApproxInfer/results"

PLOTTING_HOME = os.path.join(RESULTS_HOME, "plotting")
os.makedirs(PLOTTING_HOME, exist_ok=True)


def prepare_typed_all_evals(
    evals_list: list, keycols: pd.DataFrame, plotting_dir: str, tag: str
):
    all_time_df = pd.concat(
        [evals["time_eval"] for evals in evals_list], axis=0, ignore_index=True
    )
    all_time_df = pd.concat([keycols, all_time_df], axis=1)
    # all_time_df.to_csv(os.path.join(plotting_dir, f"time_{tag}.csv"), index=False)

    all_fvals_df = pd.concat(
        [evals["feat_eval"].iloc[-1:] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_fvals_df = pd.concat([keycols, all_fvals_df], axis=1)
    # all_fvals_df.to_csv(os.path.join(plotting_dir, f"fevals_{tag}.csv"), index=False)

    all_ppl_acc_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-2:-1] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_ppl_acc_evals_df = pd.concat([keycols, all_ppl_acc_evals_df], axis=1)
    # all_ppl_acc_evals_df.to_csv(os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.csv"), index=False)

    all_ppl_sim_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-1:] for evals in evals_list], axis=0, ignore_index=True
    )
    # add suffix _sim to all columns
    all_ppl_sim_evals_df = all_ppl_sim_evals_df.add_suffix("_sim")
    all_ppl_sim_evals_df = pd.concat([keycols, all_ppl_sim_evals_df], axis=1)
    # all_ppl_sim_evals_df.to_csv(os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.csv"), index=False)

    # concat all evals on gby_df and setting_name
    all_evals_df = pd.merge(
        all_time_df,
        all_fvals_df,
        on=keycols.columns.to_list(),
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_acc_evals_df,
        on=keycols.columns.to_list(),
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_sim_evals_df,
        on=keycols.columns.to_list(),
        how="outer",
    )
    all_evals_df.to_csv(os.path.join(plotting_dir, f"all_evals_{tag}.csv"), index=False)

    return all_evals_df


def plotting_func_1(
    df: pd.DataFrame, axes: list, xcol: str, ycols: list, label: str = None
):
    for i, col in enumerate(ycols):
        ax = axes[i]
        color = next(ax._get_lines.prop_cycler)["color"]
        df.plot(x=xcol, y=col, kind="scatter", ax=ax, color=color, legend=True)
        df.plot(
            x=xcol, y=col, kind="line", ax=ax, color=color, label=label, legend=True
        )
        ax.set_title(col)


def plotting_1(settings: list, evals_list: list, setting_name: str, tag: str):
    plotting_dir = os.path.join(PLOTTING_HOME, setting_name)
    os.makedirs(plotting_dir, exist_ok=True)

    keycols = pd.DataFrame(settings, columns=[setting_name])
    all_evals_df = prepare_typed_all_evals(evals_list, keycols, plotting_dir, tag)

    measurements = {
        "time": [
            "total_feature_loading_frac",
            "total_feature_time",
            "total_feature_estimation_time",
            "total_prediction_estimation_time",
            "total_feature_influence_time",
        ],
        "fevals": ["mse", "mae", "r2", "expv", "maxe"],
        "ppl_acc_evals": ["acc", "recall", "precision", "f1", "roc"],
        "ppl_sim_evals": [
            "acc_sim",
            "recall_sim",
            "precision_sim",
            "f1_sim",
            "roc_sim",
        ],
    }
    for measurement, ycols in measurements.items():
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        axes = axes.flatten()
        plotting_func_1(
            all_evals_df,
            axes=axes,
            xcol=setting_name,
            ycols=ycols,
        )
        fig.savefig(os.path.join(plotting_dir, f"{measurement}_{tag}.png"))
        plt.close(fig)


def plotting_gby(
    evals_list: list,
    setting_name: str,
    setting_values: list,
    gby_cols: list,
    gby_values_list: list,
    tag: str,
):
    plotting_dir = os.path.join(
        PLOTTING_HOME, setting_name, "gby_" + "-".join(gby_cols)
    )
    os.makedirs(plotting_dir, exist_ok=True)
    num_groups = len(gby_values_list)

    # create dataframe storing groupby values and setting values
    keycols = pd.DataFrame(
        {
            col: [
                gby_values[cid]
                for gby_values in gby_values_list
                for _ in setting_values
            ]
            for cid, col in enumerate(gby_cols)
        }
    )
    keycols.insert(0, setting_name, setting_values * num_groups)

    all_evals_df = prepare_typed_all_evals(evals_list, keycols, plotting_dir, tag)

    measurements = {
        "time": [
            "total_feature_loading_frac",
            "total_feature_time",
            "total_feature_estimation_time",
            "total_prediction_estimation_time",
            "total_feature_influence_time",
        ],
        "fevals": ["mse", "mae", "r2", "expv", "maxe"],
        "ppl_acc_evals": ["acc", "recall", "precision", "f1", "roc"],
        "ppl_sim_evals": [
            "acc_sim",
            "recall_sim",
            "precision_sim",
            "f1_sim",
            "roc_sim",
        ],
    }
    for measurement, ycols in measurements.items():
        fig, axes = plt.subplots(2, 3, figsize=(30, 15))
        axes = axes.flatten()
        for name, group in all_evals_df.groupby(gby_cols):
            plotting_func_1(
                df=group,
                axes=axes,
                xcol=setting_name,
                ycols=ycols,
                label=f"{name}",
                # label=name
                # if isinstance(name, str)
                # else "-".join([f"{col}={name[i]}" for i, col in enumerate(gby_cols)]),
            )
        fig.savefig(os.path.join(plotting_dir, f"{measurement}_{tag}.png"))
        plt.close(fig)

    # plot acc-efficiency tradeoff for each setting
    # we choose prediction accuracy and prediction simialrity for acc
    # we choose loading_frac, total_feature_time, three estimation times for efficiency
    fig, axes = plt.subplots(4, 5, figsize=(60, 60))
    xcols = ["acc", "acc_sim", "f1", "f1_sim"]
    ycols = [
        "total_feature_loading_frac",
        "total_feature_time",
        "total_feature_estimation_time",
        "total_prediction_estimation_time",
        "total_feature_influence_time",
    ]
    for name, group in all_evals_df.groupby(gby_cols):
        # sort group by setting_name
        group = group.sort_values(by=setting_name)
        # use setting_value as font size
        # map setting_value to font size in [10, 100]
        sorted_setting_vales = sorted(setting_values)
        size = (
            group[setting_name]
            .apply(
                lambda x: 10 + 90 * sorted_setting_vales.index(x) / len(setting_values)
            )
            .values
        )
        for i, xcol in enumerate(xcols):
            for j, ycol in enumerate(ycols):
                ax = axes[i][j]
                # use same color for line and scatter
                color = next(ax._get_lines.prop_cycler)["color"]
                group.plot(
                    x=xcol,
                    y=ycol,
                    kind="line",
                    ax=ax,
                    label=name,
                    color=color,
                    legend=True,
                )
                group.plot(
                    x=xcol,
                    y=ycol,
                    kind="scatter",
                    ax=ax,
                    s=size,
                    color=color,
                    legend=True,
                )
    fig.savefig(os.path.join(plotting_dir, f"acc-efficiency-tradef-off_{tag}.png"))


def plotting_cfgs(
    cfgs: List[Dict[str, Any]], evals_list: list, plotting_name: str, tag: str
):
    plotting_dir = os.path.join(PLOTTING_HOME, plotting_name)
    os.makedirs(plotting_dir, exist_ok=True)
    keycols = pd.DataFrame(cfgs)
    # add col cfg_id
    keycols.insert(0, "cfg_id", range(len(cfgs)))
    print(keycols)
    all_evals_df = prepare_typed_all_evals(evals_list, keycols, plotting_dir, tag)

    measurements = ["total_feature_loading_frac", "mse", "acc", "acc_sim"]
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    for mid, measurement in enumerate(measurements):
        ax = axes[mid]
        # plot bar chart to compare different settings(cfgs)
        # the x axis is the cfg id, y axis is the measurement
        all_evals_df.plot.bar(
            x="cfg_id",
            y=measurement,
            ax=ax,
            legend=False,
            title=measurement,
            rot=0,
            color="C0",
        )
        ax.set_title(measurement)
    # for the last two axes, we plot the trade-off plot for (acc, total_feature_loading_frac) and (acc_sim, total_feature_loading_frac)
    # add label to each point
    markers = ["o", "x", "+", "s", "d", "^", "v", ">", "<", "p", "h", "D"]
    for i, row in all_evals_df.iterrows():
        x1, x2, y = row["acc"], row["acc_sim"], row["total_feature_loading_frac"]
        color = next(ax._get_lines.prop_cycler)["color"]
        axes[-2].scatter(
            x1,
            y,
            alpha=0.7,
            label=f"cfg-{i}",
            color=color,
            s=200,
            marker=markers[i % len(markers)],
        )
        axes[-1].scatter(
            x2,
            y,
            alpha=0.7,
            label=f"cfg-{i}",
            color=color,
            s=200,
            marker=markers[i % len(markers)],
        )
    axes[-2].set_title("acc v.s. total_feature_loading_frac")
    axes[-1].set_title("acc_sim v.s. total_feature_loading_frac")
    axes[-2].legend()
    axes[-1].legend()

    fig.savefig(os.path.join(plotting_dir, f"cfgs_{tag}.png"))
    plt.close(fig)


class Collector:
    def __init__(self, args: OnlineParser) -> None:
        self.args = copy.deepcopy(args)

    def vary_setting(self, setting_name: str, setting_values: list, tag: str = "tmp"):
        print(f"vary_setting with {setting_name}={setting_values}, tag={tag}")
        ret = []
        for value in setting_values:
            new_args = copy.deepcopy(self.args)
            setattr(new_args, setting_name, value)
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(setting_values, ret, setting_name, tag)

    def vary_setting_gby(
        self,
        setting_name="sample_refine_max_niters",
        setting_values=[0, 1, 2, 3, 4, 5],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
        tag: str = "tmp",
    ):
        print(
            f"vary_setting_gby with {setting_name}={setting_values}, gby_cols={gby_cols}, gby_cols_values={gby_cols_values}, tag={tag}"
        )
        gby_values_list = list(itertools.product(*gby_cols_values))
        ret = []
        for gby_values in gby_values_list:
            for value in setting_values:
                new_args = copy.deepcopy(self.args)
                # set gby values
                for gby_col, gby_value in zip(gby_cols, gby_values):
                    setattr(new_args, gby_col, gby_value)
                setattr(new_args, setting_name, value)
                new_args.process_args()
                ests, evals = run_online_test(new_args)
                ret.append(evals)
        plotting_gby(ret, setting_name, setting_values, gby_cols, gby_values_list, tag)

    def vary_cfgs(self, cfgs: List[Dict[str, Any]], tag="tmp"):
        ret = []
        for cfg in cfgs:
            print(f'run with cfg={cfg}')
            new_args = copy.deepcopy(self.args)
            new_args.update_args(cfg)
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_cfgs(cfgs, ret, "cfgs", tag)


def run_no_refinement(args: OnlineParser):
    """
    Let's see if we disable refinement,
    how different init_sample_budget and policy influences the performance
    """
    tag = "norefinement"
    collector = Collector(args)
    collector.args.update_args({"sample_refine_max_niters": 0})
    collector.vary_setting(
        "init_sample_budget", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0], tag
    )
    # influence of init_sample_policy
    collector.vary_setting("init_sample_policy", ["uniform", "fimp"], tag)
    collector.vary_setting_gby(
        setting_name="init_sample_budget",
        setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
        gby_cols=["init_sample_policy"],
        gby_cols_values=[["uniform", "fimp"]],
        tag=tag,
    )


def run_refine_once(args: OnlineParser):
    """
    Let's see if we enable refinement, and refinement only once,
    how different init_sample_budget, init_policy, prediction_estimator, and online_policy influences the performance
    """
    tag = "refine-once"
    collector = Collector(args)
    collector.args.update_args({"sample_refine_max_niters": 1})
    collector.vary_setting(
        "init_sample_budget", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0], tag
    )
    # influence of init_sample_policy, and online policy
    # with default prediction estimator(auto, 1000) and default finif estimator(auto, 16000)
    collector.vary_setting("init_sample_policy", ["uniform", "fimp"], tag)
    collector.vary_setting("sample_allocation_policy", ["uniform", "fimp", "finf"], tag)
    collector.vary_setting_gby(
        setting_name="init_sample_budget",
        setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
        tag=tag,
    )
    # influence of prediction estimator with default sample_plolicy(uniform, uniform)
    collector.vary_setting_gby(
        setting_name="prediction_estimation_nsamples",
        setting_values=[100, 400, 1000, 4000, 10000],
        gby_cols=["prediction_estimator", "prediction_estimator_thresh"],
        gby_cols_values=[
            ["joint_distribution", "independent_distribution", "auto"],
            [1.0, 0.99, 0.9],
        ],
        tag=tag,
    )
    collector.vary_setting_gby(
        setting_name="init_sample_budget",
        setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
        gby_cols=["prediction_estimator", "prediction_estimation_nsamples"],
        gby_cols_values=[
            ["joint_distribution", "independent_distribution", "auto"],
            [100, 1000, 10000],
        ],
        tag=tag,
    )


def run_refine_rounds(args: OnlineParser):
    """
    Let's see the influence of different number of refinement round,
    with different init_sample_policy(uniform) and default sample_allocation_policy(uniform)
    """
    tag = "refine-rounds"
    collector = Collector(args)
    collector.vary_setting(
        "sample_refine_max_niters", [0, 1, 2, 3, 4, 5, 7, 9, 10], tag
    )
    collector.vary_setting_gby(
        setting_name="sample_refine_max_niters",
        setting_values=[0, 1, 2, 3, 4, 5, 7, 9, 10],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
        tag=tag,
    )


def run_refine_rounds_more(args: OnlineParser):
    """
    Let's see the influence of different number of refinement round,
    with different init_sample_policy(uniform) and default sample_allocation_policy(uniform)
    """
    tag = "refine-rounds-more"
    collector = Collector(args)
    collector.args.update_args(
        {"feature_influence_estimation_nsamples": 800, "init_sample_budget": 0.001}
    )
    collector.vary_setting(
        "sample_refine_max_niters", [0, 1, 2, 3, 4, 5, 7, 9, 10, 100], tag
    )
    collector.vary_setting_gby(
        setting_name="sample_refine_max_niters",
        setting_values=[0, 1, 2, 3, 4, 5, 7, 9, 10, 100],
        gby_cols=[
            "init_sample_policy",
            "sample_allocation_policy",
            "sample_refine_step_policy",
            "init_sample_budget",
        ],
        gby_cols_values=[
            ["uniform"],
            ["uniform", "finf"],
            ["uniform"],
            [0.001, 0.01, 0.1],
        ],
        tag=tag,
    )


def run_refine_rounds_iter(args: OnlineParser):
    """
    Let's see if we enable refinement, and refinement 3/5/10 round,
    how different budget and policy influences the performance
    # to avoid to much comparison, we only use default init_sample_policy(uniform)
    """
    for num_round in [3, 5, 10]:
        tag = f"refine-{num_round}"
        collector = Collector(args)
        collector.args.update_args({"sample_refine_max_niters": num_round})
        collector.vary_setting(
            "init_sample_budget", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0], tag
        )
        # influence of init_sample_policy, and online policy
        # with default prediction estimator(auto, 1000) and default finif estimator(auto, 16000)
        collector.vary_setting_gby(
            setting_name="sample_refine_step_policy",
            setting_values=["uniform", "exponential", "exponential_rev"],
            gby_cols=["init_sample_policy", "sample_allocation_policy"],
            gby_cols_values=[
                ["uniform", "fimp"],
                ["uniform", "fimp", "finf"],
            ],
            tag=tag,
        )
        collector.vary_setting_gby(
            setting_name="init_sample_budget",
            setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
            gby_cols=[
                "init_sample_policy",
                "sample_allocation_policy",
                "sample_refine_step_policy",
            ],
            gby_cols_values=[
                ["uniform", "fimp"],
                ["uniform", "fimp", "finf"],
                ["uniform", "exponential", "exponential_rev"],
            ],
            tag=tag,
        )
        # influence of prediction estimator with default sample_plolicy(uniform, uniform)
        collector.vary_setting_gby(
            setting_name="prediction_estimation_nsamples",
            setting_values=[100, 400, 1000, 4000, 10000],
            gby_cols=["prediction_estimator", "prediction_estimator_thresh"],
            gby_cols_values=[
                ["joint_distribution", "independent_distribution", "auto"],
                [1.0, 0.99, 0.9],
            ],
            tag=tag,
        )
        collector.vary_setting_gby(
            setting_name="init_sample_budget",
            setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
            gby_cols=["prediction_estimator", "prediction_estimation_nsamples"],
            gby_cols_values=[
                ["joint_distribution", "independent_distribution", "auto"],
                [100, 1000, 10000],
            ],
            tag=tag,
        )


def load_cfgs(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r") as f:
        cfgs = json.load(f)
    return cfgs


def run_cgfs(args: OnlineParser, cfg_path: str):
    """
    Let's see the comparision of selected cfgs
    """
    tag = f"{os.path.basename(cfg_path)}"
    cfgs = load_cfgs(cfg_path)
    collector = Collector(args)
    collector.vary_cfgs(cfgs, tag)


class PlottingParser(Tap):
    run_no_refinement: bool = False
    run_refine_rounds: bool = False
    run_refine_rounds_more: bool = False
    run_refine_rounds_iter: bool = False

    run_cfgs: str = None  # path to the cfg file

    run_all: bool = False

    nrounds: int = 1  # number of rounds to run

    def process_args(self) -> None:
        if self.run_all:
            self.run_no_refinement = True
            self.run_refine_rounds = True
            self.run_refine_rounds_more = True
            self.run_refine_rounds_iter = True


if __name__ == "__main__":
    default_args_dict = {
        "init_sample_budget": 0.01,
        "init_sample_policy": "uniform",
        "feature_estimator": "closed_form",
        "feature_estimation_nsamples": 1000,
        "prediction_estimator": "auto",
        "prediction_estimator_thresh": 1.0,
        "prediction_estimation_nsamples": 1000,
        "feature_influence_estimator": "auto",
        "feature_influence_estimator_thresh": 1.0,
        "feature_influence_estimation_nsamples": 16000,
        "sample_budget": 1.0,
        "sample_refine_max_niters": 0,
        "sample_refine_step_policy": "uniform",
        "sample_allocation_policy": "uniform",
    }
    default_args = OnlineParser().from_dict(default_args_dict)
    default_args.process_args()
    print(f"running plotting.py with {default_args}")

    plotting_args = PlottingParser().parse_args()
    if plotting_args.run_no_refinement:
        run_no_refinement(default_args)
    if plotting_args.run_refine_rounds:
        run_refine_rounds(default_args)
    if plotting_args.run_refine_rounds_more:
        run_refine_rounds_more(default_args)
    if plotting_args.run_refine_rounds_iter:
        run_refine_rounds_iter(default_args)
    if plotting_args.run_cfgs:
        run_cgfs(default_args, plotting_args.run_cfgs)
