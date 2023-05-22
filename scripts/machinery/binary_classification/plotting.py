from pandarallel import pandarallel

# from tap import Tap
# from typing import Literal, Tuple
# import numpy as np
import pandas as pd
import os

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
RESULTS_HOME = "/home/ckchang/ApproxInfer/results2"

PLOTTING_HOME = os.path.join(RESULTS_HOME, "plotting")
os.makedirs(PLOTTING_HOME, exist_ok=True)


def plotting_func_1(df, axes, xcol, ycols):
    for i, col in enumerate(ycols):
        ax = axes[i]
        color = next(ax._get_lines.prop_cycler)["color"]
        df.plot(x=xcol, y=col, kind="scatter", ax=ax, color=color, legend=True)
        df.plot(x=xcol, y=col, kind="line", ax=ax, color=color, legend=True)
        ax.set_title(col)


def prepare_typed_all_evals(
    evals_list: list, keycols: pd.DataFrame, plotting_dir: str, tag: str
):
    all_time_df = pd.concat(
        [evals["time_eval"] for evals in evals_list], axis=0, ignore_index=True
    )
    all_time_df = pd.concat([keycols, all_time_df], axis=1)
    all_time_df.to_csv(os.path.join(plotting_dir, f"time_{tag}.csv"), index=False)

    all_fvals_df = pd.concat(
        [evals["feat_eval"].iloc[-1:] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_fvals_df = pd.concat([keycols, all_fvals_df], axis=1)
    all_fvals_df.to_csv(os.path.join(plotting_dir, f"fevals_{tag}.csv"), index=False)

    all_ppl_acc_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-2:-1] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_ppl_acc_evals_df = pd.concat([keycols, all_ppl_acc_evals_df], axis=1)
    all_ppl_acc_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.csv"), index=False
    )

    all_ppl_sim_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-1:] for evals in evals_list], axis=0, ignore_index=True
    )
    # add suffix _sim to all columns
    all_ppl_sim_evals_df = all_ppl_sim_evals_df.add_suffix("_sim")
    all_ppl_sim_evals_df = pd.concat([keycols, all_ppl_sim_evals_df], axis=1)
    all_ppl_sim_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.csv"), index=False
    )

    # concat all evals on gby_df and setting_name
    all_evals_df = pd.merge(
        all_time_df,
        all_fvals_df,
        on=keycols.columns,
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_acc_evals_df,
        on=keycols.columns,
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_sim_evals_df,
        on=keycols.columns,
        how="outer",
    )
    all_evals_df.to_csv(os.path.join(plotting_dir, f"all_evals_{tag}.csv"), index=False)

    return all_evals_df


def plotting_1(settings: list, evals_list: list, setting_name: str, tag: str = "tmp"):
    plotting_dir = os.path.join(PLOTTING_HOME, setting_name)
    os.makedirs(plotting_dir, exist_ok=True)

    all_time_df = pd.concat([evals["time_eval"] for evals in evals_list], axis=0)
    # add settings to the first column
    all_time_df.insert(0, setting_name, settings)
    all_time_df.to_csv(os.path.join(plotting_dir, f"time_{tag}.csv"), index=False)
    print(all_time_df)

    # all columns in all_time_df now:
    # setting_name,exact_pred_time,total_feature_loading_nrows,total_feature_loading_frac,total_feature_loading_time,total_feature_time,total_feature_estimation_time,total_prediction_estimation_time,total_feature_influence_time

    # plot setting_name-total_feature_loading_frac, setting_name-total_feature_time, setting_name-total_feature_estimation_time, setting_name-total_prediction_estimation_time, setting_name-total_feature_influence_time
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    plotting_func_1(
        all_time_df,
        axes=axes,
        xcol=setting_name,
        ycols=[
            "total_feature_loading_frac",
            "total_feature_time",
            "total_feature_estimation_time",
            "total_prediction_estimation_time",
            "total_feature_influence_time",
        ],
    )
    fig.savefig(os.path.join(plotting_dir, f"time_{tag}.png"))

    all_fevals_df = pd.concat(
        [evals["feat_eval"].iloc[-1:] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_fevals_df.insert(0, setting_name, settings)
    all_fevals_df.to_csv(os.path.join(plotting_dir, f"fevals_{tag}.csv"), index=False)
    print(all_fevals_df)

    # all columns in all_fevals_df now:
    # setting_name,tag,mse,mae,r2,expv,maxe

    # plot setting_name-mse, setting_name-mae, setting_name-r2, setting_name-expv, setting_name-maxe
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    plotting_func_1(
        all_fevals_df,
        axes=axes,
        xcol=setting_name,
        ycols=["mse", "mae", "r2", "expv", "maxe"],
    )
    fig.savefig(os.path.join(plotting_dir, f"fevals_{tag}.png"))

    all_ppl_sim_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-1:] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_ppl_sim_evals_df.insert(0, setting_name, settings)
    all_ppl_sim_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.csv"), index=False
    )
    print(all_ppl_sim_evals_df)

    # all columns in all_ppl_sim_evals_df now:
    # setting_name,tag,acc,recall,precision,f1,roc,recall_micro,precision_micro,f1_micro,roc_micro,recall_weighted,precision_weighted,f1_weighted,roc_weighted

    # plot setting_name-acc, setting_name-recall, setting_name-precision, setting_name-f1, setting_name-roc
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    plotting_func_1(
        all_ppl_sim_evals_df,
        axes=axes,
        xcol=setting_name,
        ycols=["acc", "recall", "precision", "f1", "roc"],
    )
    fig.savefig(os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.png"))

    all_ppl_acc_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-2:-1] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_ppl_acc_evals_df.insert(0, setting_name, settings)
    all_ppl_acc_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.csv"), index=False
    )

    # all columns in all_ppl_acc_evals_df now:
    # setting_name,tag,acc,recall,precision,f1,roc,recall_micro,precision_micro,f1_micro,roc_micro,recall_weighted,precision_weighted,f1_weighted,roc_weighted

    # plot setting_name-acc, setting_name-recall, setting_name-precision, setting_name-f1, setting_name-roc
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    plotting_func_1(
        all_ppl_acc_evals_df,
        axes=axes,
        xcol=setting_name,
        ycols=["acc", "recall", "precision", "f1", "roc"],
    )
    fig.savefig(os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.png"))


def plotting_gby(
    evals_list: list,
    setting_name: str,
    setting_values: list,
    gby_cols: list,
    gby_values_list: list,
    tag: str = "tmp",
):
    plotting_dir = os.path.join(
        PLOTTING_HOME, setting_name, "gby_" + "-".join(gby_cols)
    )
    os.makedirs(plotting_dir, exist_ok=True)
    num_groups = len(gby_values_list)
    # create dataframe storing groupby values and setting values
    gby_df = pd.DataFrame(
        {
            col: [
                gby_values[cid]
                for gby_values in gby_values_list
                for _ in setting_values
            ]
            for cid, col in enumerate(gby_cols)
        }
    )
    gby_df.insert(0, setting_name, setting_values * num_groups)

    all_time_df = pd.concat(
        [evals["time_eval"] for evals in evals_list], axis=0, ignore_index=True
    )
    all_time_df = pd.concat([gby_df, all_time_df], axis=1)
    all_time_df.to_csv(os.path.join(plotting_dir, f"time_{tag}.csv"), index=False)

    all_fvals_df = pd.concat(
        [evals["feat_eval"].iloc[-1:] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_fvals_df = pd.concat([gby_df, all_fvals_df], axis=1)
    all_fvals_df.to_csv(os.path.join(plotting_dir, f"fevals_{tag}.csv"), index=False)

    all_ppl_acc_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-2:-1] for evals in evals_list],
        axis=0,
        ignore_index=True,
    )
    all_ppl_acc_evals_df = pd.concat([gby_df, all_ppl_acc_evals_df], axis=1)
    all_ppl_acc_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.csv"), index=False
    )

    all_ppl_sim_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-1:] for evals in evals_list], axis=0, ignore_index=True
    )
    # add suffix _sim to all columns
    all_ppl_sim_evals_df = all_ppl_sim_evals_df.add_suffix("_sim")
    all_ppl_sim_evals_df = pd.concat([gby_df, all_ppl_sim_evals_df], axis=1)
    all_ppl_sim_evals_df.to_csv(
        os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.csv"), index=False
    )

    # concat all evals on gby_df and setting_name
    all_evals_df = pd.merge(
        all_time_df,
        all_fvals_df,
        on=gby_cols + [setting_name],
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_acc_evals_df,
        on=gby_cols + [setting_name],
        how="outer",
    )
    all_evals_df = pd.merge(
        all_evals_df,
        all_ppl_sim_evals_df,
        on=gby_cols + [setting_name],
        how="outer",
    )
    all_evals_df.to_csv(os.path.join(plotting_dir, f"all_evals_{tag}.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    ycols = [
        "total_feature_loading_frac",
        "total_feature_time",
        "total_feature_estimation_time",
        "total_prediction_estimation_time",
        "total_feature_influence_time",
    ]
    gby_time_df = all_time_df.groupby(gby_cols)
    for i, col in enumerate(ycols):
        ax = axes[i]
        for name, group in gby_time_df:
            color = next(ax._get_lines.prop_cycler)["color"]
            group.plot(
                x=setting_name, y=col, kind="scatter", ax=ax, color=color, legend=True
            )
            group.plot(
                x=setting_name,
                y=col,
                kind="line",
                ax=ax,
                label=name,
                color=color,
                legend=True,
            )
    fig.savefig(os.path.join(plotting_dir, f"time_{tag}.png"))

    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    ycols = ["mse", "mae", "r2", "expv", "maxe"]
    gby_time_df = all_fvals_df.groupby(gby_cols)
    for i, col in enumerate(ycols):
        ax = axes[i]
        for name, group in gby_time_df:
            color = next(ax._get_lines.prop_cycler)["color"]
            group.plot(
                x=setting_name, y=col, kind="scatter", ax=ax, color=color, legend=True
            )
            group.plot(
                x=setting_name,
                y=col,
                kind="line",
                ax=ax,
                label=name,
                color=color,
                legend=True,
            )
    fig.savefig(os.path.join(plotting_dir, "fvals_{tag}.png"))

    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    ycols = ["acc", "recall", "precision", "f1", "roc"]
    gby_time_df = all_ppl_acc_evals_df.groupby(gby_cols)
    for i, col in enumerate(ycols):
        ax = axes[i]
        for name, group in gby_time_df:
            color = next(ax._get_lines.prop_cycler)["color"]
            group.plot(
                x=setting_name, y=col, kind="scatter", ax=ax, color=color, legend=True
            )
            group.plot(
                x=setting_name,
                y=col,
                kind="line",
                ax=ax,
                label=name,
                color=color,
                legend=True,
            )
    fig.savefig(os.path.join(plotting_dir, f"ppl_acc_evals_{tag}.png"))

    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    ycols = ["acc_sim", "recall_sim", "precision_sim", "f1_sim", "roc_sim"]
    gby_time_df = all_ppl_sim_evals_df.groupby(gby_cols)
    for i, col in enumerate(ycols):
        ax = axes[i]
        for name, group in gby_time_df:
            color = next(ax._get_lines.prop_cycler)["color"]
            group.plot(
                x=setting_name, y=col, kind="scatter", ax=ax, color=color, legend=True
            )
            group.plot(
                x=setting_name,
                y=col,
                kind="line",
                ax=ax,
                label=name,
                color=color,
                legend=True,
            )
    fig.savefig(os.path.join(plotting_dir, f"ppl_sim_evals_{tag}.png"))

    # plot acc-efficiency tradeoff for each setting
    # we choose prediction accuracy and prediction simialrity for acc
    # we choose loading_frac, total_feature_time, three estimation times for efficiency
    fig, axes = plt.subplots(2, 5, figsize=(50, 30))
    xcols = ["acc", "acc_sim"]
    ycols = [
        "total_feature_loading_frac",
        "total_feature_time",
        "total_feature_estimation_time",
        "total_prediction_estimation_time",
        "total_feature_influence_time",
    ]
    gby_time_df = all_evals_df.groupby(gby_cols)
    for i, xcol in enumerate(xcols):
        for j, ycol in enumerate(ycols):
            ax = axes[i][j]
            for name, group in gby_time_df:
                # sort group by setting_name
                group = group.sort_values(by=setting_name)
                # use setting_value as font size
                size = group[setting_name] * 100
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


class Collector:
    def __init__(self, args: OnlineParser) -> None:
        self.args = copy.deepcopy(args)

    def vary_init_budget(self, init_budgets=[0.01, 0.05, 0.1, 0.5, 1.0]):
        ret = []
        for init_budget in init_budgets:
            new_args = copy.deepcopy(self.args)
            new_args.init_sample_budget = init_budget
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(init_budgets, ret, "init_budget")
        return ret

    def vary_sample_budget(self, sample_budgets=[0.01, 0.05, 0.1, 0.5, 1.0]):
        ret = []
        for sample_budget in sample_budgets:
            new_args = copy.deepcopy(self.args)
            new_args.sample_budget = sample_budget
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(sample_budgets, ret, "sample_budget")
        return ret

    def vary_sample_refine_max_niters(
        self, sample_refine_max_niters=[0, 1, 2, 3, 4, 5]
    ):
        ret = []
        for sample_refine_max_niter in sample_refine_max_niters:
            new_args = copy.deepcopy(self.args)
            new_args.sample_refine_max_niters = sample_refine_max_niter
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(sample_refine_max_niters, ret, "sample_refine_max_niters")
        return ret

    def vary_feature_estimation_nsamples(
        self, feature_estimation_nsamples=[100, 1000, 10000]
    ):
        assert self.args.feature_estimator != "closed_form"
        ret = []
        for nsamples in feature_estimation_nsamples:
            new_args = copy.deepcopy(self.args)
            new_args.feature_estimation_nsamples = nsamples
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(feature_estimation_nsamples, ret, "feature_estimation_nsamples")
        return ret

    def vary_prediction_estimation_nsamples(
        self, prediction_estimation_nsamples=[100, 1000, 10000]
    ):
        ret = []
        for nsamples in prediction_estimation_nsamples:
            new_args = copy.deepcopy(self.args)
            new_args.prediction_estimation_nsamples = nsamples
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(
            prediction_estimation_nsamples, ret, "prediction_estimation_nsamples"
        )
        return ret

    def vary_feature_influence_estimation_nsamples(
        self, feature_influence_estimation_nsamples=[800, 8000, 80000]
    ):
        ret = []
        for nsamples in feature_influence_estimation_nsamples:
            new_args = copy.deepcopy(self.args)
            new_args.feature_influence_estimation_nsamples = nsamples
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(
            feature_influence_estimation_nsamples,
            ret,
            "feature_influence_estimation_nsamples",
        )
        return ret

    def vary_init_sample_policy(self, init_sample_policy=["uniform", "fimp"]):
        ret = []
        for policy in init_sample_policy:
            new_args = copy.deepcopy(self.args)
            new_args.init_sample_policy = policy
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(init_sample_policy, ret, "init_sample_policy")

    def vary_setting(self, setting_name: str, setting_values: list):
        ret = []
        for value in setting_values:
            new_args = copy.deepcopy(self.args)
            setattr(new_args, setting_name, value)
            new_args.process_args()
            ests, evals = run_online_test(new_args)
            ret.append(evals)
        plotting_1(setting_values, ret, setting_name)

    def vary_setting_gby(
        self,
        setting_name="sample_refine_max_niters",
        setting_values=[0, 1, 2, 3, 4, 5],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
    ):
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
        plotting_gby(ret, setting_name, setting_values, gby_cols, gby_values_list)


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    print(f"running plotting.py with {args}")
    collector = Collector(args)

    collector.vary_setting(
        "init_sample_budget", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0]
    )
    collector.vary_setting("init_sample_policy", ["uniform", "fimp"])

    collector.vary_setting("feature_estimator", ["closed_form", "bootstrap"])
    # collector.vary_setting("feature_estimation_nsamples", [100, 1000, 10000])

    collector.vary_setting(
        "prediction_estimator",
        ["joint_distribution", "independent_distribution", "auto"],
    )
    collector.vary_setting("prediction_estimation_nsamples", [100, 1000, 10000])

    # collector.vary_setting("feature_influence_estimator", ["shap", "lime", "auto"])
    collector.vary_setting("feature_influence_estimation_nsamples", [800, 8000, 80000])

    collector.vary_setting(
        "sample_budget", [0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0]
    )
    collector.vary_setting("sample_refine_max_niters", [0, 1, 2, 3, 4, 5, 7, 9, 10])
    collector.vary_setting(
        "sample_refine_step_policy", ["uniform", "exponential", "exponential_rev"]
    )
    collector.vary_setting("sample_allocation_policy", ["uniform", "fimp", "finf"])

    collector.vary_setting_gby(
        setting_name="sample_refine_max_niters",
        setting_values=[0, 1, 2, 3, 4, 5, 7, 9, 10],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
    )

    # set sample_refine_max_niters as 5
    collector.args.sample_refine_max_niters = 5
    collector.args.process_args()
    collector.vary_setting_gby(
        setting_name="init_sample_budget",
        setting_values=[0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 1.0],
        gby_cols=["init_sample_policy", "sample_allocation_policy"],
        gby_cols_values=[["uniform", "fimp"], ["uniform", "fimp", "finf"]],
    )
