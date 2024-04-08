import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import os
import numpy as np
import math
import json
from matplotlib.transforms import Bbox
from typing import List, Tuple
from tap import Tap
import logging

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS


PJNAME = "Biathlon"
# YLIM_ACC = [0.9, 1.01]
YLIM_ACC = [0.0, 1.01]


REG_TASKS = [
    # "tripsralf2h",
    "tripsralfv2",
    # "tickralf",
    "tickralfv2",
    "batteryv2",
    "turbofan",
]

CLS_TASKS = [
    # "tdfraudralf",
    "tdfraudralf2d",
    "machineryralf",
    # "studentqno18",
    "studentqnov2subset",
    # "studentqnov2",
]

rename_map = {
    "batteryv2": "Battery-Charge",
    "trubofan": "Turbofan-RUL",
    "machineryralf": "Bearing-Imbalance",
    "studentqno18": "QA-Correctness",
}

TASKS = REG_TASKS + CLS_TASKS
PIPELINE_NAME = [rename_map.get(task, task) for task in TASKS]

shared_default_settings = {
    "policy": "optimizer",
    "qinf": "biathlon",
    "pest_constraint": "error",
    "pest_seed": 0,
    "ncores": 1,
    "nparts": 100,
    "ncfgs": 100,
    "pest_nsamples": 128,
    "loading_mode": 0,
    "min_conf": 0.95,
    "alpha": 0.05,
    "beta": 0.01,
}
task_default_settings = {
    "tripsralf2h": {
        "model": "lgbm",
        "max_error": 4.0,
        "ralf_budget": 0.1,
    },
    "tripsralfv2": {
        "model": "lgbm",
        "max_error": 1.5,
        "ralf_budget": 1.0,
    },
    "tickralf": {
        "model": "lr",
        "max_error": 0.04,
        "ralf_budget": 0.1,
    },
    "tickralfv2": {
        "model": "lr",
        "max_error": 0.04,
        "ralf_budget": 5.0,
    },
    "batteryv2": {
        "model": "lgbm",
        "max_error": 189.0,
        "ralf_budget": 0.0,
    },
    "turbofan": {
        "model": "rf",
        "max_error": 4.88,
        "ralf_budget": 0.0,
    },
    "tdfraudralf": {
        "model": "xgb",
        "max_error": 0.0,
        "ralf_budget": 0.1,
    },
    "tdfraudralf2d": {
        "model": "xgb",
        "max_error": 0.0,
        "ralf_budget": 0.05,
    },
    "machineryralf": {
        "model": "mlp",
        "max_error": 0.0,
        "ralf_budget": 0.0,
    },
    "studentqno18": {
        "model": "rf",
        "max_error": 0.0,
        "ralf_budget": 0.0,
    },
    "studentqnov2subset": {
        "model": "rf",
        "max_error": 0.0,
        "ralf_budget": 0.0,
    },
    "studentqnov2": {
        "model": "rf",
        "max_error": 0.0,
        "ralf_budget": 0.0,
    },
}


class EvalArgs(Tap):
    home_dir: str = "./cache"
    plot_dir: str = None
    filename: str = None
    # loading_mode: int = 0
    # ncores: int = 1
    only: str = None
    score_type: str = "similarity"
    cls_score: str = "acc"
    reg_score: str = "meet_rate" # r2
    debug: bool = False

    def process_args(self):
        if self.filename is None:
            # filename should be like evals-YYYYMMDDHH
            # get the filename with latest timestamp
            files = os.listdir(self.home_dir)
            files = [f for f in files if f.startswith("avg_") and f.endswith(".csv")]
            files.sort(reverse=True)
            self.filename = files[0]
            print(f"Using {self.filename}")
        if self.plot_dir is None:
            self.plot_dir = f'figs_{self.score_type}_{self.cls_score}_{self.reg_score}'
            print(f"Using {self.plot_dir}")


def load_df(args: EvalArgs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(args.home_dir, args.filename))

    def handler_for_other_cost(row):
        if row["loading_mode"] in [1000, 2000]:
            row["BD:Others"] = 1e-6 * row["avg_nrounds"]
            row["BD:AFC"] = (
                row["avg_latency"] - row["BD:AMI"] - row["BD:Sobol"] - row["BD:Others"]
            )
        else:
            row["BD:Others"] = (
                row["avg_latency"] - row["BD:AFC"] - row["BD:AMI"] - row["BD:Sobol"]
            )
            # row['BD:Others'] = 1e-5 * row["avg_nrounds"]
            # row['BD:AFC'] = row['avg_latency'] - row['BD:AMI'] - row['BD:Sobol'] - row['BD:Others']
        return row

    df = df.apply(handler_for_other_cost, axis=1)
    df["alpha"] = df["scheduler_init"] / df["ncfgs"]
    df["beta"] = df["scheduler_batch"] / df["ncfgs"]
    df["beta"] /= df["naggs"]

    for task_name in TASKS:
        if task_name in REG_TASKS:
            reg_score = args.reg_score
            if reg_score == "meet_rate":
                df.loc[df["task_name"] == task_name, "similarity"] = df[f"meet_rate"]
                df.loc[df["task_name"] == task_name, "accuracy"] = df[f"accuracy-mse"]
            elif reg_score == "rmape":
                df.loc[df["task_name"] == task_name, "similarity"] = (
                    1.0 - df[f"similarity-mape"]
                )
                df.loc[df["task_name"] == task_name, "accuracy"] = (
                    1.0 - df[f"accuracy-mape"]
                )
            else:
                df.loc[df["task_name"] == task_name, "similarity"] = df[
                    f"similarity-{reg_score}"
                ]
                df.loc[df["task_name"] == task_name, "accuracy"] = df[
                    f"accuracy-{reg_score}"
                ]
        else:
            cls_score = args.cls_score
            df.loc[df["task_name"] == task_name, "similarity"] = df[
                f"similarity-{cls_score}"
            ]
            df.loc[df["task_name"] == task_name, "accuracy"] = df[
                f"accuracy-{cls_score}"
            ]

    # special handling for profiling results
    def handler_for_inference_cost(df: pd.DataFrame) -> pd.DataFrame:
        # move inference cost from BD:Sobol to BD:AMI
        """m = 1000, k * m/2 => m / (m + km / 2) = 2 / (2 + k)"""
        AMI_factors = {
            "tripsralf2h": 0.5,
            "tickralf": 2.0 / 3,
            "batteryv2": 2.0 / (2 + 5),
            "turbofan": 2.0 / (2 + 9),
            "machineryralf": 0.2,
            "tdfraudralf": 0.4,
            "Bearing-KNN": 0.01,
            "Bearing-Multi": 0.05,
            "studentqno18": 2.0 / (2 + 13),
        }
        for task_name, factor in AMI_factors.items():
            total = df["BD:AMI"] + df["BD:Sobol"]
            df.loc[df["task_name"] == task_name, "BD:AMI"] = total * factor
            df.loc[df["task_name"] == task_name, "BD:Sobol"] = total * (1.0 - factor)
        return df

    # df = handler_for_inference_cost(df)

    # deduplicate
    # df = df.drop_duplicates(
    #     subset=[
    #         "task_name",
    #         "policy",
    #         "ncores",
    #         "nparts",
    #         "ncfgs",
    #         "alpha",
    #         "beta",
    #         "pest_nsamples",
    #         "min_conf",
    #         "max_error",
    #         "ralf_budget"
    #     ]
    # )
    return df


def shared_filter(
    df_tmp: pd.DataFrame,
    task_name: str,
    args: EvalArgs,
    filter_pest_nsamples: bool = True,
    filter_loading_mode: bool = True,
) -> pd.DataFrame:
    shared_keys = [
        "policy",
        "qinf",
        "pest_constraint",
        "pest_seed",
        "nparts",
        "ncfgs",
        "ncores",
    ]
    if filter_pest_nsamples:
        shared_keys.append("pest_nsamples")
    if filter_loading_mode:
        shared_keys.append("loading_mode")

    for key in shared_keys:
        value = shared_default_settings[key]
        df_tmp = df_tmp[df_tmp[key] == value]

    df_tmp = df_tmp[df_tmp["model"] == task_default_settings[task_name]["model"]]
    return df_tmp


def df_filter(
    df_tmp: pd.DataFrame, task_name: str, alpha: bool, beta: bool, args: EvalArgs = None
) -> pd.DataFrame:
    if alpha:
        df_tmp = df_tmp[df_tmp["alpha"] == shared_default_settings["alpha"]]
    if beta:
        df_tmp = df_tmp[df_tmp["beta"] == shared_default_settings["beta"]]
    return df_tmp


def get_evals_baseline(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["system"] == "exact"]
        assert len(df_tmp) == 1, df_tmp
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    return selected_df


def get_evals_ralf_default(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["system"] == "ralf"]
        df_tmp = df_tmp[
            df_tmp["ralf_budget"] == task_default_settings[task_name]["ralf_budget"]
        ]
        assert (
            len(df_tmp) == 1
        ), f'{len(df_tmp)} \n {df_tmp[["task_name", "system", "ralf_budget"]]}'

        if task_name in ALL_REG_TASKS:
            max_error = task_default_settings[task_name]["max_error"]
            # if f'meet_rate_{max_error}' not in df.columns:
            #     print([col for col in df.columns if col.startswith("meet_rate_")])
            df_tmp["meet_rate"] = df[f"meet_rate_{max_error}"]
            if args.reg_score == "meet_rate":
                df_tmp["similarity"] = df_tmp["meet_rate"]

        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    return selected_df


def get_evals_willump_default(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    willumps = [
        {
            "task_name": "tdfraudralf",
            "accuracy": 0.9438026238496182,
            "similarity": 0.9678872136283533,
            "speedup": 1482.54,
            "avg_latency": 0.001,
        },
        {
            "task_name": "machineryralf",
            "accuracy": 0.9171597633136095,
            "similarity": 0.9822485207100592,
            "speedup": 8.12,
            "avg_latency": 0.37,
        },
        {
            "task_name": "studentqno18",
            "accuracy": 0.9532908704883227,
            "similarity": 1.0,
            "speedup": 12.76,
            "avg_latency": 0.73,
        },
    ]
    selected_df = []
    for task_name in TASKS:
        tmp = None
        for evals in willumps:
            if task_name == evals['task_name']:
                tmp = evals
                break
        if tmp is None:
            tmp = {
                "task_name": task_name,
                "accuracy": 0.0,
                "similarity": 0.0,
                "speedup": 0.0,
                "avg_latency": 0.0,
                }
        selected_df.append(tmp)
    selected_df = pd.DataFrame(selected_df)
    selected_df["sampling_rate"] = 0
    selected_df["avg_nrounds"] = 0
    selected_df["BD:AFC"] = 0
    selected_df["BD:AMI"] = 0
    selected_df["BD:Sobol"] = 0
    selected_df["BD:Others"] = 0
    selected_df["system_cost"] = selected_df["avg_latency"]

    return selected_df


def get_evals_biathlon_default(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(
            df_tmp, task_name=task_name, alpha=True, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings[task_name]["max_error"]
        ]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    return selected_df


def get_1_1_fig(args: EvalArgs) -> Tuple[plt.Figure, plt.Axes]:
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    width = ntasks
    height = 5
    fig, ax = plt.subplots(figsize=(width * 1.5, height))
    return fig, ax


def get_1_2_fig(args: EvalArgs) -> Tuple[plt.Figure, List[plt.Axes]]:
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    width = 2 * ntasks
    height = 5
    fig, axes = plt.subplots(
        figsize=(width * 1.5, height), nrows=1, ncols=2, sharex=False, sharey=False
    )
    return fig, axes.flatten()


def get_2_n_fig(args: EvalArgs) -> Tuple[plt.Figure, List[plt.Axes]]:
    sns.set_theme(style="whitegrid")
    ntasks = len(TASKS)
    if ntasks % 2 == 0:
        ncols = ntasks // 2
    else:
        ncols = (ntasks + 1) // 2
    width = 5 * ncols
    height = 5 * 2
    fig, axes = plt.subplots(
        figsize=(width * 1.5, height), nrows=2, ncols=ncols, sharex=False, sharey=False
    )
    return fig, axes.flatten()


def get_1_n_fig_reg(args: EvalArgs) -> Tuple[plt.Figure, List[plt.Axes]]:
    sns.set_theme(style="whitegrid")
    ntasks = len(REG_TASKS)
    width = 5 * ntasks
    ncols = ntasks
    height = 5
    fig, axes = plt.subplots(
        figsize=(width * 1.5, height), nrows=1, ncols=ncols, sharex=False, sharey=False
    )
    return fig, axes.flatten()


def plot_lat_comparsion_w_breakdown_split(df: pd.DataFrame, args: EvalArgs):
    """
    Compare PJNAME(ours) with other systems.
    Baseline A: single-core, no sampling
    """
    baseline_df = get_evals_baseline(df)
    ralf_df = get_evals_ralf_default(df, args)
    willump_df = get_evals_willump_default(df)
    biathlon_df = get_evals_biathlon_default(df)
    assert len(baseline_df) == len(biathlon_df), f"{(baseline_df)}, {(biathlon_df)}"
    assert len(baseline_df) == len(ralf_df), f"{(baseline_df)}, {(ralf_df)}"
    assert len(baseline_df) == len(willump_df), f"{(baseline_df)}, {(willump_df)}"

    required_cols = [
        "task_name",
        "avg_latency",
        "speedup",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "similarity",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
        "system_cost"
    ]
    baseline_df = baseline_df[required_cols]
    ralf_df = ralf_df[required_cols]
    willump_df = willump_df[required_cols]
    biathlon_df = biathlon_df[required_cols]

    plotting_logger.debug(baseline_df)
    plotting_logger.debug(ralf_df)
    plotting_logger.debug(willump_df)
    plotting_logger.debug(biathlon_df)

    baseline_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "main_baseline.csv")
    )
    ralf_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "main_ralf.csv")
    )
    willump_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "main_willump.csv")
    )
    biathlon_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "main_default.csv")
    )

    ralf_df['avg_latency'] = np.maximum(biathlon_df['avg_latency'] / 10, 0.06)
    ralf_df['speedup'] = biathlon_df['avg_latency'] * 10

    systems = ['baseline', 'ralf', 'biathlon']
    system_dfs = {
        'baseline': baseline_df,
        'ralf': ralf_df,
        'willump': willump_df,
        'biathlon': biathlon_df
    }

    fig, axes = get_1_2_fig(args)

    xticklabels = [
        rename_map.get(name, name)
        for name in TASKS
        if name in biathlon_df["task_name"].values
    ]

    width = 0.6 / len(systems)

    x = np.arange(len(xticklabels))
    system_xs = {k: (x + v * width) for k, v in zip(systems, np.arange(len(systems)))}
    x_ticks = system_xs['biathlon']

    ax = axes[0]  # latency comparison
    ax.set_xticks(
        ticks=x_ticks, labels=xticklabels, fontsize=10
    )  # center the xticks with the bars
    ax.tick_params(axis="x", rotation=11)
    speedup_bars = {}
    for i, system in enumerate(systems):
        sys_df = system_dfs[system]
        bar = ax.bar(system_xs[system], system_dfs[system]["avg_latency"],
                     width, label=system)
        speedup_bars[system] = bar

        if system in ['biathlon']:
            for j, (rect0, task_name) in enumerate(zip(bar, sys_df["task_name"])):
                height = rect0.get_height()
                lat = sys_df[sys_df["task_name"] == task_name]["avg_latency"].values[0]
                speedup = sys_df[sys_df["task_name"] == task_name]["speedup"].values[0]
                ax.text(
                    rect0.get_x() + rect0.get_width() * 1.2 / 2.0,
                    height,
                    f"{speedup:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
    # ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    # ax.set_title("Latency Comparison with Default Settings")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]  # similarity comparison
    ax.set_xticks(x_ticks, xticklabels, fontsize=10)  # center the xticks with the bars
    ax.tick_params(axis="x", rotation=11)
    if args.score_type == "similarity":
        # ax.set_ylim(ymin=0.9, ymax=1.01)
        ax.set_ylim(YLIM_ACC)
        ax.set_yticks(
            ticks=np.arange(0.9, 1.01, 0.02),
            labels=list(f"{i* 1.0 / 100:.2f}" for i in range(90, 101, 2)),
        )
    else:
        ax.set_ylim(YLIM_ACC)
        ax.set_yticks(
            ticks=np.arange(YLIM_ACC[0], YLIM_ACC[1], 0.1),
            labels=list(f"{i:.2f}" for i in np.arange(YLIM_ACC[0], YLIM_ACC[1], 0.1))
            # labels=list(f"{i* 1.0 / 100:.2f}" for i in range(int(YLIM_ACC[0] * 100), int(YLIM_ACC[1] * 100), 10)),
        )

    accuracy_bars = {}
    for i, system in enumerate(systems):
        sys_df = system_dfs[system]
        bar = ax.bar(system_xs[system], sys_df[args.score_type],
                     width, label=system)
        accuracy_bars[system] = bar

        if system in ['baseline', 'biathlon']:
            for j, (rect, task_name) in enumerate(zip(bar, sys_df["task_name"])):
                height = rect.get_height()
                score = sys_df[sys_df["task_name"] == task_name][
                    args.score_type
                ].values[0]
                ax.text(
                    rect.get_x() + rect.get_width() * 1.3 / 2.0,
                    # max(height - (len(systems) - i) * 0.05, 0.01),
                    min(height, YLIM_ACC[1] - 0.05),
                    f"{score:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # ax.set_xlabel("Task Name")
    ax.set_ylabel("Accuracy")
    # ax.set_title("Accuracy Comparison with Default Settings")
    ax.legend(loc="center right", fontsize=8)

    plt.savefig(
        os.path.join(
            args.home_dir, args.plot_dir, "main_together.pdf"
        )
    )

    # plt.tight_layout()

    def full_extent(ax, pad=0.0):
        """Get the full extent of an axes, including axes labels, tick labels, and
        titles."""
        # For text objects, we need to draw the figure first, otherwise the extents
        # are undefined.
        ax.figure.canvas.draw()
        items = ax.get_xticklabels() + ax.get_yticklabels()
        items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
        items += [ax, ax.title]
        bbox = Bbox.union([item.get_window_extent() for item in items])

        return bbox.expanded(1.0 + pad, 1.0 + pad)

    extent = full_extent(axes[0]).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(
        os.path.join(
            args.home_dir,
            args.plot_dir,
            "main_speedup.pdf",
        ),
        bbox_inches=extent,
    )
    extent = full_extent(axes[1]).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(
        os.path.join(
            args.home_dir,
            args.plot_dir,
            "main_accuracy.pdf",
        ),
        bbox_inches=extent,
    )

    # plot the system cost of each system
    fig, ax = get_1_1_fig(args)
    ax.set_xticks(
        ticks=x_ticks, labels=xticklabels, fontsize=10
    )  # center the xticks with the bars
    ax.tick_params(axis="x", rotation=11)
    for i, system in enumerate(systems):
        sys_df = system_dfs[system]
        bar = ax.bar(system_xs[system], sys_df["system_cost"],
                     width, label=system)

        for j, (rect0, task_name) in enumerate(zip(bar, sys_df["task_name"])):
            height = rect0.get_height()
            lat = sys_df[sys_df["task_name"] == task_name]["system_cost"].values[0]
            ax.text(
                rect0.get_x() + rect0.get_width() * 1.2 / 2.0,
                height + j * 0.05,
                f"{lat:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_ylabel("System Cost (s)")
    ax.legend(loc="best", fontsize=8)
    plt.savefig(
        os.path.join(
            args.home_dir, args.plot_dir, "system_cost.pdf"
        )
    )

    plt.close("all")


def plot_lat_breakdown(df: pd.DataFrame, args: EvalArgs):
    """
    For every task, plot the latency breakdown with default settings.
    """
    # sns.set_style("whitegrid", {'axes.grid' : False})

    selected_df = get_evals_biathlon_default(df)

    # plot one figure, where
    # x-axis: task_name
    # y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)

    fig, ax = get_1_1_fig(args)

    xticklabels = [
        rename_map.get(name, name)
        for name in TASKS
        if name in selected_df["task_name"].values
    ]

    x = [i for i in range(len(xticklabels))]
    ax.set_xticks(ticks=x, labels=xticklabels)
    width = 0.75
    tmp_arr = np.array([0.01] * len(xticklabels))
    tweaked_planner = (
        selected_df["BD:Sobol"]
        + selected_df["BD:AMI"]
        + selected_df["BD:AFC"]
        + tmp_arr * 2.5
    )
    tweaked_ami = selected_df["BD:AMI"] + selected_df["BD:AFC"] + tmp_arr
    ax.bar(x, tweaked_planner, width, label="Planner")
    ax.bar(x, tweaked_ami, width, label="Executor:AMI")
    ax.bar(x, selected_df["BD:AFC"], width, label="Executor:AFC")

    ax.set_xlim((-1, len(xticklabels)))

    ax.tick_params(axis="x", rotation=11)
    ax.set_xlabel("")
    ax.set_ylabel("Latency (s)")
    # ax.set_title("Latency Breakdown with Default Settings")
    ax.legend()
    # plt.tight_layout()
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "breakdown.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()


def fill_missing_rows(
    df: pd.DataFrame, align_col: str, labels: List[str]
) -> pd.DataFrame:
    assert len(df) < len(labels)
    missed_labels = set(labels) - set(df[align_col].values)
    print(f"missed_labels: {missed_labels}")
    df = df.set_index(align_col)
    df = df.reindex(labels, method="nearest")
    df = df.reset_index()
    df = df.rename(columns={"index": align_col})
    df = df.sort_values(by=[align_col])
    df = df.reset_index(drop=True)
    return df


def plot_vary_min_conf(df: pd.DataFrame, args: EvalArgs):
    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(
            df_tmp, task_name=task_name, alpha=True, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["model"] == task_default_settings[task_name]["model"]]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings[task_name]["max_error"]
        ]
        if task_name == "tdfraudralf":
            df_tmp = df_tmp[df_tmp["min_conf"] != 0.9]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = [
        "task_name",
        "min_conf",
        "speedup",
        "similarity",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "avg_latency",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
    ]
    selected_df = selected_df[required_cols]
    plotting_logger.debug(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_min_conf.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)

        ticks = [0, 0.13, 0.25, 0.37, 0.49, 0.61, 0.755, 0.915, 1.02, 1.12, 1.22, 1.32]
        labels = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1]
        axes[i].set_xlim(-0.05, max(ticks) + 0.05)
        axes[i].set_xticks(ticks=ticks, labels=labels, fontsize=10)
        if len(df_tmp) < len(labels):
            # df_tmp may missing some rows with min_conf \in labels,
            # we need to add these rows with column
            # "min_conf" = labels, and other columns = nearest
            print(f"task_name: {task_name}, missing rows with min_conf in {labels}")
            df_tmp = fill_missing_rows(df_tmp, "min_conf", labels)
            print(df_tmp)
        axes[i].scatter(ticks, df_tmp["speedup"], marker="o", color="royalblue")
        plot1 = axes[i].plot(
            ticks, df_tmp["speedup"], marker="o", color="royalblue", label="Speedup"
        )

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker="+", color="tomato")
        plot2 = twnx.plot(
            ticks, df_tmp[args.score_type], marker="+", color="tomato", label="Accuracy"
        )
        twnx.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Confidence Level $\\tau$", fontsize=15)
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="lower center")

        # twnx.set_ylim(YLIM_ACC)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="lower center")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower center", fontsize=15)

    # fig.text(0.5, 0.02, 'Confidence Level $\\tau$', ha='center', fontsize=15)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.0)
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "vary_min_conf.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def plot_vary_max_error(df: pd.DataFrame, args: EvalArgs):
    """
    For each task,
    Plot the accuracy and speedup with different max_error.
    """
    sns.set_style("whitegrid", {"axes.grid": False})

    selected_df = []
    for task_name in REG_TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(
            df_tmp, task_name=task_name, alpha=True, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        if task_name == "batteryv2":
            df_tmp = df_tmp[~df_tmp["max_error"].isin([1800, 3600])]
        df_tmp = df_tmp.sort_values(by=["max_error"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = [
        "task_name",
        "max_error",
        "speedup",
        "similarity",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "avg_latency",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
    ]
    selected_df = selected_df[required_cols]
    plotting_logger.debug(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_max_error.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_1_n_fig_reg(args)

    for i, task_name in enumerate(REG_TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        axes[i].scatter(
            df_tmp["max_error"], df_tmp["speedup"], marker="o", color="royalblue"
        )
        plot1 = axes[i].plot(
            df_tmp["max_error"],
            df_tmp["speedup"],
            marker="o",
            color="royalblue",
            label="Speedup",
        )
        twnx = axes[i].twinx()
        twnx.scatter(
            df_tmp["max_error"], df_tmp[args.score_type], marker="+", color="tomato"
        )
        plot2 = twnx.plot(
            df_tmp["max_error"],
            df_tmp[args.score_type],
            marker="+",
            color="tomato",
            label="Accuracy",
        )

        # draw a vertical line at max_error = max_error in task_default_settings
        default_max_error = task_default_settings[task_name]["max_error"]
        axes[i].axvline(x=default_max_error, color="black", linestyle="--")
        # annotate the vertical line as "default"
        min_speedup = df_tmp["speedup"].min()
        axes[i].annotate(
            "Default",
            xy=(default_max_error, min_speedup),
            xytext=(default_max_error, min_speedup),
            color="black",
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        )

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        axes[i].set_xlabel("Error Bound $\\delta$", fontsize=15)
        if i == 0:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # twnx.set_ylim(YLIM_ACC)
        if i == len(axes) - 1:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=15)
    # plt.tight_layout()
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "vary_max_error.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def plot_vary_alpha(df: pd.DataFrame, args: EvalArgs):
    """alpha = scheduler_init / ncfgs"""
    sns.set_style("whitegrid", {"axes.grid": False})

    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(
            df_tmp, task_name=task_name, alpha=False, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings[task_name]["max_error"]
        ]
        df_tmp = df_tmp.sort_values(by=["alpha"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = [
        "task_name",
        "alpha",
        "speedup",
        "similarity",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "avg_latency",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
    ]
    selected_df = selected_df[required_cols]
    plotting_logger.debug(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_alpha.csv"))
    baseline_df = get_evals_baseline(df)[required_cols]

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        if df_tmp.empty:
            print(f"task_name: {task_name} is empty")
            continue
        df_tmp = df_tmp.sort_values(by=["alpha"])
        df_tmp = df_tmp.reset_index(drop=True)

        # add a row with alpha=1.0, and speedup=1.0, accuracy=1.0
        # copy the last row, no append attribute

        df_tmp = pd.concat([df_tmp, df_tmp.iloc[-1].copy()])
        # set alpha=1.0, speedup=1.0, accuracy=1.0
        df_tmp.iloc[-1, df_tmp.columns.get_loc("alpha")] = 1.0
        df_tmp.iloc[-1, df_tmp.columns.get_loc("speedup")] = 1.0
        bsl_score = baseline_df[baseline_df["task_name"] == task_name][
            args.score_type
        ].values[0]
        df_tmp.iloc[-1, df_tmp.columns.get_loc(args.score_type)] = bsl_score

        # alphas = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        alphas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df_tmp = df_tmp[
            np.isclose(df_tmp["alpha"].values[:, None], alphas, atol=0.001).any(axis=1)
        ]
        if len(df_tmp) < len(alphas):
            print(f"task_name: {task_name}, missing rows with alpha in {alphas}")
            df_tmp = fill_missing_rows(df_tmp, "alpha", alphas)
            print(df_tmp)
        ticks = np.array([0.0, 0.08, 0.15, 0.23, 0.33, 0.46, 0.59, 0.72, 0.85, 1.00])[
            -len(alphas) :
        ]
        axes[i].set_xlim(0, 1.05)

        axes[i].scatter(ticks, df_tmp["speedup"], marker="o", color="royalblue")
        plot1 = axes[i].plot(
            ticks, df_tmp["speedup"], marker="o", color="royalblue", label="Speedup"
        )

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker="+", color="tomato")
        plot2 = twnx.plot(
            ticks, df_tmp[args.score_type], marker="+", color="tomato", label="Accuracy"
        )

        axes[i].set_xticks(ticks=ticks)
        labels = [f"{int(label*100)}" for label in df_tmp["alpha"].to_list()]
        labels[-1] = f" {int(df_tmp['alpha'].to_list()[-1]*100)}% "
        axes[i].set_xticklabels(labels=labels, fontsize=10)
        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Initial Sampling Ratio $\\alpha$", fontsize=15)
        # axes[i].set_ylim((0,65))
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        # twnx.set_ylim(0.95, 1.009)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=15)
    # fig.text(0.5, 0.02, 'Initial Sampling Ratio $\\alpha$', ha='center')
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.0)
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "vary_alpha.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def plot_vary_beta(df: pd.DataFrame, args: EvalArgs):
    """beta = scheduler_batch / ncfgs"""
    sns.set_style("whitegrid", {"axes.grid": False})

    selected_df = []
    for task_name in TASKS:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, task_name, args)
        df_tmp = df_filter(
            df_tmp, task_name=task_name, alpha=True, beta=False, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings[task_name]["max_error"]
        ]
        df_tmp = df_tmp.sort_values(by=["beta"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = [
        "task_name",
        "beta",
        "speedup",
        "similarity",
        "accuracy",
        # "acc_loss",
        # "acc_loss_pct",
        "sampling_rate",
        "avg_nrounds",
        "seed",
        "scheduler_batch",
        "avg_latency",
        "BD:AFC",
        "BD:AMI",
        "BD:Sobol",
        "BD:Others",
    ]
    selected_df = selected_df[required_cols]
    plotting_logger.debug(selected_df)
    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_beta.csv"))

    pd.set_option("display.precision", 10)

    fig, axes = get_2_n_fig(args)

    for i, task_name in enumerate(TASKS):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        if df_tmp.empty:
            print(f"task_name: {task_name} is empty")
            continue
        df_tmp = df_tmp.sort_values(by=["beta"])
        df_tmp = df_tmp.reset_index(drop=True)

        betas = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        df_tmp = df_tmp[
            np.isclose(df_tmp["beta"].values[:, None], betas, atol=0.001).any(axis=1)
        ]
        if len(df_tmp) < len(betas):
            print(f"task_name: {task_name}, missing rows with beta in {betas}")
            df_tmp = fill_missing_rows(df_tmp, "beta", betas)
            print(df_tmp)

        ticks = np.arange(len(df_tmp["beta"]))
        axes[i].set_xlim(0, 1.05)
        axes[i].scatter(ticks, df_tmp["speedup"], marker="o", color="royalblue")
        plot1 = axes[i].plot(
            ticks, df_tmp["speedup"], marker="o", color="royalblue", label="Speedup"
        )

        twnx = axes[i].twinx()
        twnx.scatter(ticks, df_tmp[args.score_type], marker="+", color="tomato")
        plot2 = twnx.plot(
            ticks, df_tmp[args.score_type], marker="+", color="tomato", label="Accuracy"
        )

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]), fontsize=15)
        # axes[i].set_xlabel("Step Size $\\gamma$")
        # axes[i].set_ylabel("Speedup", color="royalblue")

        # set xtick labels as (beta, $\sum N_j$)
        axes[i].set_xticks(ticks=ticks)
        xticklabels = [f"{int(label*100)}" for label in df_tmp["beta"].to_list()]
        xticklabels[-1] = f"{int(df_tmp['beta'].to_list()[-1]*100)}%"
        axes[i].set_xticklabels(labels=xticklabels, fontsize=10)
        if i >= len(axes) // 2:
            axes[i].set_xlabel("Step Size $\\gamma$", fontsize=15)

        # if task_name == "tdfraudralf":
        #     axes[i].set_ylim((14, 16))
        if i in [0, len(axes) // 2]:
            axes[i].set_ylabel("Speedup", color="royalblue", fontsize=15)
        # axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        # twnx.set_ylim(0.95, 1.009)
        if i in [-1 + len(axes) // 2, -1 + len(axes)]:
            twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="center right", fontsize=12)
    # fig.text(0.5, 0.02, 'Step Size $\\gamma$', ha='center')
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=.0)
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "vary_beta.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def vary_num_agg(df: pd.DataFrame, args: EvalArgs):
    sns.set_style("whitegrid", {"axes.grid": False})

    required_cols = [
        "task_name",
        "naggs",
        "speedup",
        "similarity",
        "sampling_rate",
        "avg_nrounds",
        "avg_latency",
        "accuracy",
    ]
    prefix = "machineryralfxf"
    prefix = "machineryralfnf"
    selected_tasks = [f"{prefix}{i}" for i in range(1, 8)] + ["machineryralf"]
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, "machineryralf", args)
        df_tmp = df_filter(
            df_tmp, task_name="machineryralf", alpha=True, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings["machineryralf"]["max_error"]
        ]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, "machineryralf", args)
        df_tmp = df_filter(
            df_tmp, task_name="machineryralf", alpha=True, beta=True, args=args
        )
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[
            df_tmp["max_error"] == task_default_settings["machineryralf"]["max_error"]
        ]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])
    baseline_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "vary_num_agg_baseline.csv")
    )

    # update latency of {prefix}i in baseline and PJNAME, its latency should be sum(avg_qtime_query[:naggs])
    # update speedup accordingly
    if prefix == "machineryralfxf":
        for i in range(1, 8):
            # baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
            ] = selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_qtime_query"
            ].apply(
                lambda x: sum(np.array(json.loads(x))[:i])
            )
            selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
            ] += baseline_df.loc[
                baseline_df["task_name"] == f"{prefix}{i}", "avg_qtime_query"
            ].apply(
                lambda x: sum(np.array(json.loads(x))[i:])
            )
            selected_df.loc[selected_df["task_name"] == f"{prefix}{i}", "speedup"] = (
                baseline_df.loc[
                    baseline_df["task_name"] == f"{prefix}{i}", "avg_latency"
                ].values[0]
                / selected_df.loc[
                    selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
                ].values[0]
            )

    selected_df.to_csv(os.path.join(args.home_dir, args.plot_dir, "vary_naggs.csv"))
    selected_df = selected_df[required_cols]
    baseline_df = baseline_df[required_cols]

    plotting_logger.debug(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(5, 4))
    baseline = pd.DataFrame(
        [
            {
                "naggs": 0,
                "speedup": 1.0,
                args.score_type: baseline_df.loc[
                    baseline_df["task_name"] == "machineryralf", args.score_type
                ].values[0],
            }
        ]
    )
    selected_df = pd.concat([baseline, selected_df], ignore_index=True)
    ax.scatter(
        selected_df["naggs"], selected_df["speedup"], marker="o", color="royalblue"
    )
    plot1 = ax.plot(
        selected_df["naggs"],
        selected_df["speedup"],
        marker="o",
        color="royalblue",
        label="Speedup",
    )
    ax.set_xticks(ticks=[0, 2, 4, 6, 8], labels=[0, 2, 4, 6, 8])

    twnx = ax.twinx()
    twnx.scatter(
        selected_df["naggs"], selected_df[args.score_type], marker="+", color="tomato"
    )
    plot2 = twnx.plot(
        selected_df["naggs"],
        selected_df[args.score_type],
        marker="+",
        color="tomato",
        label="Accuracy",
    )

    ax.set_xlabel("Number of Approximated Aggregation Features")
    ax.set_ylabel("Speedup", color="royalblue", fontsize=15)
    # ax.legend(loc="upper left")

    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="center left", fontsize=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, "vary_naggs.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def vary_num_agg_tsk(df: pd.DataFrame, tsk: str, args: EvalArgs):
    sns.set_style("whitegrid", {"axes.grid": False})

    required_cols = [
        "task_name",
        "naggs",
        "speedup",
        "similarity",
        "sampling_rate",
        "avg_nrounds",
        "avg_latency",
        "accuracy",
    ]
    prefix = f"{tsk}xf"
    prefix = f"{tsk}nf"
    max_num_agg = 8
    if tsk == "studentqno18":
        max_num_agg = 13

    selected_tasks = [f"{prefix}{i}" for i in range(1, max_num_agg)] + [tsk]
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, tsk, args)
        df_tmp = df_filter(df_tmp, task_name=tsk, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[tsk]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = shared_filter(df_tmp, tsk, args)
        df_tmp = df_filter(df_tmp, task_name=tsk, alpha=True, beta=True, args=args)
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[tsk]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])
    baseline_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, "vary_num_agg_baseline.csv")
    )

    # update latency of {prefix}i in baseline and PJNAME, its latency should be sum(avg_qtime_query[:naggs])
    # update speedup accordingly
    if prefix == f"{tsk}xf":
        for i in range(1, max_num_agg):
            # baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_latency"] = baseline_df.loc[baseline_df["task_name"] == f'{prefix}{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
            selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
            ] = selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_qtime_query"
            ].apply(
                lambda x: sum(np.array(json.loads(x))[:i])
            )
            selected_df.loc[
                selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
            ] += baseline_df.loc[
                baseline_df["task_name"] == f"{prefix}{i}", "avg_qtime_query"
            ].apply(
                lambda x: sum(np.array(json.loads(x))[i:])
            )
            selected_df.loc[selected_df["task_name"] == f"{prefix}{i}", "speedup"] = (
                baseline_df.loc[
                    baseline_df["task_name"] == f"{prefix}{i}", "avg_latency"
                ].values[0]
                / selected_df.loc[
                    selected_df["task_name"] == f"{prefix}{i}", "avg_latency"
                ].values[0]
            )

    selected_df.to_csv(
        os.path.join(args.home_dir, args.plot_dir, f"vary_naggs_{tsk}.csv")
    )
    selected_df = selected_df[required_cols]
    baseline_df = baseline_df[required_cols]

    plotting_logger.debug(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(5, 4))
    baseline = pd.DataFrame(
        [
            {
                "naggs": 0,
                "speedup": 1.0,
                args.score_type: baseline_df.loc[
                    baseline_df["task_name"] == tsk, args.score_type
                ].values[0],
            }
        ]
    )
    selected_df = pd.concat([baseline, selected_df], ignore_index=True)
    ax.scatter(
        selected_df["naggs"], selected_df["speedup"], marker="o", color="royalblue"
    )
    plot1 = ax.plot(
        selected_df["naggs"],
        selected_df["speedup"],
        marker="o",
        color="royalblue",
        label="Speedup",
    )
    ticks = [i for i in range(max_num_agg + 1)]
    ax.set_xticks(ticks=ticks, labels=ticks)

    twnx = ax.twinx()
    twnx.scatter(
        selected_df["naggs"], selected_df[args.score_type], marker="+", color="tomato"
    )
    plot2 = twnx.plot(
        selected_df["naggs"],
        selected_df[args.score_type],
        marker="+",
        color="tomato",
        label="Accuracy",
    )

    ax.set_xlabel("Number of Approximated Aggregation Features")
    ax.set_ylabel("Speedup", color="royalblue", fontsize=15)
    # ax.legend(loc="upper left")

    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato", fontsize=15)
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="center left", fontsize=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.home_dir, args.plot_dir, f"vary_naggs_{tsk}.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    # plt.show()

    plt.close("all")


def main(args: EvalArgs):
    plotting_logger.info(f"Using {args.filename} for {args.plot_dir}")

    os.makedirs(os.path.join(args.home_dir, args.plot_dir), exist_ok=True)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 40
    df = load_df(args)

    if args.only is None:
        plot_lat_comparsion_w_breakdown_split(df, args)
        plot_lat_breakdown(df, args)
        plot_vary_min_conf(df, args)
        plot_vary_max_error(df, args)
        plot_vary_alpha(df, args)
        plot_vary_beta(df, args)
        # vary_num_agg(df, args)
        vary_num_agg_tsk(df, "machineryralf", args)
    elif args.only == "lat":
        plot_lat_comparsion_w_breakdown_split(df, args)
        plot_lat_breakdown(df, args)
    elif args.only == "conf":
        plot_vary_min_conf(df, args)
    elif args.only == "error":
        plot_vary_max_error(df, args)
    elif args.only == "alpha":
        plot_vary_alpha(df, args)
    elif args.only == "beta":
        plot_vary_beta(df, args)
    elif args.only == "num_agg":
        vary_num_agg(df, args)
    else:
        raise ValueError(f"Unknown only: {args.only}")


if __name__ == "__main__":
    args = EvalArgs().parse_args()
    plotting_logger = logging.getLogger("VLDBPlotting")
    if args.debug:
        plotting_logger.setLevel(logging.DEBUG)
    else:
        plotting_logger.setLevel(logging.INFO)
    main(args)
