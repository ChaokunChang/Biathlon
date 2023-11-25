import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
import json

from tap import Tap

PJNAME = "Biathlon"


class EvalArgs(Tap):
    home_dir: str = "./cache"
    filename: str = "evals.csv"
    loading_mode: int = 0
    ncores: int = 1
    beta_of_all: bool = False


def load_df(args: EvalArgs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(args.home_dir, args.filename))
    df['BD:Others'] = df['avg_latency'] - df['BD:AFC'] - df['BD:AMI'] - df['BD:Sobol']

    # special handling for profiling results
    def handler_soboltime(df: pd.DataFrame) -> pd.DataFrame:
        # AMI and Sobol share some computation
        # update BD:Sobol as max(BD:Sobol - BD:AMI, 0)
        df["BD:Sobol"] = df["BD:Sobol"] - df["BD:AMI"]
        df["BD:Sobol"] = df["BD:Sobol"].apply(lambda x: max(x, 0))
        old_lat = df["avg_latency"]
        df["avg_latency"] = df['BD:AFC'] + df['BD:AMI'] + df['BD:Sobol'] + df['BD:Others']
        df['speedup'] = (old_lat * df['speedup']) / df['avg_latency']
        return df

    def handler_filter_ncfgs(df: pd.DataFrame) -> pd.DataFrame:
        # delte rows with ncfgs = 2
        df = df[df["ncfgs"] != 2]
        # df = df[df["ncfgs"] != 50]
        return df

    def handler_loading_mode(df: pd.DataFrame) -> pd.DataFrame:
        # keep only the rows with the specified mode
        loading_mode = args.loading_mode
        df = df[df["loading_mode"] == loading_mode]
        return df

    # df = handler_soboltime(df)
    df = handler_filter_ncfgs(df)
    df = handler_loading_mode(df)
    return df


tasks = [
    "trips",
    # "tick-v1",
    "Tick-Price",
    # "cheaptrips",
    "Bearing-MLP",
    # "machinery-v2",
    "Bearing-KNN",
]

reg_tasks = [
    "trips",
    # "tick-v1",
    "Tick-Price",
]

shared_default_settings = {
    "policy": "optimizer",
    "ncores": None,
    "min_conf": 0.95,
    "nparts": 100,
    "ncfgs": 100,
    "scheduler_init": 5,
    "scheduler_batch": 5
}
task_default_settings = {
    "trips": {
        "model_name": "lgbm",
        "max_error": 1.0
    },
    "tick-v1": {
        "model_name": "lr",
        "max_error": 0.01,
    },
    "Tick-Price": {
        "model_name": "lr",
        "max_error": 0.01,
    },
    "cheaptrips": {
        "model_name": "xgb",
        "max_error": 0.0,
    },
    "Bearing-MLP": {
        "model_name": "mlp",
        "max_error": 0.0,
    },
    "machinery-v2": {
        "model_name": "dt",
        "max_error": 0.0,
    },
    "Bearing-KNN": {
        "model_name": "knn",
        "max_error": 0.0,
    },
}


def get_evals_baseline(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings[task_name]["model_name"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    # print(selected_df)
    return selected_df


def get_evals_basic(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings[task_name]["model_name"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    return selected_df


def get_evals_with_default_settings(df: pd.DataFrame, args: EvalArgs = None) -> pd.DataFrame:
    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings[task_name]["model_name"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    # print(selected_df)
    return selected_df


def plot_lat_comparsion_w_breakdown(df: pd.DataFrame, args: EvalArgs):
    """
    Compare PJNAME(ours) with other systems.
    Baseline A: single-core, no sampling
    """
    baseline_df = get_evals_baseline(df)
    default_df = get_evals_with_default_settings(df)
    assert len(baseline_df) == len(default_df)
    required_cols = ["task_name", "avg_latency", "speedup",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "similarity", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    baseline_df = baseline_df[required_cols]
    default_df = default_df[required_cols]
    print(baseline_df)
    print(default_df)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(figsize=(10, 8), nrows=2, ncols=1, sharex=True, sharey=False)

    xticklabels = default_df['task_name'].values

    width = 0.35
    x = [i for i in range(len(tasks))]
    x1 = [i - width for i in x]

    ax = axes[0]  # latency comparison
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    # draw baseline on x1, from bottom to up is AFC, AMI, Sobol, Others
    ax.bar(x1, baseline_df['BD:AFC'], width, label="Baseline-AFC")
    ax.bar(x1, baseline_df['BD:AMI'] + baseline_df['BD:Sobol'], width, bottom=baseline_df['BD:AFC'], label="Baseline-Others")
    # ax.bar(x1, baseline_df['BD:Sobol'], width, bottom=baseline_df['BD:AFC'] + baseline_df['BD:AMI'], label="Baseline-Planner")
    # ax.bar(x1, baseline_df['BD:Others'], width, bottom=baseline_df['BD:AFC'] + baseline_df['BD:AMI'] + baseline_df['BD:Sobol'], label="Baseline-Others")

    # draw default on x, from bottom to up is AFC, AMI, Sobol, Others
    ax.bar(x, default_df['BD:AFC'], width, label=f"{PJNAME}-AFC")
    ax.bar(x, default_df['BD:AMI'], width, bottom=default_df['BD:AFC'], label=f"{PJNAME}-AMI")
    ax.bar(x, default_df['BD:Sobol'], width, bottom=default_df['BD:AFC'] + default_df['BD:AMI'], label=f"{PJNAME}-Planner")
    # ax.bar(x, default_df['BD:Others'], width, bottom=default_df['BD:AFC'] + default_df['BD:AMI'] + default_df['BD:Sobol'], label=f"{PJNAME}-Others")

    # add speedup on top of the bar of PJNAME
    for i, task_name in enumerate(default_df['task_name']):
        lat = default_df[default_df["task_name"] == task_name]["avg_latency"].values[0]
        speedup = default_df[default_df["task_name"] == task_name]["speedup"].values[0]
        ax.text(i, lat + 0.01, "{:.2f}x".format(speedup), ha="center")

    ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Comparison with Default Settings")
    ax.legend()

    ax = axes[1]  # similarity comparison
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)

    # draw baseline on x1, similarity
    ax.bar(x1, baseline_df['similarity'], width, label="Baseline")
    # draw default on x, similarity
    ax.bar(x, default_df['similarity'], width, label=f"{PJNAME}")

    # add acc_loss on top of the bar of PJNAME
    for i, task_name in enumerate(default_df['task_name']):
        similarity = default_df[default_df["task_name"] == task_name]["similarity"].values[0]
        ax.text(i, similarity + 0.01, "{:.2f}%".format(similarity*100), ha="center")

    ax.set_xlabel("Task Name")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison with Default Settings")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "lat_comparison_default_w_beakdown.pdf"))
    # plt.show()

    plt.close("all")


def plot_lat_breakdown(df: pd.DataFrame, args: EvalArgs):
    """
    For every task, plot the latency breakdown with default settings.
    """
    selected_df = get_evals_with_default_settings(df)

    # plot one figure, where
    # x-axis: task_name
    # y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)

    selected_df["sns_AFC"] = selected_df["BD:AFC"]
    selected_df["sns_AMI"] = selected_df["BD:AMI"] + selected_df["sns_AFC"]
    selected_df["sns_Sobol"] = selected_df["BD:Sobol"] + selected_df["sns_AMI"]
    # selected_df["sns_Others"] = selected_df["BD:Others"] + selected_df["sns_Sobol"]

    fig, ax = plt.subplots(figsize=(8, 5))
    # sns.barplot(x="task_name", y="sns_Others", data=selected_df, ax=ax, label="Others")
    sns.barplot(x="task_name", y="sns_Sobol", data=selected_df, ax=ax, label="Planner")
    sns.barplot(x="task_name", y="sns_AMI", data=selected_df, ax=ax, label="Executor:AMI")
    sns.barplot(x="task_name", y="sns_AFC", data=selected_df, ax=ax, label="Executor:AFC")
    ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Breakdown with Default Settings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "lat_breakdown_default.pdf"))
    # plt.show()


def plot_vary_min_conf(df: pd.DataFrame, args: EvalArgs):
    df = get_evals_basic(df)
    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        # df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "min_conf", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    sns.set_theme(style="whitegrid")
    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    elif len(tasks) in [5, 6]:
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        df_tmp = df_tmp.sort_values(by=["min_conf"])
        df_tmp = df_tmp.reset_index(drop=True)

        axes[i].scatter(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="blue", label="Accuracy")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Min Confidence")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Accuracy", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_min_conf.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_max_error(df: pd.DataFrame, args: EvalArgs):
    """
    For each task,
    Plot the accuracy and speedup with different max_error.
    """
    df = get_evals_basic(df)
    selected_df = []
    for task_name in reg_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        # df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["max_error"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "max_error", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(reg_tasks) == 2:
        fig, axes = plt.subplots(figsize=(12, 6), nrows=1, ncols=2, sharex=False, sharey=True)
    elif len(reg_tasks) > 2:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(reg_tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="blue", label="Accuracy")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Max Error")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Accuracy", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_max_error.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_alpha(df: pd.DataFrame, args: EvalArgs):
    """ alpha = scheduler_init / ncfgs
    """
    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        # df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df['alpha'] = selected_df['scheduler_init'] / selected_df['ncfgs']
    required_cols = ["task_name", "alpha", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    elif len(tasks) in [5, 6]:
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        df_tmp = df_tmp.sort_values(by=["alpha"])
        df_tmp = df_tmp.reset_index(drop=True)

        axes[i].scatter(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="blue", label="Accuracy")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Accuracy", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_alpha.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_beta(df: pd.DataFrame, args: EvalArgs):
    """ beta = scheduler_batch / ncfgs
    """
    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        # df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df['beta'] = selected_df['scheduler_batch'] / selected_df['ncfgs']
    if args.beta_of_all:
        selected_df["beta"] /= selected_df['naggs']
    required_cols = ["task_name", "beta", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    elif len(tasks) in [5, 6]:
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        df_tmp = df_tmp.sort_values(by=["beta"])
        df_tmp = df_tmp.reset_index(drop=True)

        axes[i].scatter(df_tmp["beta"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["beta"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="blue", label="Accuracy")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Beta")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Accuracy", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_beta.pdf"))
    # plt.show()

    plt.close("all")


def vary_alpha_beta(df: pd.DataFrame, args: EvalArgs):
    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        # df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        # df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df['alpha'] = selected_df['scheduler_init'] / selected_df['ncfgs']
    selected_df['beta'] = selected_df['scheduler_batch'] / selected_df['ncfgs']
    if args.beta_of_all:
        selected_df["beta"] /= selected_df['naggs']
    required_cols = ["task_name", "alpha", "beta", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    elif len(tasks) in [5, 6]:
        fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        # for each scheduler_batch, plot two lines: speedup and similarity
        # two lines are plotted in the same figure with different markers but same color
        # speedup is dashed line, similarity is solid line
        all_betas = df_tmp["beta"].unique()
        all_colors = sns.color_palette("hls", len(all_betas))
        for beta, color in zip(all_betas, all_colors): 
            df_tmp_beta = df_tmp[df_tmp["beta"] == beta]
            df_tmp_beta = df_tmp_beta.sort_values(by=["alpha"])
            df_tmp_beta = df_tmp_beta.reset_index(drop=True)

            axes[i].scatter(df_tmp_beta["alpha"], df_tmp_beta["speedup"], marker='o', color=color)
            axes[i].plot(df_tmp_beta["alpha"], df_tmp_beta["speedup"], marker='o', color=color, label=f"Speedup(beta={beta})", linestyle="--")

            twnx = axes[i].twinx()
            twnx.scatter(df_tmp_beta["alpha"], df_tmp_beta[acc_metric], marker='+', color=color)
            twnx.plot(df_tmp_beta["alpha"], df_tmp_beta[acc_metric], marker='+', color=color, label=f"Accuracy")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Speedup")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Accuracy")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_alpha_beta.pdf"))
    # plt.show()

    plt.close()


def vary_num_agg(df: pd.DataFrame, args: EvalArgs):
    required_cols = ["task_name", "naggs", "speedup", "similarity",
                     "avg_latency", "accuracy"]
    selected_tasks = [f'machineryxf{i}' for i in range(1, 8)] + ['Bearing-MLP']
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Bearing-MLP"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Bearing-MLP"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])

    # update latency of machineryxfi in baseline and PJNAME, its latency should be sum(avg_qtime_query[:naggs])
    # update speedup accordingly
    for i in range(1, 8):
        # baseline_df.loc[baseline_df["task_name"] == f'machineryxf{i}', "avg_latency"] = baseline_df.loc[baseline_df["task_name"] == f'machineryxf{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
        selected_df.loc[selected_df["task_name"] == f'machineryxf{i}', "avg_latency"] = selected_df.loc[selected_df["task_name"] == f'machineryxf{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[:i]))
        selected_df.loc[selected_df["task_name"] == f'machineryxf{i}', "avg_latency"] += baseline_df.loc[baseline_df["task_name"] == f'machineryxf{i}', "avg_qtime_query"].apply(lambda x: sum(np.array(json.loads(x))[i:]))
        selected_df.loc[selected_df["task_name"] == f'machineryxf{i}', "speedup"] = baseline_df.loc[baseline_df["task_name"] == f'machineryxf{i}', "avg_latency"].values[0] / selected_df.loc[selected_df["task_name"] == f'machineryxf{i}', "avg_latency"].values[0]

    baseline_df = baseline_df[required_cols]
    selected_df = selected_df[required_cols]

    print(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="orange")
    ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="orange", label="Speedup")

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df["similarity"], marker='+', color="blue")
    twnx.plot(selected_df["naggs"], selected_df["similarity"], marker='+', color="blue", label="Accuracy")

    ax.set_xlabel("Number of Aggregation Operators")
    ax.set_ylabel("Speedup", color="orange")
    ax.legend(loc="upper left")

    twnx.set_ylabel("Accuracy", color="blue")
    twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_num_agg.pdf"))
    # plt.show()

    plt.close("all")


def vary_num_nf(df: pd.DataFrame, args: EvalArgs):
    required_cols = ["task_name", "naggs", "speedup", "similarity",
                     "avg_latency", "accuracy"]
    selected_tasks = [f'machineryf{i}' for i in range(1, 8)] + ['Bearing-MLP']
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Bearing-MLP"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["naggs"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Bearing-MLP"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Bearing-MLP"]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["naggs"])

    baseline_df = baseline_df[required_cols]
    selected_df = selected_df[required_cols]

    print(selected_df)

    # plot as a scatter line chart
    # x-axis: naggs
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="orange")
    ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="orange", label="Speedup")

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df["similarity"], marker='+', color="blue")
    twnx.plot(selected_df["naggs"], selected_df["similarity"], marker='+', color="blue", label="Accuracy")

    ax.set_xlabel("Number of Aggregation Operators")
    ax.set_ylabel("Speedup", color="orange")
    ax.legend(loc="upper left")

    twnx.set_ylabel("Accuracy", color="blue")
    twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_num_nf.pdf"))
    # plt.show()

    plt.close("all")


def main(args: EvalArgs):
    df = load_df(args)
    # plot_lat_comparsion_w_breakdown(df, args)
    # plot_lat_breakdown(df, args)
    # plot_vary_min_conf(df, args)
    # plot_vary_max_error(df, args)
    # plot_vary_alpha(df, args)
    # plot_vary_beta(df, args)
    # vary_alpha_beta(df, args)
    vary_num_agg(df, args)
    vary_num_nf(df, args)

    # print(get_evals_with_default_settings(df))


if __name__ == "__main__":
    args = EvalArgs().parse_args()
    shared_default_settings["ncores"] = args.ncores
    main(args)
