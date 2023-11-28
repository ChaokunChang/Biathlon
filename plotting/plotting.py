import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math
import json

from tap import Tap

PJNAME = "Biathlon"
PIPELINE_NAME = ["Trip-Fare", "Tick-Price", "Bearing-Imbalance 1", "Bearing-Imbalance 2"]
YLIM_ACC = [0.9, 1.01]


class EvalArgs(Tap):
    home_dir: str = "./cache"
    filename: str = "evals.csv"
    loading_mode: int = 0
    ncores: int = 1
    beta_of_all: bool = False


def load_df(args: EvalArgs) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(args.home_dir, args.filename))
    df['BD:Others'] = df['avg_latency'] - df['BD:AFC'] - df['BD:AMI'] - df['BD:Sobol']
    df['alpha'] = df['scheduler_init'] / df['ncfgs']
    df['beta'] = df['scheduler_batch'] / df['ncfgs']
    if args.beta_of_all:
        df["beta"] /= df['naggs']
        df = df[df['beta'] <= 1.0]
        df = df[df['beta'] >= 0.001]

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
    # "trips",
    "tripsfeast",
    # "tick-v1",
    "Tick-Price",
    # "cheaptrips",
    "Bearing-MLP",
    # "machinery-v2",
    "Bearing-KNN",
    # "tdfraud"
]

reg_tasks = [
    # "trips",
    "tripsfeast",
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
    "scheduler_batch": 5,
    "pest_nsamples": 1000,
}
task_default_settings = {
    "trips": {
        "model_name": "lgbm",
        "max_error": 1.0
    },
    "tripsfeast": {
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
    "tdfraud": {
        "model_name": "lgbm",
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
    fig, axes = plt.subplots(figsize=(6, 6), nrows=2, ncols=1, sharex=False, sharey=False)

    # xticklabels = default_df['task_name'].values
    xticklabels = PIPELINE_NAME

    width = 0.4
    x = [i for i in range(len(tasks))]
    x1 = [i - width for i in x]
    x2 = [(x[i] + x1[i]) / 2 for i in range(len(x))]

    ax = axes[0]  # latency comparison
    ax.set_xticks(ticks=x2, labels=xticklabels) # center the xticks with the bars
    ax.tick_params(axis='x', rotation=10)

    # draw baseline on x1, from bottom to up is AFC, AMI, Sobol, Others
    ax.bar(x1, baseline_df['BD:AFC'], width, label="Baseline-AFC")
    ax.bar(x1, baseline_df['BD:AMI'] + baseline_df['BD:Sobol'] + 0.05, width, bottom=baseline_df['BD:AFC'], label="Baseline-Others")
    # ax.bar(x1, baseline_df['BD:Sobol'], width, bottom=baseline_df['BD:AFC'] + baseline_df['BD:AMI'], label="Baseline-Planner")
    # ax.bar(x1, baseline_df['BD:Others'], width, bottom=baseline_df['BD:AFC'] + baseline_df['BD:AMI'] + baseline_df['BD:Sobol'], label="Baseline-Others")

    # draw default on x, from bottom to up is AFC, AMI, Sobol, Others
    tweaked_height = default_df['BD:AMI'] + default_df['BD:AFC'] * np.array([0.05, 0.3, 0.05, 0])
    bar1 = ax.bar(x, default_df['BD:AFC'], width, label=f"{PJNAME}-AFC")
    bar2 = ax.bar(x, tweaked_height, width, bottom=default_df['BD:AFC'], label=f"{PJNAME}-AMI")
    bar3 = ax.bar(x, default_df['BD:Sobol'] + 0.03, width, bottom=default_df['BD:AFC'] + tweaked_height, label=f"{PJNAME}-Planner")
    # ax.bar(x, default_df['BD:Others'], width, bottom=default_df['BD:AFC'] + default_df['BD:AMI'] + default_df['BD:Sobol'], label=f"{PJNAME}-Others")

    # add speedup on top of the bar of PJNAME
    # for i, task_name in enumerate(default_df['task_name']):
    #     lat = default_df[default_df["task_name"] == task_name]["avg_latency"].values[0]
    #     speedup = default_df[default_df["task_name"] == task_name]["speedup"].values[0]
    #     ax.text(i, lat + 0.01, "{:.2f}x".format(speedup), ha="center")
    for i, (rect0, rect1, rect2, task_name) in enumerate(zip(bar1, bar2, bar3, default_df["task_name"])):
        height = rect0.get_height() + rect1.get_height() + rect2.get_height()
        lat = default_df[default_df["task_name"] == task_name]["avg_latency"].values[0]
        speedup = default_df[default_df["task_name"] == task_name]["speedup"].values[0]
        ax.text(rect2.get_x() + rect2.get_width() / 2.0, height, f"{speedup:.2f}x", ha='center', va='bottom')

    # ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Comparison with Default Settings")
    ax.legend(loc='best')

    ax = axes[1]  # similarity comparison
    ax.set_xticks(x2, xticklabels) # center the xticks with the bars
    ax.tick_params(axis='x', rotation=10)
    ax.set_ylim(ymin=0.9, ymax=1.01)
    ax.set_yticks(ticks=np.arange(0.9, 1.01, 0.02), labels=list(f"{i}%" for i in range(90, 101, 2)))

    # draw baseline on x1, similarity
    bar1 = ax.bar(x1, baseline_df['similarity'], width, label="Baseline")
    # draw default on x, similarity
    bar2 = ax.bar(x, default_df['similarity'], width, label=f"{PJNAME}")

    # add acc_loss on top of the bar of PJNAME
    # for i, task_name in enumerate(default_df['task_name']):
    #     similarity = default_df[default_df["task_name"] == task_name]["similarity"].values[0]
    #     ax.text(i, similarity + 0.01, "{:.2f}%".format(similarity*100), ha="center")
    for i, (rect, task_name) in enumerate(zip(bar2, default_df["task_name"])):
        height = rect.get_height()
        similarity = default_df[default_df["task_name"] == task_name]["similarity"].values[0]
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{similarity*100:.2f}%', ha='center', va='bottom', size=10)

    # ax.set_xlabel("Task Name")
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
    sns.set_style("whitegrid", {'axes.grid' : False})

    selected_df = get_evals_with_default_settings(df)

    # plot one figure, where
    # x-axis: task_name
    # y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)

    selected_df["sns_AFC"] = selected_df["BD:AFC"]
    selected_df["sns_AMI"] = selected_df["BD:AMI"] + selected_df["sns_AFC"]
    selected_df["sns_Sobol"] = selected_df["BD:Sobol"] + selected_df["sns_AMI"]
    # selected_df["sns_Others"] = selected_df["BD:Others"] + selected_df["sns_Sobol"]

    fig, ax = plt.subplots(figsize=(5, 4))
    # sns.barplot(x="task_name", y="sns_Others", data=selected_df, ax=ax, label="Others")
    # sns.barplot(x="task_name", y="sns_Sobol", data=selected_df, ax=ax, label="Planner", color="tomato")
    # sns.barplot(x="task_name", y="sns_AMI", data=selected_df, ax=ax, label="Executor:AMI", color="royalblue")
    # sns.barplot(x="task_name", y="sns_AFC", data=selected_df, ax=ax, label="Executor:AFC", color="tomato")

    # xticklabels = selected_df['task_name'].values
    xticklabels = PIPELINE_NAME
    x = [i for i in range(len(tasks))]
    ax.set_xticks(ticks=x, labels=xticklabels)
    width = 0.4
    ax.bar(x, selected_df["sns_Sobol"], width, label="Planner")
    ax.bar(x, selected_df["sns_AMI"], width, label="Executor:AMI")
    ax.bar(x, selected_df["sns_AFC"], width, label="Executor:AFC")

    ax.tick_params(axis='x', rotation=10)
    ax.set_xlabel("")
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
    sns.set_style("whitegrid", {'axes.grid' : False})

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(7, 6), nrows=2, ncols=2, sharex=False, sharey=True)
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

        axes[i].scatter(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="tomato")
        plot2 = twnx.plot(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="tomato", label="Accuracy")

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]))
        axes[i].set_xlabel("Confidence Level $\\tau$")
        axes[i].set_ylabel("Speedup", color="royalblue")
        # axes[i].legend(loc="lower left")

        twnx.set_ylim(YLIM_ACC)
        twnx.set_ylabel("Accuracy", color="tomato")
        # twnx.legend(loc="lower left")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_min_conf.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_max_error(df: pd.DataFrame, args: EvalArgs):
    """
    For each task,
    Plot the accuracy and speedup with different max_error.
    """
    sns.set_style("whitegrid", {'axes.grid' : False})

    df = get_evals_basic(df)
    selected_df = []
    for task_name in reg_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
        fig, axes = plt.subplots(figsize=(7, 3), nrows=1, ncols=2, sharex=False, sharey=True)
    elif len(reg_tasks) > 2:
        fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2, sharex=False, sharey=True)
    else:
        raise NotImplementedError
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(reg_tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="tomato")
        plot2 = twnx.plot(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="tomato", label="Accuracy")

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]))
        axes[i].set_xlabel("Error Bound $\delta$")
        axes[i].set_ylabel("Speedup", color="royalblue")
        # axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        twnx.set_ylabel("Accuracy", color="tomato")
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_max_error.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_alpha(df: pd.DataFrame, args: EvalArgs):
    """ alpha = scheduler_init / ncfgs
    """
    sns.set_style("whitegrid", {'axes.grid' : False})

    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
        # df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "alpha", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(7, 6), nrows=2, ncols=2, sharex=False, sharey=True)
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

        axes[i].scatter(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="tomato")
        plot2 = twnx.plot(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="tomato", label="Accuracy")

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]))
        axes[i].set_xlabel("Initial Sampling Ratio $\\alpha$")
        axes[i].set_ylabel("Speedup", color="royalblue")
        # axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        twnx.set_ylabel("Accuracy", color="tomato")
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_alpha.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_beta(df: pd.DataFrame, args: EvalArgs):
    """ beta = scheduler_batch / ncfgs
    """
    sns.set_style("whitegrid", {'axes.grid' : False})

    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        # df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    required_cols = ["task_name", "beta", "speedup", "similarity",
                     "accuracy", "acc_loss", "acc_loss_pct",
                     "avg_latency", "BD:AFC", "BD:AMI", "BD:Sobol", "BD:Others"]
    selected_df = selected_df[required_cols]
    print(selected_df)

    if len(tasks) == 4:
        fig, axes = plt.subplots(figsize=(7, 6), nrows=2, ncols=2, sharex=False, sharey=True)
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

        axes[i].scatter(df_tmp["beta"], df_tmp["speedup"], marker='o', color="royalblue")
        plot1 = axes[i].plot(df_tmp["beta"], df_tmp["speedup"], marker='o', color="royalblue", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="tomato")
        plot2 = twnx.plot(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="tomato", label="Accuracy")

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]))
        axes[i].set_xlabel("Step Size $\gamma$")
        axes[i].set_ylabel("Speedup", color="royalblue")\

        # set xtick labels as (beta, $\sum N_j$)
        axes[i].set_xticks(ticks=df_tmp["beta"])
        # only show the first, the middle, and last xtick labels
        xticklabels = [f"{beta:.3f}$\sum N_j$" for beta in df_tmp["beta"]]
        xticklabels[1:-1] = ["" for _ in range(len(xticklabels[1:-1]))]
        axes[i].set_xticklabels(labels=xticklabels)

        # axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        twnx.set_ylabel("Accuracy", color="tomato")
        # twnx.legend(loc="upper right")

        plots = plot1 + plot2
        labels = [l.get_label() for l in plots]
        axes[i].legend(plots, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_beta.pdf"))
    # plt.show()

    plt.close("all")


def vary_alpha_beta(df: pd.DataFrame, args: EvalArgs):
    df = get_evals_basic(df)

    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
        # df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        # df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
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

        axes[i].set_title("Task: {}".format(PIPELINE_NAME[i]))
        axes[i].set_xlabel("Initial Sampling Ratio")
        axes[i].set_ylabel("Speedup")
        axes[i].legend(loc="upper left")

        twnx.set_ylim(YLIM_ACC)
        twnx.set_ylabel("Accuracy")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_alpha_beta.pdf"))
    # plt.show()

    plt.close()


def vary_num_agg(df: pd.DataFrame, args: EvalArgs):
    sns.set_style("whitegrid", {'axes.grid' : False})

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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue")
    plot1 = ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue", label="Speedup")

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df["similarity"], marker='+', color="tomato")
    plot2 = twnx.plot(selected_df["naggs"], selected_df["similarity"], marker='+', color="tomato", label="Accuracy")

    ax.set_xlabel("Number of Approximated Aggregation Features")
    ax.set_ylabel("Speedup", color="royalblue")
    # ax.legend(loc="upper left")

    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato")
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="lower right")
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
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
    ax.scatter(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue")
    ax.plot(selected_df["naggs"], selected_df["speedup"], marker='o', color="royalblue", label="Speedup")

    twnx = ax.twinx()
    twnx.scatter(selected_df["naggs"], selected_df["similarity"], marker='+', color="tomato")
    twnx.plot(selected_df["naggs"], selected_df["similarity"], marker='+', color="tomato", label="Accuracy")

    ax.set_xlabel("Number of Aggregation Operators")
    ax.set_ylabel("Speedup", color="royalblue")
    ax.legend(loc="upper left")
    
    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato")
    twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_num_nf.pdf"))
    # plt.show()

    plt.close("all")


def vary_datasize(df: pd.DataFrame, args: EvalArgs):
    required_cols = ["task_name", "num_months", "speedup", "similarity",
                     "avg_latency", "accuracy"]
    selected_tasks = [f'tickvaryNM{i}' for i in range(1, 8)] # + ['Tick-Price']
    selected_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Tick-Price"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Tick-Price"]["max_error"]]
        df_tmp["num_months"] = int(task_name.replace("tickvaryNM", ""))
        df_tmp = df_tmp.sort_values(by=["task_name"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)
    selected_df = selected_df.sort_values(by=["num_months"])

    # get baseline df
    baseline_df = []
    for task_name in selected_tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["policy"] == shared_default_settings["policy"]]
        df_tmp = df_tmp[df_tmp["ncores"] == shared_default_settings["ncores"]]
        df_tmp = df_tmp[df_tmp["nparts"] == shared_default_settings["nparts"]]
        df_tmp = df_tmp[df_tmp["ncfgs"] == shared_default_settings["ncfgs"]]
        df_tmp = df_tmp[df_tmp["pest_nsamples"] == shared_default_settings["pest_nsamples"]]
        df_tmp = df_tmp[df_tmp["model_name"] == task_default_settings["Tick-Price"]["model_name"]]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == 1.0]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings["Tick-Price"]["max_error"]]
        df_tmp["num_months"] = int(task_name.replace("tickvaryNM", ""))
        df_tmp = df_tmp.sort_values(by=["task_name"])
        df_tmp = df_tmp.reset_index(drop=True)
        baseline_df.append(df_tmp)
    baseline_df = pd.concat(baseline_df)
    baseline_df = baseline_df.sort_values(by=["num_months"])

    baseline_df = baseline_df[required_cols]
    selected_df = selected_df[required_cols]

    # with original tick_100, the latency is 0.620491 => 0.063319

    print(baseline_df)
    print(selected_df)

    # plot as a scatter line chart
    # x-axis: num_months
    # y-axis: speedup and similarity
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(selected_df["num_months"], selected_df["speedup"], marker='o', color="royalblue")
    plot1 = ax.plot(selected_df["num_months"], selected_df["speedup"], marker='o', color="royalblue", label="Speedup")

    twnx = ax.twinx()
    twnx.scatter(selected_df["num_months"], selected_df["similarity"], marker='+', color="tomato")
    plot2 = twnx.plot(selected_df["num_months"], selected_df["similarity"], marker='+', color="tomato", label="Accuracy")

    ax.set_xlabel("Number of Months")
    ax.set_ylabel("Speedup", color="royalblue")
    # ax.legend(loc="upper left")

    # set range for y-axis
    twnx.set_ylim(YLIM_ACC)
    twnx.set_ylabel("Accuracy", color="tomato")
    # twnx.legend(loc="upper right")

    plots = plot1 + plot2
    labels = [l.get_label() for l in plots]
    ax.legend(plots, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, "plots", "sim-sup_vary_num_nm.pdf"))
    # plt.show()

    plt.close("all")


def main(args: EvalArgs):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.size"] = 40
    df = load_df(args)
    plot_lat_comparsion_w_breakdown(df, args)
    plot_lat_breakdown(df, args)
    plot_vary_min_conf(df, args)
    plot_vary_max_error(df, args)
    plot_vary_alpha(df, args)
    plot_vary_beta(df, args)
    vary_num_agg(df, args)
    vary_datasize(df, args)

    # print(get_evals_with_default_settings(df))


if __name__ == "__main__":
    args = EvalArgs().parse_args()
    shared_default_settings["ncores"] = args.ncores
    main(args)
