import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math


HOME_DIR = "./cache"


def load_df(csv_dir: str = HOME_DIR) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(csv_dir, "evals.csv"))
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

    # df = handler_soboltime(df)
    df = handler_filter_ncfgs(df)
    return df


tasks = [
    "trips",
    "tick-v1",
    "tick-v2",
    "cheaptrips",
    "machinery-v1",
    "machinery-v2",
    # "machinery-v3",
]

shared_default_settings = {
    "policy": "optimizer",
    "ncores": 1,
    "min_conf": 0.9,
    "nparts": 100,
    "ncfgs": 100,
    "scheduler_init": 1,
    "scheduler_batch": 1
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
    "tick-v2": {
        "model_name": "lr",
        "max_error": 0.01,
    },
    "cheaptrips": {
        "model_name": "xgb",
        "max_error": 0.0,
    },
    "machinery-v1": {
        "model_name": "mlp",
        "max_error": 0.0,
    },
    "machinery-v2": {
        "model_name": "dt",
        "max_error": 0.0,
    },
    "machinery-v3": {
        "model_name": "knn",
        "max_error": 0.0,
    },
}


def get_evals_basic(df: pd.DataFrame) -> pd.DataFrame:
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


def get_evals_with_default_settings(df: pd.DataFrame) -> pd.DataFrame:
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
    print(selected_df)
    return selected_df


def get_exact_evals_with_default_settings(df: pd.DataFrame) -> pd.DataFrame:
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
    print(selected_df)
    return selected_df


def plot_default(df: pd.DataFrame):
    """
    For every task, plot the accuracy and speedup with default settings.
    """
    selected_df = get_evals_with_default_settings(df)

    # plot one figure, where
    # x-axis: task_name
    # y-axis: speedup (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="task_name", y="speedup", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    # for each bar, add text showing the speedup and acc_loss_pct
    for i, task_name in enumerate(tasks):
        speedup = selected_df[selected_df["task_name"] == task_name]["speedup"].values[0]
        ax.text(i, speedup + 0.05, "{:.2f}".format(speedup), ha="center")
        # acc_loss_pct = selected_df[selected_df["task_name"] == task_name]["acc_loss_pct"].values[0]
        # ax.text(i, speedup + 0.05, "{:.2f}, {:.4f}".format(speedup, acc_loss_pct), ha="center")
    ax.set_xlabel("Task Name")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup with Default Settings")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_default.pdf"))
    # plt.show()

    plt.close("all")


def plot_lat_breakdown(df: pd.DataFrame):
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
    selected_df["sns_Others"] = selected_df["BD:Others"] + selected_df["sns_Sobol"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="task_name", y="sns_Others", data=selected_df, ax=ax, label="Others")
    sns.barplot(x="task_name", y="sns_Sobol", data=selected_df, ax=ax, label="Planner")
    sns.barplot(x="task_name", y="sns_AMI", data=selected_df, ax=ax, label="Executor:AMI")
    sns.barplot(x="task_name", y="sns_AFC", data=selected_df, ax=ax, label="Executor:AFC")
    ax.set_xlabel("Task Name")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency Breakdown with Default Settings")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "lat_breakdown_default.pdf"))
    # plt.show()

    exact_df = get_exact_evals_with_default_settings(df)
    exact_df["sns_AFC"] = exact_df["BD:AFC"]
    exact_df["sns_AMI"] = exact_df["BD:AMI"] + exact_df["sns_AFC"]
    exact_df["sns_Sobol"] = exact_df["BD:Sobol"] + exact_df["sns_AMI"]
    exact_df["sns_Others"] = exact_df["BD:Others"] + exact_df["sns_Sobol"]

    # plot one figure with two subplots, where
    # left subplot for exact_df:
    #   x-axis: task_name
    #   y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)
    # right subplot for selected_df:
    #   x-axis: task_name
    #   y-axis: BD:AFC, BD:AMI, BD:Sobol, BD:Others (stacked)

    fig, axes = plt.subplots(figsize=(15, 8), nrows=1, ncols=2, sharex=True, sharey=True)
    sns.barplot(x="task_name", y="sns_Others", data=exact_df, ax=axes[0], label="Others")
    sns.barplot(x="task_name", y="sns_Sobol", data=exact_df, ax=axes[0], label="Planner")
    sns.barplot(x="task_name", y="sns_AMI", data=exact_df, ax=axes[0], label="Executor:AMI")
    sns.barplot(x="task_name", y="sns_AFC", data=exact_df, ax=axes[0], label="Executor:AFC")
    axes[0].set_xlabel("Task Name")
    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title("Latency Breakdown of Baseline")
    axes[0].legend()
    sns.barplot(x="task_name", y="sns_Others", data=selected_df, ax=axes[1], label="Others")
    sns.barplot(x="task_name", y="sns_Sobol", data=selected_df, ax=axes[1], label="Planner")
    sns.barplot(x="task_name", y="sns_AMI", data=selected_df, ax=axes[1], label="Executor:AMI")
    sns.barplot(x="task_name", y="sns_AFC", data=selected_df, ax=axes[1], label="Executor:AFC")
    axes[1].set_xlabel("Task Name")
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_title("Latency Breakdown of Marksman")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "lat_breakdown_default_w_exact.pdf"))
    # plt.show()

    plt.close("all")


def plot_lat_comparsion(df: pd.DataFrame):
    """
    Compare Marksman(ours) with other systems.
    Baseline A: single-core, no sampling
    Baseline B: single-core, with sampling but min_conf=0.5
    """
    selected_df = get_evals_with_default_settings(df)

    # plot two sub figures, where
    # x-axis: task_name
    # y-axis-1: avg_latency (bar)
    # y-axis-2: accuracy (bar)
    # three groups of bars: Marksman, Baseline

    df = selected_df[['task_name', 'avg_latency', 'speedup',
                      'accuracy', 'acc_loss', 'acc_loss_pct']]
    df['avg_latency_baselineA'] = df['avg_latency'] * df['speedup']
    df['accuracy_baselineA'] = df['accuracy'] + df['acc_loss']
    print(df)

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(figsize=(10, 8), nrows=2, ncols=1, sharex=True, sharey=False)

    def plot_lat(ax, df):
        width = 0.35
        x = [i for i in range(len(df))]
        x1 = [i - width for i in x]

        ax.set_xticks(x)
        ax.set_xticklabels(tasks)

        ax.bar(x1, df['avg_latency_baselineA'], width, label="Baseline")
        ax.bar(x, df['avg_latency'], width, label="Marksman")

        # add speedup on top of the bar of Marksman
        for i, task_name in enumerate(df['task_name']):
            lat = df[df["task_name"] == task_name]["avg_latency"].values[0]
            speedup = df[df["task_name"] == task_name]["speedup"].values[0]
            ax.text(i, lat + 0.01, "{:.2f}x".format(speedup), ha="center")

        ax.set_xlabel("Task Name")
        ax.set_ylabel("Latency (s)")
        ax.set_title("Latency Comparison with Default Settings")
        ax.legend()

    def plot_acc(ax, df):
        width = 0.35
        x = [i for i in range(len(df))]
        x1 = [i - width for i in x]

        ax.set_xticks(x)
        ax.set_xticklabels(tasks)

        ax.bar(x1, df['accuracy_baselineA'], width, label="Baseline")
        ax.bar(x, df['accuracy'], width, label="Marksman")

        # add acc_loss on top of the bar of Marksman
        for i, task_name in enumerate(df['task_name']):
            acc = df[df["task_name"] == task_name]["accuracy"].values[0]
            acc_loss = df[df["task_name"] == task_name]["acc_loss"].values[0]
            acc_increased = acc_loss < 0
            ax.text(i, acc + 0.01, ("+" if acc_increased else "") + "{:.4f}".format(-acc_loss), ha="center")

        ax.set_xlabel("Task Name")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy Comparison with Default Settings")
        # ax.legend()

    plot_lat(axes[0], df)
    plot_acc(axes[1], df)

    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "lat_comparison_default.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_min_conf(df: pd.DataFrame):
    """
    For each task,
    Plot the accuracy and speedup with different min_conf.
    """
    df = get_evals_basic(df)
    selected_df = []
    for task_name in tasks:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        # df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)    
    print(selected_df)

    # plot one figure, where
    # x-axis: min_conf
    # y-axis: speedup (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="min_conf", y="speedup", hue="task_name", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Min Confidence")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup with Different Min Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_min_conf.pdf"))
    # plt.show()

    # plot one figure, where
    # x-axis: min_conf
    # y-axis: accuracy (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="min_conf", y="accuracy", hue="task_name", data=selected_df, ax=ax)
    ax.set_xlabel("Min Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy with Different Min Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "accuracy_vary_min_conf.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: min_conf
    # y-axis-1: speedup (bar)
    # y-axis-2: accuracy (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="min_conf", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Min Confidence")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Min Confidence")
    sns.barplot(x="min_conf", y="accuracy", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Min Confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy with Different Min Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_acc_vary_min_conf.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: min_conf
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_loss (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="min_conf", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Min Confidence")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Min Confidence")
    sns.barplot(x="min_conf", y="acc_loss", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Min Confidence")
    axes[1].set_ylabel("Accuracy Loss")
    axes[1].set_title("Accuracy Loss with Different Min Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accloss_vary_min_conf.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: min_conf
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_diff = abs(acc_loss) (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="min_conf", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Min Confidence")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Min Confidence")
    print(selected_df)
    sns.barplot(x="min_conf", y="acc_diff", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Min Confidence")
    axes[1].set_ylabel("Accuracy Discrepancy")
    axes[1].set_title("Accuracy Discrepancy with Different Min Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accdis_vary_min_conf.pdf"))
    # plt.show()

    # 2x3 subplots, each subplot for one task
    # each plot has two scattered lines: speedup and accuracy
    # x-axis: min_conf
    # y-axis: speedup (scatter + line)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="min_conf", y="speedup", data=df_tmp, ax=axes[i])
        sns.lineplot(x="min_conf", y="speedup", data=df_tmp, ax=axes[i])

        # sns.scatterplot(x="min_conf", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")
        # sns.lineplot(x="min_conf", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")

        axes[i].set_xlabel("Min Confidence")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_min_conf.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    acc_metric = "acc_diff"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Min Confidence")
        axes[i].set_ylabel("Accuracy Discrepancy")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "acc_vary_min_conf.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="min_conf", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Min Confidence")
        axes[i].set_ylabel("Similarity")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim_vary_min_conf.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["min_conf"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["min_conf"], df_tmp[acc_metric], marker='+', color="blue", label="Similarity")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Min Confidence")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Similarity", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim-sup_vary_min_conf.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_max_error(df: pd.DataFrame):
    """
    For each task,
    Plot the accuracy and speedup with different max_error.
    """
    df = get_evals_basic(df)
    selected_df = []
    for task_name in ['tick-v1', 'tick-v2', 'trips']:
        df_tmp = df[df["task_name"] == task_name]
        df_tmp = df_tmp[df_tmp["scheduler_init"] == shared_default_settings["scheduler_init"]]
        df_tmp = df_tmp[df_tmp["scheduler_batch"] == shared_default_settings["scheduler_batch"]]
        df_tmp = df_tmp[df_tmp["min_conf"] == shared_default_settings["min_conf"]]
        # df_tmp = df_tmp[df_tmp["max_error"] == task_default_settings[task_name]["max_error"]]
        df_tmp = df_tmp.sort_values(by=["sampling_rate"])
        df_tmp = df_tmp.reset_index(drop=True)
        selected_df.append(df_tmp)
    selected_df = pd.concat(selected_df)    
    print(selected_df)

    # remove the rows with max_error = 0.05
    selected_df = selected_df[selected_df["max_error"] != 0.05]

    # plot one figure, where
    # x-axis: max_error
    # y-axis: speedup (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="max_error", y="speedup", hue="task_name", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Max Error")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup with Different Max Error")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_max_error.pdf"))
    # plt.show()

    # plot one figure, where
    # x-axis: max_error
    # y-axis: accuracy (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="max_error", y="accuracy", hue="task_name", data=selected_df, ax=ax)
    ax.set_xlabel("Max Error")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy with Different Max Error")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "accuracy_vary_max_error.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: min_conf
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_loss (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="max_error", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Max Error")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Max Error")
    sns.barplot(x="max_error", y="acc_loss", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Max Error")
    axes[1].set_ylabel("Accuracy Loss")
    axes[1].set_title("Accuracy Loss with Different Max Error")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accloss_vary_max_error.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: min_conf
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_diff = abs(acc_loss) (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="max_error", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Max Error")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Max Error")
    sns.barplot(x="max_error", y="acc_diff", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Max Error")
    axes[1].set_ylabel("Accuracy Discrepancy")
    axes[1].set_title("Accuracy Discrepancy with Different Max Error")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accdis_vary_max_error.pdf"))
    # plt.show()

    # 2x3 subplots, each subplot for one task
    # each plot has two scattered lines: speedup and accuracy
    # x-axis: max_error
    # y-axis: speedup (scatter + line)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="max_error", y="speedup", data=df_tmp, ax=axes[i])
        sns.lineplot(x="max_error", y="speedup", data=df_tmp, ax=axes[i])

        # sns.scatterplot(x="max_error", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")
        # sns.lineplot(x="max_error", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")

        axes[i].set_xlabel("Max Error")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_max_error.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    acc_metric = "acc_diff"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Max Error")
        axes[i].set_ylabel("Accuracy Discrepancy")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "acc_vary_max_error.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=True)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="max_error", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Max Error")
        axes[i].set_ylabel("Similarity")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim_vary_max_error.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["max_error"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["max_error"], df_tmp[acc_metric], marker='+', color="blue", label="Similarity")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Max Error")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Similarity", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim-sup_vary_max_error.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_alpha(df: pd.DataFrame):
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
    print(selected_df)

    # plot one figure, where
    # x-axis: alpha
    # y-axis: speedup (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="alpha", y="speedup", hue="task_name", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup with Different Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_alpha.pdf"))
    # plt.show()

    # plot one figure, where
    # x-axis: alpha
    # y-axis: accuracy (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="alpha", y="accuracy", hue="task_name", data=selected_df, ax=ax)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy with Different Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "accuracy_vary_alpha.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: alpha
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_loss (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="alpha", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Alpha")
    sns.barplot(x="alpha", y="acc_loss", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Accuracy Loss")
    axes[1].set_title("Accuracy Loss with Different Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accloss_vary_alpha.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: alpha
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_diff = abs(acc_loss) (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="alpha", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Initial Sampling Percentage")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Initial Sampling Percentage")
    sns.barplot(x="alpha", y="acc_diff", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Initial Sampling Percentage")
    axes[1].set_ylabel("Accuracy Discrepancy")
    axes[1].set_title("Accuracy Discrepancy with Different Initial Sampling Percentages")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accdis_vary_alpha.pdf"))
    # plt.show()

    # 2x3 subplots, each subplot for one task
    # each plot has two scattered lines: speedup and accuracy
    # x-axis: alpha
    # y-axis: speedup (scatter + line)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="alpha", y="speedup", data=df_tmp, ax=axes[i])
        sns.lineplot(x="alpha", y="speedup", data=df_tmp, ax=axes[i])

        # sns.scatterplot(x="alpha", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")
        # sns.lineplot(x="alpha", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")

        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_alpha.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    acc_metric = "acc_diff"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Accuracy Discrepancy")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "acc_vary_alpha.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="alpha", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Similarity")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim_vary_alpha.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["alpha"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["alpha"], df_tmp[acc_metric], marker='+', color="blue", label="Similarity")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Initial Sampling Percentage")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Similarity", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim-sup_vary_alpha.pdf"))
    # plt.show()

    plt.close("all")


def plot_vary_beta(df: pd.DataFrame):
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
    print(selected_df)

    # plot one figure, where
    # x-axis: beta
    # y-axis: speedup (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="beta", y="speedup", hue="task_name", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup with Different Beta")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_beta.pdf"))
    # plt.show()

    # plot one figure, where
    # x-axis: beta
    # y-axis: accuracy (bar)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.barplot(x="beta", y="accuracy", hue="task_name", data=selected_df, ax=ax)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy with Different Beta")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "accuracy_vary_beta.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: beta
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_loss (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="beta", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Alpha")
    sns.barplot(x="beta", y="acc_loss", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Accuracy Loss")
    axes[1].set_title("Accuracy Loss with Different Alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accloss_vary_beta.pdf"))
    # plt.show()

    # plot two figures as subplots, where
    # x-axis: beta
    # y-axis-1: speedup (bar)
    # y-axis-2: acc_diff = abs(acc_loss) (bar)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1, sharex=False, sharey=False)
    sns.barplot(x="beta", y="speedup", hue="task_name", data=selected_df, ax=axes[0])
    axes[0].set_xlabel("Beta")
    axes[0].set_ylabel("Speedup")
    axes[0].set_title("Speedup with Different Beta")
    sns.barplot(x="beta", y="acc_diff", hue="task_name", data=selected_df, ax=axes[1])
    axes[1].set_xlabel("Beta")
    axes[1].set_ylabel("Accuracy Discrepancy")
    axes[1].set_title("Accuracy Discrepancy with Different Beta")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_accdis_vary_beta.pdf"))
    # plt.show()

    # 2x3 subplots, each subplot for one task
    # each plot has two scattered lines: speedup and accuracy
    # x-axis: beta
    # y-axis: speedup (scatter + line)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="beta", y="speedup", data=df_tmp, ax=axes[i])
        sns.lineplot(x="beta", y="speedup", data=df_tmp, ax=axes[i])

        # sns.scatterplot(x="beta", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")
        # sns.lineplot(x="beta", y="acc_diff", data=df_tmp, ax=axes[i].twinx(), color="orange")

        axes[i].set_xlabel("Beta")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vary_beta.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    acc_metric = "acc_diff"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Beta")
        axes[i].set_ylabel("Accuracy Discrepancy")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "acc_vary_beta.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=True, sharey=True)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        sns.scatterplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])
        sns.lineplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])
        # sns.barplot(x="beta", y=acc_metric, data=df_tmp, ax=axes[i])

        axes[i].set_xlabel("Beta")
        axes[i].set_ylabel("Similarity")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim_vary_beta.pdf"))
    # plt.show()

    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    acc_metric = "similarity"
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]

        axes[i].scatter(df_tmp["beta"], df_tmp["speedup"], marker='o', color="orange")
        axes[i].plot(df_tmp["beta"], df_tmp["speedup"], marker='o', color="orange", label="Speedup")

        twnx = axes[i].twinx()
        twnx.scatter(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="blue")
        twnx.plot(df_tmp["beta"], df_tmp[acc_metric], marker='+', color="blue", label="Similarity")

        axes[i].set_title("Task: {}".format(task_name))
        axes[i].set_xlabel("Beta")
        axes[i].set_ylabel("Speedup", color="orange")
        axes[i].legend(loc="upper left")

        twnx.set_ylabel("Similarity", color="blue")
        twnx.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "sim-sup_vary_beta.pdf"))
    # plt.show()

    plt.close("all")


def plot_speeup_vs_accuracy(df: pd.DataFrame):
    """
    For each task,
    Plot the speedup vs. accuracy with different min_conf.
    """
    selected_df = get_evals_basic(df)

    # plot one figure, where
    # x-axis: accuracy
    # y-axis: speedup (scatter)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x="accuracy", y="speedup", hue="task_name", size="scheduler_init", style="scheduler_batch", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs. Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_accuracy.pdf"))

    # plot one figure, where
    # x-axis: similarity
    # y-axis: speedup (scatter)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x="similarity", y="speedup", hue="task_name", size="scheduler_init", style="scheduler_batch", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs. Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_similarity.pdf"))

    # plot one figure, where
    # x-axis: acc_diff
    # y-axis: speedup (scatter)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x="acc_diff", y="speedup", hue="task_name", size="scheduler_init", style="scheduler_batch", data=selected_df, ax=ax)
    # draw a horizontal line at y=1
    ax.axhline(y=1, color="r", linestyle="--")
    ax.set_xlabel("Accuracy Discrepancy")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs. Accuracy Discrepancy")
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_accdis.pdf"))

    plt.close("all")


def vary_alpha_beta(df: pd.DataFrame):
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
    print(selected_df)

    # plot 2x3 figures, each figure for one task
    # x-axis: accuracy
    # y-axis: speedup (scatter)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        sns.scatterplot(x="accuracy", y="speedup", hue="alpha", style="beta", data=df_tmp, ax=axes[i])
        # draw a horizontal line at y=1
        axes[i].axhline(y=1, color="r", linestyle="--")
        axes[i].set_xlabel("Accuracy")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_accuracy_6.pdf"))

    # plot 2x3 figures, each figure for one task
    # x-axis: similarity
    # y-axis: speedup (scatter)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        sns.scatterplot(x="similarity", y="speedup", hue="alpha", style="beta", data=df_tmp, ax=axes[i])
        # draw a horizontal line at y=1
        axes[i].axhline(y=1, color="r", linestyle="--")
        axes[i].set_xlabel("Similarity")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_similarity_6.pdf"))

    # plot 2x3 figures, each figure for one task
    # x-axis: acc_diff
    # y-axis: speedup (scatter)

    fig, axes = plt.subplots(figsize=(12, 8), nrows=2, ncols=3, sharex=False, sharey=False)
    axes = axes.flatten()
    for i, task_name in enumerate(tasks):
        df_tmp = selected_df[selected_df["task_name"] == task_name]
        sns.scatterplot(x="acc_diff", y="speedup", hue="alpha", style="beta", data=df_tmp, ax=axes[i])
        # draw a horizontal line at y=1
        axes[i].axhline(y=1, color="r", linestyle="--")
        axes[i].set_xlabel("Accuracy Discrepancy")
        axes[i].set_ylabel("Speedup")
        axes[i].set_title("Task: {}".format(task_name))
    plt.tight_layout()
    plt.savefig(os.path.join(HOME_DIR, "plots", "speedup_vs_accdis_6.pdf"))

    plt.close("all")


def main():
    df = load_df()
    plot_default(df)
    plot_lat_breakdown(df)
    plot_lat_comparsion(df)
    plot_vary_min_conf(df)
    plot_vary_max_error(df)
    plot_vary_alpha(df)
    plot_vary_beta(df)
    plot_speeup_vs_accuracy(df)
    vary_alpha_beta(df)

    print(get_evals_with_default_settings(df))


if __name__ == "__main__":
    main()
