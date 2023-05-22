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


def plotting_1(settings: list, evals_list: list, setting_name: str):
    plotting_dir = os.path.join(PLOTTING_HOME, setting_name)
    os.makedirs(plotting_dir, exist_ok=True)

    all_time_df = pd.concat([evals["time_eval"] for evals in evals_list], axis=0)
    # add settings to the first column
    all_time_df.insert(0, setting_name, settings)
    all_time_df.to_csv(
        os.path.join(plotting_dir, f"{setting_name}_time.csv"), index=False
    )
    print(all_time_df)

    # all columns in all_time_df now:
    # setting_name,exact_pred_time,total_feature_loading_nrows,total_feature_loading_frac,total_feature_loading_time,total_feature_time,total_feature_estimation_time,total_prediction_estimation_time,total_feature_influence_time

    # plot setting_name-total_feature_loading_frac, setting_name-total_feature_time, setting_name-total_feature_estimation_time, setting_name-total_prediction_estimation_time, setting_name-total_feature_influence_time
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    for i, col in enumerate(
        [
            "total_feature_loading_frac",
            "total_feature_time",
            "total_feature_estimation_time",
            "total_prediction_estimation_time",
            "total_feature_influence_time",
        ]
    ):
        ax = axes[i]
        all_time_df.plot(x=setting_name, y=col, kind="scatter", ax=ax)
        all_time_df.plot(x=setting_name, y=col, kind="line", ax=ax)
        ax.set_title(col)
    fig.savefig(os.path.join(plotting_dir, f"{setting_name}_time.png"))

    all_fevals_df = pd.concat(
        [evals["feat_eval"].iloc[-1:] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_fevals_df.insert(0, setting_name, settings)
    all_fevals_df.to_csv(
        os.path.join(plotting_dir, f"{setting_name}_fevals.csv"), index=False
    )
    print(all_fevals_df)

    # all columns in all_fevals_df now:
    # setting_name,tag,mse,mae,r2,expv,maxe

    # plot setting_name-mse, setting_name-mae, setting_name-r2, setting_name-expv, setting_name-maxe
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    for i, col in enumerate(["mse", "mae", "r2", "expv", "maxe"]):
        ax = axes[i]
        all_fevals_df.plot(x=setting_name, y=col, kind="scatter", ax=ax)
        all_fevals_df.plot(x=setting_name, y=col, kind="line", ax=ax)
        ax.set_title(col)
    fig.savefig(os.path.join(plotting_dir, f"{setting_name}_fevals.png"))

    all_ppl_sim_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-1:] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_ppl_sim_evals_df.insert(0, setting_name, settings)
    all_ppl_sim_evals_df.to_csv(
        os.path.join(plotting_dir, f"{setting_name}_ppl_sim_evals.csv"), index=False
    )
    print(all_ppl_sim_evals_df)

    # all columns in all_ppl_sim_evals_df now:
    # setting_name,tag,acc,recall,precision,f1,roc,recall_micro,precision_micro,f1_micro,roc_micro,recall_weighted,precision_weighted,f1_weighted,roc_weighted

    # plot setting_name-acc, setting_name-recall, setting_name-precision, setting_name-f1, setting_name-roc
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    for i, col in enumerate(["acc", "recall", "precision", "f1", "roc"]):
        ax = axes[i]
        all_ppl_sim_evals_df.plot(x=setting_name, y=col, kind="scatter", ax=ax)
        all_ppl_sim_evals_df.plot(x=setting_name, y=col, kind="line", ax=ax)
        ax.set_title(col)
    fig.savefig(os.path.join(plotting_dir, f"{setting_name}_ppl_sim_evals.png"))

    all_ppl_acc_evals_df = pd.concat(
        [evals["ppl_eval"].iloc[-2:-1] for evals in evals_list], axis=0
    )
    # add settings to the first column
    all_ppl_acc_evals_df.insert(0, setting_name, settings)
    all_ppl_acc_evals_df.to_csv(
        os.path.join(plotting_dir, f"{setting_name}_ppl_acc_evals.csv"), index=False
    )

    # all columns in all_ppl_acc_evals_df now:
    # setting_name,tag,acc,recall,precision,f1,roc,recall_micro,precision_micro,f1_micro,roc_micro,recall_weighted,precision_weighted,f1_weighted,roc_weighted

    # plot setting_name-acc, setting_name-recall, setting_name-precision, setting_name-f1, setting_name-roc
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))
    axes = axes.flatten()
    for i, col in enumerate(["acc", "recall", "precision", "f1", "roc"]):
        ax = axes[i]
        all_ppl_acc_evals_df.plot(x=setting_name, y=col, kind="scatter", ax=ax)
        all_ppl_acc_evals_df.plot(x=setting_name, y=col, kind="line", ax=ax)
        ax.set_title(col)
    fig.savefig(os.path.join(plotting_dir, f"{setting_name}_ppl_acc_evals.png"))


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


# def collect_1(args: OnlineParser, all_samples):
#     sample_strategy = args.sample_strategy
#     job_dir = args.job_dir

#     # columns: sample, feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
#     all_metrics = pd.DataFrame(
#         columns=[
#             "sample",
#             "fr2",
#             "fmaxe",
#             "pacc",
#             "pacc-sim",
#             "proc",
#             "proc-sim",
#             "pconf",
#             "pconf-median",
#             "feature_time",
#             "pred_time",
#             "pconf_time",
#             "cpu_time",
#             "nrows",
#         ]
#     )
#     for sample in tqdm(all_samples):
#         feature_dir = os.path.join(job_dir, "features", f"sample_{sample}")
#         fevals_path = os.path.join(
#             feature_dir, f"fevals_{sample_strategy}_{args.low_conf_threshold}.csv"
#         )
#         evals_path = os.path.join(
#             feature_dir, f"evals_{sample_strategy}_{args.low_conf_threshold}.csv"
#         )

#         if not (os.path.exists(fevals_path) and os.path.exists(evals_path)):
#             command = f"/home/ckchang/anaconda3/envs/amd/bin/python \
#                     /home/ckchang/ApproxInfer/scripts/machinery/binary_classification/online_test.py \
#                     --task {args.task} \
#                     --model_name {args.model_name} \
#                     --sample_strategy {sample_strategy} \
#                     --sample_budget_each {sample} \
#                     --low_conf_threshold {args.low_conf_threshold} \
#                     --npoints_for_conf {args.npoints_for_conf}"
#             command += " > /dev/null"
#             os.system(command)

#         fevals = pd.read_csv(fevals_path)
#         evals = pd.read_csv(evals_path)

#         feature_r2 = fevals.iloc[-1]["r2"]
#         fmaxe = fevals.iloc[:-1]["maxe"].max()

#         oracle_acc = evals.iloc[0]["acc"]
#         oracle_roc = evals.iloc[0]["roc"]
#         pred_accuracy = evals.iloc[1]["acc"]
#         pconf = evals.iloc[1]["mean_conf"]
#         pconf_median = evals.iloc[1]["median_conf"]
#         pred_similarity = evals.iloc[2]["acc"]
#         feature_time = evals.iloc[2]["feature_time"]
#         pred_time = evals.iloc[2]["pred_time"]
#         pconf_time = evals.iloc[2]["pconf_time"]
#         nrows = evals.iloc[2]["nrows"]
#         row = {
#             "sample": sample,
#             "oracle_acc": oracle_acc,
#             "oracle_roc": oracle_roc,
#             "fr2": feature_r2,
#             "fmaxe": fmaxe,
#             "pacc": pred_accuracy,
#             "pacc-sim": pred_similarity,
#             "proc": evals.iloc[1]["roc"],
#             "proc-sim": evals.iloc[2]["roc"],
#             "pconf": pconf,
#             "pconf-median": pconf_median,
#             "feature_time": feature_time,
#             "pred_time": pred_time,
#             "pconf_time": pconf_time,
#             "cpu_time": feature_time + pred_time + pconf_time,
#             "nrows": nrows,
#         }
#         row_df = pd.DataFrame(row, index=[0])
#         print(row_df)
#         all_metrics = pd.concat([all_metrics, row_df])
#     all_metrics.to_csv(
#         os.path.join(
#             job_dir,
#             f"machinery_metrics_{args.sample_strategy}_{args.low_conf_threshold}.csv",
#         )
#     )


# def plot_1(args: OnlineParser):
#     """
#     Plot the metrics with different sample rates.
#     metrics include: feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
#     """
#     nchunks = args.sample_nchunks
#     job_dir = args.job_dir

#     # columns: sample, feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
#     all_metrics = pd.DataFrame(
#         columns=[
#             "sample",
#             "fr2",
#             "fmaxe",
#             "pacc",
#             "pacc-sim",
#             "proc",
#             "proc-sim",
#             "pconf",
#             "pconf-median",
#             "feature_time",
#             "pred_time",
#             "pconf_time",
#             "cpu_time",
#             "nrows",
#         ]
#     )

#     all_samples = [0.001]
#     all_samples += [i * 0.005 for i in range(1, 20)]
#     all_samples += [i * 1.0 / nchunks for i in range(1, nchunks)]
#     all_samples += [
#         i * 1.0 / (5 * nchunks) for i in range(5 * nchunks - 5 + 1, 5 * nchunks + 1)
#     ]
#     collect_1(args, list(reversed(all_samples)))

#     all_metrics = pd.read_csv(
#         os.path.join(
#             job_dir,
#             f"machinery_metrics_{args.sample_strategy}_{args.low_conf_threshold}.csv",
#         )
#     )
#     oracle_acc = all_metrics.loc[0, "oracle_acc"]
#     oracle_roc = all_metrics.loc[0, "oracle_roc"]

#     # plotting
#     fig, axs = plt.subplots(2, 2, figsize=(16, 12))
#     axs[0, 0].plot(all_metrics["sample"], all_metrics["fr2"], label="feature_r2")
#     axs[0, 0].plot(
#         all_metrics["sample"],
#         [1.0] * len(all_metrics["sample"]),
#         label="oracle",
#         linestyle="--",
#     )

#     axs[0, 1].plot(
#         all_metrics["sample"],
#         [oracle_acc] * len(all_metrics["sample"]),
#         label="oracle_acc",
#         linestyle="--",
#     )
#     axs[0, 1].plot(
#         all_metrics["sample"],
#         [oracle_roc] * len(all_metrics["sample"]),
#         label="oracle_roc",
#         linestyle="--",
#     )
#     axs[0, 1].plot(all_metrics["sample"], all_metrics["pacc"], label="pacc")
#     axs[0, 1].plot(all_metrics["sample"], all_metrics["proc"], label="proc")

#     axs[1, 0].plot(all_metrics["sample"], all_metrics["pconf"], label="pconf")
#     axs[1, 0].plot(
#         all_metrics["sample"], all_metrics["pconf-median"], label="pconf-median"
#     )
#     axs[1, 0].plot(
#         all_metrics["sample"],
#         [1.0] * len(all_metrics["sample"]),
#         label="oracle_conf",
#         linestyle="--",
#     )

#     axs[1, 1].plot(all_metrics["sample"], all_metrics["pacc-sim"], label="pacc-sim")
#     axs[1, 1].plot(all_metrics["sample"], all_metrics["proc-sim"], label="proc-sim")
#     axs[1, 1].plot(
#         all_metrics["sample"],
#         [1.0] * len(all_metrics["sample"]),
#         label="oracle_sim",
#         linestyle="--",
#     )

#     for i in range(2):
#         for j in range(2):
#             axs[i, j].set_xlabel("sample rate")
#             axs[i, j].set_ylabel("metrics")
#             axs[i, j].legend()

#     plt.savefig(os.path.join(job_dir, f"machinery_metrics_{args.sample_strategy}.png"))
#     plt.show()

#     # plot time measurements, cpu_time = feature_time + pred_time + pconf_time
#     fig, axs = plt.subplots(1, 2, figsize=(16, 6))
#     axs[0].plot(
#         all_metrics["sample"], all_metrics["feature_time"], label="feature_time"
#     )
#     axs[0].plot(all_metrics["sample"], all_metrics["pred_time"], label="pred_time")
#     axs[0].plot(all_metrics["sample"], all_metrics["pconf_time"], label="pconf_time")
#     axs[0].set_xlabel("sample rate")
#     axs[0].set_ylabel("time (s)")
#     axs[0].legend()

#     # axs[1].plot(all_metrics['sample'], all_metrics['feature_time'] /
#     #             all_metrics['cpu_time'], label='feature_time')
#     # axs[1].plot(all_metrics['sample'], all_metrics['pred_time'] /
#     #             all_metrics['cpu_time'], label='pred_time')
#     # axs[1].plot(all_metrics['sample'], all_metrics['pconf_time'] /
#     #             all_metrics['cpu_time'], label='pconf_time')
#     axs[1].plot(all_metrics["sample"], all_metrics["nrows"], label="nrows")
#     axs[1].set_xlabel("sample rate")
#     axs[1].set_ylabel("nrows per request")
#     axs[1].legend()

#     plt.savefig(os.path.join(job_dir, f"machinery_cpu_time_{args.sample_strategy}.png"))

#     if args.sample_strategy != "equal":
#         equal_metrics = pd.read_csv(
#             os.path.join(
#                 job_dir, f"machinery_metrics_equal_{args.low_conf_threshold}.csv"
#             )
#         )
#         # equal is the baseline strategy, if not equal, we need to compare with the equal strategy
#         # we want to the trade-off between accuracy and cpu time, on different sample rates
#         # the x-axis is accuracy (acc and roc), the y-axis is cpu time, the color indicates sample rate
#         # we use scatter plot to show the trade-off
#         colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(all_metrics["sample"])))
#         fig, axs = plt.subplots(2, 2, figsize=(16, 12))
#         axs[0][0].scatter(
#             equal_metrics["pacc"],
#             equal_metrics["nrows"],
#             label="equal",
#             marker="x",
#             color=colors,
#         )
#         axs[0][0].scatter(
#             all_metrics["pacc"],
#             all_metrics["nrows"],
#             label=args.sample_strategy,
#             marker="*",
#             color=colors,
#         )
#         axs[0][0].plot(
#             equal_metrics["pacc"], equal_metrics["nrows"], label="equal", linestyle="--"
#         )
#         axs[0][0].plot(
#             all_metrics["pacc"], all_metrics["nrows"], label=args.sample_strategy
#         )
#         axs[0][0].set_xlabel("accuracy")
#         axs[0][0].set_ylabel("cpu time (s)")
#         axs[0][0].legend()

#         axs[0][1].scatter(
#             equal_metrics["pacc-sim"],
#             equal_metrics["nrows"],
#             label="equal",
#             marker="x",
#             color=colors,
#         )
#         axs[0][1].scatter(
#             all_metrics["pacc-sim"],
#             all_metrics["nrows"],
#             label=args.sample_strategy,
#             marker="*",
#             color=colors,
#         )
#         axs[0][1].plot(
#             equal_metrics["pacc-sim"],
#             equal_metrics["nrows"],
#             label="equal",
#             linestyle="--",
#         )
#         axs[0][1].plot(
#             all_metrics["pacc-sim"], all_metrics["nrows"], label=args.sample_strategy
#         )
#         axs[0][1].set_xlabel("acc similarity")
#         axs[0][1].set_ylabel("load cpu time")
#         axs[0][1].legend()

#         axs[1][0].scatter(
#             equal_metrics["proc"],
#             equal_metrics["nrows"],
#             label="equal",
#             marker="x",
#             color=colors,
#         )
#         axs[1][0].scatter(
#             all_metrics["proc"],
#             all_metrics["nrows"],
#             label=args.sample_strategy,
#             marker="*",
#             color=colors,
#         )
#         axs[1][0].plot(
#             equal_metrics["proc"], equal_metrics["nrows"], label="equal", linestyle="--"
#         )
#         axs[1][0].plot(
#             all_metrics["proc"], all_metrics["nrows"], label=args.sample_strategy
#         )
#         axs[1][0].set_xlabel("roc")
#         axs[1][0].set_ylabel("nrows")
#         axs[1][0].legend()

#         axs[1][1].scatter(
#             equal_metrics["proc-sim"],
#             equal_metrics["nrows"],
#             label="equal",
#             marker="x",
#             color=colors,
#         )
#         axs[1][1].scatter(
#             all_metrics["proc-sim"],
#             all_metrics["nrows"],
#             label=args.sample_strategy,
#             marker="*",
#             color=colors,
#         )
#         axs[1][1].plot(
#             equal_metrics["proc-sim"],
#             equal_metrics["nrows"],
#             label="equal",
#             linestyle="--",
#         )
#         axs[1][1].plot(
#             all_metrics["proc-sim"], all_metrics["nrows"], label=args.sample_strategy
#         )
#         axs[1][1].set_xlabel("roc similarity")
#         axs[1][1].set_ylabel("nrows")
#         axs[1][1].legend()

#         plt.savefig(
#             os.path.join(
#                 job_dir, f"machinery_cpu_time_{args.sample_strategy}_vs_equal.png"
#             )
#         )


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    print(f"running plotting.py with {args}")
    collector = Collector(args)
    collector.vary_sample_refine_max_niters()

    # use max_niters = 5 for following experiments
    collector.args.sample_refine_max_niters = 5
    collector.args.process_args()

    collector.vary_init_budget()
    collector.vary_sample_budget()

    collector.vary_prediction_estimation_nsamples()
    collector.vary_feature_influence_estimation_nsamples()
