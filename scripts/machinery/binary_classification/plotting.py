import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tap import Tap
from typing import Literal
# import seaborn as sns

DATA_HOME = "/home/ckchang/ApproxInfer/data"
RESULTS_HOME = "/home/ckchang/ApproxInfer/results"


class OnlineParser(Tap):
    database = 'machinery_more'
    segment_size = 50000

    # path to the task directory
    task = "binary_classification"

    model_name: str = 'xgb'  # model name
    model_type: Literal['regressor', 'classifier'] = 'classifier'  # model type
    multi_class: bool = False  # multi class classification

    max_sample_budget: float = 1.0  # max sample budget each in avg
    init_sample_budget: float = 0.01  # initial sample budget each in avg
    sample_budget: float = 0.1  # sample budget each in avg

    init_sample_policy: str = 'equal'  # initial sample policy

    prediction_estimator: Literal['joint_distribution', 'feature_distribution'] = 'joint_distribution'  # prediction estimator
    prediction_estimator_thresh: float = 0.0  # prediction estimator threshold
    prediction_estimation_nsamples: int = 1000  # number of points for prediction estimation
    
    sample_step_policy: Literal['one_step', 'ten_steps_equal', 'exponential_increase', 'auto'] = 'one_step'  # policy to increase sample budget
    sample_allocation_policy: Literal['equal', 'fimp', 'finf', 'auto'] = 'equal'  # sample allocation policy

    clear_cache: bool = False  # clear cache

    sample_nchunks: int = 10  # number of chunks to sample

    def process_args(self) -> None:
        self.job_dir: str = os.path.join(
            RESULTS_HOME, self.database, f'{self.task}_{self.model_name}')


def collect_1(args: OnlineParser, all_samples):
    sample_strategy = args.sample_strategy
    job_dir = args.job_dir

    # columns: sample, feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
    all_metrics = pd.DataFrame(columns=['sample', 'fr2', 'fmaxe',
                                        'pacc', 'pacc-sim', 'proc',
                                        'proc-sim', 'pconf', 'pconf-median',
                                        'feature_time', 'pred_time', 'pconf_time',
                                        'cpu_time', 'nrows'])
    for sample in tqdm(all_samples):
        feature_dir = os.path.join(job_dir, 'features', f'sample_{sample}')
        fevals_path = os.path.join(
            feature_dir, f'fevals_{sample_strategy}_{args.low_conf_threshold}.csv')
        evals_path = os.path.join(
            feature_dir, f'evals_{sample_strategy}_{args.low_conf_threshold}.csv')

        if not (os.path.exists(fevals_path) and os.path.exists(evals_path)):
            command = f'/home/ckchang/anaconda3/envs/amd/bin/python \
                    /home/ckchang/ApproxInfer/scripts/machinery/binary_classification/online_test.py \
                    --task {args.task} \
                    --model_name {args.model_name} \
                    --sample_strategy {sample_strategy} \
                    --sample_budget_each {sample} \
                    --low_conf_threshold {args.low_conf_threshold} \
                    --npoints_for_conf {args.npoints_for_conf}'
            command += ' > /dev/null'
            os.system(command)

        fevals = pd.read_csv(fevals_path)
        evals = pd.read_csv(evals_path)

        feature_r2 = fevals.iloc[-1]['r2']
        fmaxe = fevals.iloc[:-1]['maxe'].max()

        oracle_acc = evals.iloc[0]['acc']
        oracle_roc = evals.iloc[0]['roc']
        pred_accuracy = evals.iloc[1]['acc']
        pconf = evals.iloc[1]['mean_conf']
        pconf_median = evals.iloc[1]['median_conf']
        pred_similarity = evals.iloc[2]['acc']
        feature_time = evals.iloc[2]['feature_time']
        pred_time = evals.iloc[2]['pred_time']
        pconf_time = evals.iloc[2]['pconf_time']
        nrows = evals.iloc[2]['nrows']
        row = {'sample': sample, 'oracle_acc': oracle_acc, 'oracle_roc': oracle_roc,
               'fr2': feature_r2, 'fmaxe': fmaxe,
               'pacc': pred_accuracy, 'pacc-sim': pred_similarity,
               'proc': evals.iloc[1]['roc'], 'proc-sim': evals.iloc[2]['roc'],
               'pconf': pconf, 'pconf-median': pconf_median,
               'feature_time': feature_time,
               'pred_time': pred_time,
               'pconf_time': pconf_time,
               'cpu_time': feature_time + pred_time + pconf_time,
               'nrows': nrows}
        row_df = pd.DataFrame(row, index=[0])
        print(row_df)
        all_metrics = pd.concat([all_metrics, row_df])
    all_metrics.to_csv(os.path.join(
        job_dir, f'machinery_metrics_{args.sample_strategy}_{args.low_conf_threshold}.csv'))


def plot_1(args: OnlineParser):
    """
    Plot the metrics with different sample rates.
    metrics include: feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
    """
    nchunks = args.sample_nchunks
    job_dir = args.job_dir

    # columns: sample, feature_r2, prediction_accuracy, feature_time, pred_time, pconf_time
    all_metrics = pd.DataFrame(columns=['sample', 'fr2', 'fmaxe',
                                        'pacc', 'pacc-sim',
                                        'proc', 'proc-sim',
                                        'pconf', 'pconf-median',
                                        'feature_time',
                                        'pred_time',
                                        'pconf_time',
                                        'cpu_time', 'nrows'])

    all_samples = [0.001]
    all_samples += [i * 0.005 for i in range(1, 20)]
    all_samples += [i * 1.0 / nchunks for i in range(1, nchunks)]
    all_samples += [i * 1.0 / (5 * nchunks)
                    for i in range(5*nchunks - 5 + 1, 5*nchunks + 1)]
    collect_1(args, list(reversed(all_samples)))

    all_metrics = pd.read_csv(os.path.join(
        job_dir, f'machinery_metrics_{args.sample_strategy}_{args.low_conf_threshold}.csv'))
    oracle_acc = all_metrics.loc[0, 'oracle_acc']
    oracle_roc = all_metrics.loc[0, 'oracle_roc']

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].plot(all_metrics['sample'],
                   all_metrics['fr2'], label='feature_r2')
    axs[0, 0].plot(all_metrics['sample'], [1.0] *
                   len(all_metrics['sample']), label='oracle', linestyle='--')

    axs[0, 1].plot(all_metrics['sample'], [oracle_acc] *
                   len(all_metrics['sample']), label='oracle_acc', linestyle='--')
    axs[0, 1].plot(all_metrics['sample'], [oracle_roc] *
                   len(all_metrics['sample']), label='oracle_roc', linestyle='--')
    axs[0, 1].plot(all_metrics['sample'], all_metrics['pacc'], label='pacc')
    axs[0, 1].plot(all_metrics['sample'], all_metrics['proc'], label='proc')

    axs[1, 0].plot(all_metrics['sample'], all_metrics['pconf'], label='pconf')
    axs[1, 0].plot(all_metrics['sample'],
                   all_metrics['pconf-median'], label='pconf-median')
    axs[1, 0].plot(all_metrics['sample'], [
                   1.0] * len(all_metrics['sample']), label='oracle_conf', linestyle='--')

    axs[1, 1].plot(all_metrics['sample'],
                   all_metrics['pacc-sim'], label='pacc-sim')
    axs[1, 1].plot(all_metrics['sample'],
                   all_metrics['proc-sim'], label='proc-sim')
    axs[1, 1].plot(all_metrics['sample'], [1.0] *
                   len(all_metrics['sample']), label='oracle_sim', linestyle='--')

    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel('sample rate')
            axs[i, j].set_ylabel('metrics')
            axs[i, j].legend()

    plt.savefig(os.path.join(
        job_dir, f'machinery_metrics_{args.sample_strategy}.png'))
    plt.show()

    # plot time measurements, cpu_time = feature_time + pred_time + pconf_time
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].plot(all_metrics['sample'], all_metrics['feature_time'],
                label='feature_time')
    axs[0].plot(all_metrics['sample'], all_metrics['pred_time'],
                label='pred_time')
    axs[0].plot(all_metrics['sample'], all_metrics['pconf_time'],
                label='pconf_time')
    axs[0].set_xlabel('sample rate')
    axs[0].set_ylabel('time (s)')
    axs[0].legend()

    # axs[1].plot(all_metrics['sample'], all_metrics['feature_time'] /
    #             all_metrics['cpu_time'], label='feature_time')
    # axs[1].plot(all_metrics['sample'], all_metrics['pred_time'] /
    #             all_metrics['cpu_time'], label='pred_time')
    # axs[1].plot(all_metrics['sample'], all_metrics['pconf_time'] /
    #             all_metrics['cpu_time'], label='pconf_time')
    axs[1].plot(all_metrics['sample'], all_metrics['nrows'], label='nrows')
    axs[1].set_xlabel('sample rate')
    axs[1].set_ylabel('nrows per request')
    axs[1].legend()

    plt.savefig(os.path.join(
        job_dir, f'machinery_cpu_time_{args.sample_strategy}.png'))

    if args.sample_strategy != 'equal':
        equal_metrics = pd.read_csv(os.path.join(
            job_dir, f'machinery_metrics_equal_{args.low_conf_threshold}.csv'))
        # equal is the baseline strategy, if not equal, we need to compare with the equal strategy
        # we want to the trade-off between accuracy and cpu time, on different sample rates
        # the x-axis is accuracy (acc and roc), the y-axis is cpu time, the color indicates sample rate
        # we use scatter plot to show the trade-off
        colors = matplotlib.cm.rainbow(
            np.linspace(0, 1, len(all_metrics['sample'])))
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs[0][0].scatter(equal_metrics['pacc'], equal_metrics['nrows'],
                          label='equal', marker='x', color=colors)
        axs[0][0].scatter(all_metrics['pacc'], all_metrics['nrows'],
                          label=args.sample_strategy, marker='*', color=colors)
        axs[0][0].plot(equal_metrics['pacc'], equal_metrics['nrows'],
                       label='equal', linestyle='--')
        axs[0][0].plot(all_metrics['pacc'], all_metrics['nrows'],
                       label=args.sample_strategy)
        axs[0][0].set_xlabel('accuracy')
        axs[0][0].set_ylabel('cpu time (s)')
        axs[0][0].legend()

        axs[0][1].scatter(equal_metrics['pacc-sim'], equal_metrics['nrows'],
                          label='equal', marker='x', color=colors)
        axs[0][1].scatter(all_metrics['pacc-sim'], all_metrics['nrows'],
                          label=args.sample_strategy, marker='*', color=colors)
        axs[0][1].plot(equal_metrics['pacc-sim'],
                       equal_metrics['nrows'], label='equal', linestyle='--')
        axs[0][1].plot(all_metrics['pacc-sim'], all_metrics['nrows'],
                       label=args.sample_strategy)
        axs[0][1].set_xlabel('acc similarity')
        axs[0][1].set_ylabel('load cpu time')
        axs[0][1].legend()

        axs[1][0].scatter(equal_metrics['proc'], equal_metrics['nrows'],
                          label='equal', marker='x', color=colors)
        axs[1][0].scatter(all_metrics['proc'], all_metrics['nrows'],
                          label=args.sample_strategy, marker='*', color=colors)
        axs[1][0].plot(equal_metrics['proc'], equal_metrics['nrows'],
                       label='equal', linestyle='--')
        axs[1][0].plot(all_metrics['proc'], all_metrics['nrows'],
                       label=args.sample_strategy)
        axs[1][0].set_xlabel('roc')
        axs[1][0].set_ylabel('nrows')
        axs[1][0].legend()

        axs[1][1].scatter(equal_metrics['proc-sim'], equal_metrics['nrows'],
                          label='equal', marker='x', color=colors)
        axs[1][1].scatter(all_metrics['proc-sim'], all_metrics['nrows'],
                          label=args.sample_strategy, marker='*', color=colors)
        axs[1][1].plot(equal_metrics['proc-sim'],
                       equal_metrics['nrows'], label='equal', linestyle='--')
        axs[1][1].plot(all_metrics['proc-sim'], all_metrics['nrows'],
                       label=args.sample_strategy)
        axs[1][1].set_xlabel('roc similarity')
        axs[1][1].set_ylabel('nrows')
        axs[1][1].legend()

        plt.savefig(os.path.join(
            job_dir, f'machinery_cpu_time_{args.sample_strategy}_vs_equal.png'))


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    plot_1(args)
