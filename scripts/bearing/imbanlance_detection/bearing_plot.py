import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from tap import Tap
import seaborn as sns


class OnlineParser(Tap):
    task: str = "status_classification"  # task name
    sample_strategy: str = 'equal'  # sample strategy
    sample_budget_each: float = 0.1  # sample budget each in avg
    sample_nchunks: int = 10  # number of chunks to sample
    low_conf_threshold: float = 0.8  # low confidence threshold


def plot_1(args: OnlineParser):
    """
    Plot the metrics with different sample rates.
    metrics include: feature_r2, prediction_accuracy, cpu_time, load_cpu_time, compute_cpu_time
    """
    task = args.task
    sample_strategy = args.sample_strategy
    nchunks = args.sample_nchunks

    job_dir = os.path.join('./', task)

    # columns: sample, feature_r2, prediction_accuracy, cpu_time, load_cpu_time, compute_cpu_time
    all_metrics = pd.DataFrame(columns=[
                               'sample', 'fr2', 'fmaxe', 'pacc', 'pacc-sim', 'proc', 'proc-sim', 'pconf', 'pconf-median', 'cpu_time', 'load_cpu_time', 'compute_cpu_time'])

    oracle_acc = 1.0
    oracle_roc = 1.0
    for i in tqdm(range(1, nchunks+1)):
        sample = i * 1.0 / nchunks
        command = f'/home/ckchang/anaconda3/envs/amd/bin/python online_test.py --task {task} --sample_strategy {sample_strategy} --sample_budget_each {sample} --low_conf_threshold {args.low_conf_threshold}'
        command += ' > /dev/null'
        os.system(command)
        feature_dir = os.path.join(job_dir, 'features', f'sample_{sample}')
        fevals_path = os.path.join(
            feature_dir, f'fevals_{sample_strategy}.csv')
        evals_path = os.path.join(feature_dir, f'evals_{sample_strategy}.csv')
        fevals = pd.read_csv(fevals_path)
        evals = pd.read_csv(evals_path)
        feature_r2 = fevals.iloc[-1]['r2']
        fmaxe = fevals.iloc[:-1]['maxe'].max()
        load_cpu_time = fevals.iloc[-1]['load_cpu_time']
        compute_cpu_time = fevals.iloc[-1]['compute_cpu_time']
        cpu_time = fevals.iloc[-1]['cpu_time']
        oracle_acc = evals.iloc[0]['acc']
        oracle_roc = evals.iloc[0]['roc']
        pred_accuracy = evals.iloc[1]['acc']
        pconf = evals.iloc[1]['mean_conf']
        pconf_median = evals.iloc[1]['median_conf']
        pred_similarity = evals.iloc[2]['acc']
        row = {'sample': sample,
               'fr2': feature_r2, 'fmaxe': fmaxe,
               'pacc': pred_accuracy, 'pacc-sim': pred_similarity,
               'proc': evals.iloc[1]['roc'], 'proc-sim': evals.iloc[2]['roc'],
               'pconf': pconf, 'pconf-median': pconf_median,
               'cpu_time': cpu_time, 'load_cpu_time': load_cpu_time, 'compute_cpu_time': compute_cpu_time}
        print(row)
        all_metrics = pd.concat([all_metrics, pd.DataFrame(row, index=[0])])
        # all_metrics = all_metrics.append(row, ignore_index=True)
    all_metrics.to_csv(os.path.join(
        job_dir, f'bearing_metrics_{args.sample_strategy}.csv'))

    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs[0, 0].plot(all_metrics['sample'],
                   all_metrics['fr2'], label='feature_r2')
    # axs[1,0].plot(all_metrics['sample'], all_metrics['fmaxe'], label='feature_maxe')
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
        job_dir, f'bearing_metrics_{args.sample_strategy}.png'))
    plt.show()

    # plot time measurements, cpu_time = load_cpu_time + compute_cpu_time
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].plot(all_metrics['sample'], all_metrics['cpu_time'],
                label='cpu_time')
    axs[0].plot(all_metrics['sample'], all_metrics['load_cpu_time'],
                label='load_cpu_time')
    axs[0].plot(all_metrics['sample'], all_metrics['compute_cpu_time'],
                label='compute_cpu_time')
    axs[0].set_xlabel('sample rate')
    axs[0].set_ylabel('time (s)')
    axs[0].legend()

    axs[1].plot(all_metrics['sample'], all_metrics['load_cpu_time'] /
                all_metrics['cpu_time'], label='load_cpu_time')
    axs[1].plot(all_metrics['sample'], all_metrics['compute_cpu_time'] /
                all_metrics['cpu_time'], label='compute_cpu_time')
    axs[1].set_xlabel('sample rate')
    axs[1].set_ylabel('fraction of cpu time')
    axs[1].legend()

    plt.savefig(os.path.join(
        job_dir, f'bearing_cpu_time_{args.sample_strategy}.png'))

    if args.sample_strategy != 'equal':
        equal_metrics = pd.read_csv(os.path.join(
            job_dir, f'bearing_metrics_equal.csv'))
        # equal is the baseline strategy, if not equal, we need to compare with the equal strategy
        # we want to the trade-off between accuracy and cpu time, on different sample rates
        # the x-axis is accuracy (acc and roc), the y-axis is cpu time, the color indicates sample rate
        # we use scatter plot to show the trade-off
        colors = matplotlib.cm.rainbow(
            np.linspace(0, 1, len(all_metrics['sample'])))
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].scatter(equal_metrics['pacc'], equal_metrics['cpu_time'],
                       label='equal', marker='x', color=colors)
        axs[0].scatter(all_metrics['pacc'], all_metrics['cpu_time'],
                       label=args.sample_strategy, marker='*', color=colors)
        axs[0].plot(equal_metrics['pacc'], equal_metrics['cpu_time'],
                    label='equal', linestyle='--')
        axs[0].plot(all_metrics['pacc'], all_metrics['cpu_time'],
                    label=args.sample_strategy)
        axs[0].set_xlabel('accuracy')
        axs[0].set_ylabel('cpu time (s)')
        axs[0].legend()

        axs[1].scatter(equal_metrics['pacc'], equal_metrics['load_cpu_time'],
                       label='equal', marker='x', color=colors)
        axs[1].scatter(all_metrics['pacc'], all_metrics['load_cpu_time'],
                       label=args.sample_strategy, marker='*', color=colors)
        axs[1].plot(equal_metrics['pacc'],
                    equal_metrics['load_cpu_time'], label='equal', linestyle='--')
        axs[1].plot(all_metrics['pacc'], all_metrics['load_cpu_time'],
                    label=args.sample_strategy)
        axs[1].set_xlabel('accuracy')
        axs[1].set_ylabel('load cpu time')
        axs[1].legend()

        plt.savefig(os.path.join(
            job_dir, f'bearing_cpu_time_{args.sample_strategy}_vs_equal.png'))


if __name__ == "__main__":
    args = OnlineParser().parse_args()
    plot_1(args)
