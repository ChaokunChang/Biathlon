import os
import sys
import json
import pandas as pd
from tap import Tap
from tqdm import tqdm
import numpy as np

from apxinfer.examples.all_tasks import ALL_REG_TASKS, ALL_CLS_TASKS

# read all csv files in the directory, and merge them into one dataframe
# save the dataframe to a csv file


class MergerArgs(Tap):
    csv_dir: str = "./cache/evals"
    output_dir: str = "./cache"
    output_filename: str = "evals.csv"
    recursive: bool = False
    avg: bool = False
    debug: bool = False


def parse_filename(filename, verbose: bool = False):
    if verbose:
        print(filename)
    # remove the extension
    filename = filename.replace(".csv", "")
    elements = filename.split("_")
    task_name = elements[0]
    _ = elements[1]
    model_name = elements[2]
    nparts = int(elements[3])
    ncfgs = int(elements[4])
    ncores = int(elements[5])
    max_error = float(elements[6])
    return task_name, model_name, nparts, ncfgs, ncores, max_error


def merge_csv(args: MergerArgs):
    csv_dir = args.csv_dir
    # get all files in the directory
    files = []
    if args.recursive:
        for root, dirs, filenames in os.walk(csv_dir):
            for filename in filenames:
                if filename.endswith(".csv"):
                    files.append(os.path.join(root, filename))
    else:
        files = [os.path.join(csv_dir, filename) for filename in os.listdir(csv_dir) if filename.endswith(".csv")]
    print(f"Found {len(files)} csv files in {csv_dir}")

    df = pd.DataFrame()
    for fpath in tqdm(files):
        if fpath.endswith(".csv"):
            fname = os.path.basename(fpath)
            task_name, model_name, nparts, ncfgs, ncores, max_error = parse_filename(fname)
            # if task_name not in select_names:
            #     continue
            df_tmp = pd.read_csv(os.path.join(fpath))
            df_tmp["task_name"] = task_name
            df_tmp["model_name"] = model_name
            # df_tmp["nparts"] = nparts # already in the csv file
            df_tmp["ncfgs"] = ncfgs
            # df_tmp["ncores"] = ncores # already in the csv file
            # df_tmp["max_error"] = max_error # already in the csv file
            # move the columns to the front
            cols = df_tmp.columns.tolist()
            cols = cols[-3:] + cols[:-3]
            df_tmp = df_tmp[cols]

            if "pest_nsamples" not in df_tmp.columns:
                df_tmp["pest_nsamples"] = 1000

            # set accuracy
            if task_name in ALL_REG_TASKS:
                acc_type = "r2"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]
            elif task_name in ALL_CLS_TASKS:
                acc_type = "f1"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]
            else:
                raise ValueError(f"unknown task: {task_name}")

            # get exact acc from the row with 1.0 min_conf
            exact_acc = df_tmp[df_tmp['min_conf'] == 1.0]['accuracy'].values[0]
            df_tmp['acc_loss'] = exact_acc - df_tmp['accuracy']
            df_tmp['acc_loss_pct'] = df_tmp['acc_loss'] / exact_acc
            df_tmp['acc_diff'] = abs(df_tmp['acc_loss'])
            # list_cols : ["avg_sample_query", "avg_qtime_query"]
            df_tmp['avg_sample_query'] = df_tmp['avg_sample_query'].apply(lambda x: json.loads(x))
            df_tmp['avg_qtime_query'] = df_tmp['avg_qtime_query'].apply(lambda x: json.loads(x))
            df = pd.concat([df, df_tmp])
    return df


def seed_selection(df: pd.DataFrame) -> pd.DataFrame:
    seeds_dict = {
        "tripsralf2h": [1, 2, 3],
        "tickralf": [0, 1, 2],
        # "tickvaryNM8": [0, 1, 2],
        # "battery": [0, 4, 3],
        "batteryv2": [3, 4, 1],
        "turbofan": [1, 2, 3],

        "machinery": [0, 2, 4],
        "tdfraudralf": [0, 1, 2],
        "studentqno18": [0, 4, 3]
    }
    for task_name, seeds in seeds_dict.items():
        df = df[(df['task_name'] != task_name) | ((df['task_name'] == task_name) & df['seed'].isin(seeds))]
    return df


def main():
    args = MergerArgs().parse_args()
    raw_df = merge_csv(args)
    print(f'columns: {raw_df.columns}')
    useless_cols = ['run_shared', 'nocache',
                    'interpreter', 'min_confs',
                    'run_offline', 'run_baseline']
    df = raw_df.drop(columns=useless_cols, errors='ignore')
    df = seed_selection(df)
    if args.avg:
        keys = ['task_name', 'model_name', 'model',
                'task_home', 'agg_qids',
                'qinf', 'policy', 'nparts', 'ncfgs',
                'ncores', 'loading_mode',
                'scheduler_batch', 'scheduler_init',
                'max_error', 'min_conf',
                'pest', 'pest_constraint',
                'pest_nsamples', 'pest_seed',
                'ralf_budget']
        # deduplicate by the keys + ['seed'], keep the first
        print(f'total number of rows: {len(df)}')
        df = df.groupby(keys + ['seed']).first().reset_index()
        print(f'total number of rows: {len(df)}')
        df_grp = df.groupby(keys)

        if args.debug:
            # select the rows with task_name=Fraud-Detection, 
            # loading_mode=0, scheduler_batch=3, scheduler_init=1, 
            # min_conf in [0.98, 0.99, 0.995, 0.999]
            tmp_rows = df[df['task_name'] == 'Fraud-Detection']
            tmp_rows = tmp_rows[tmp_rows['loading_mode'] == 0]
            tmp_rows = tmp_rows[tmp_rows['scheduler_init'] == 1]
            tmp_rows = tmp_rows[tmp_rows['scheduler_batch'] == 3]
            # tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.95, 0.98, 0.99, 0.995, 0.999])]
            # tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.6, 0.7, 0.8, 0.9, 0.95])]
            tmp_rows = tmp_rows[tmp_rows['min_conf'].isin([0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999])]
            tmp_rows = tmp_rows.sort_values(by=['min_conf', 'seed'])
            print(tmp_rows[["min_conf", 'seed', 'avg_latency', "speedup", "similarity"]])

        list_cols = ["avg_sample_query", "avg_qtime_query"]
        non_list_cols = [col for col in df.columns if col not in list_cols + keys]
        # for columns except list_cols, take the mean;
        # for list_cols, take the mean of each element
        # print(f'non_list_cols: {non_list_cols}')
        df_non_list = df_grp[non_list_cols].mean().reset_index()
        for col in list_cols:
            df_list = df_grp[col].apply(lambda x: np.mean(x.tolist(), axis=0).tolist()).reset_index()
        df = pd.merge(df_non_list, df_list, on=keys)

    # add column naggs, which = len(agg_qids)
    df['naggs'] = df['agg_qids'].apply(lambda x: len(json.loads(x)))
    print(f'total number of rows: {len(df)}')
    df.to_csv(os.path.join(args.output_dir, args.output_filename), index=False)


if __name__ == "__main__":
    main()
