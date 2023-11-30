import os
import sys
import json
import pandas as pd
from tap import Tap
from tqdm import tqdm

# read all csv files in the directory, and merge them into one dataframe
# parse the filename to get the task name, model name, nparts, ncfgs, ncores, and max_error,
# and add them as columns to the dataframe
# save the dataframe to a csv file


class EvalArgs(Tap):
    csv_dir: str = "./cache/evals"
    output_dir: str = "./cache"
    output_filename: str = "evals.csv"
    recursive: bool = False
    avg: bool = False


select_names = ["Trips-Fare", "tickvaryNM8", "Bearing-MLP", "Fraud-Detection"]
select_names += [f'machineryxf{i}' for i in range(1, 9)]
select_names += [f'tickvaryNM{i}' for i in range(2, 40)]


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
    if task_name == "tick":
        task_name = "tick-v1"
    elif task_name == "tickv2":
        task_name = "tick-v2"
    elif task_name == "tripsfeast":
        task_name = "Trips-Fare"
    elif task_name == "machinery":
        if model_name == "mlp":
            task_name = "Bearing-MLP"
        elif model_name == "dt":
            task_name = "machinery-v2"
        elif model_name == "knn":
            task_name = "Bearing-KNN"
    elif task_name == "machinerymulti":
        task_name = "Bearing-Multi"
    elif task_name == "tdfraud":
        task_name = "Fraud-Detection"
    return task_name, model_name, nparts, ncfgs, ncores, max_error


def merge_csv(args: EvalArgs):
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
            if task_name not in select_names:
                continue
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
            if task_name in ["tick-v1", "tick-v2", "tickvaryNM1",
                             "tickvaryNM8",
                             "Tick-Price", "tripsfeast", "Trips-Fare"]:
                acc_type = "r2"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]
            if task_name in ['cheaptrips', "machinery-v1", "machinery-v2",
                             "machinery-v3", "machinerymulti", "tdfraud",
                             "tdfraudrandom",
                             "Fraud-Detection", "Bearing-MLP",
                             "Bearing-Multi", "Bearing-KNN",]:
                acc_type = "f1"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]

            # get exact acc from the row with 1.0 min_conf
            exact_acc = df_tmp[df_tmp['min_conf'] == 1.0]['accuracy'].values[0]
            df_tmp['acc_loss'] = exact_acc - df_tmp['accuracy']
            df_tmp['acc_loss_pct'] = df_tmp['acc_loss'] / exact_acc
            df_tmp['acc_diff'] = abs(df_tmp['acc_loss'])
            df = pd.concat([df, df_tmp])
    return df


def tmp_handle_tdfraud(df: pd.DataFrame) -> pd.DataFrame:
    # for task_name=Fraud-Detection, only keep the rows with seed = 1 and seed = 2
    df = df[(df['task_name'] != 'Fraud-Detection') | (df['seed'].isin([1, 2]))]
    return df


def tmp_handle_tickvaryNM8(df: pd.DataFrame) -> pd.DataFrame:
    # for task_name=Fraud-Detection, only keep the rows with seed = 1 and seed = 2
    df = df[(df['task_name'] != 'tickvaryNM8') | (df['seed'].isin([0, 1, 2]))]
    return df


def seed_selection(df: pd.DataFrame) -> pd.DataFrame:
    seeds_dict = {
        "Trips-Fare": [1, 2, 3],
        "tickvaryNM8": [0, 1, 2],
        "Bearing-MLP": [0, 2, 4],
        "Fraud-Detection": [1, 2]
    }
    df = df[(df['task_name'] != 'Trips-Fare') | (df['seed'].isin(seeds_dict['Trips-Fare']))]
    df = df[(df['task_name'] != 'tickvaryNM8') | (df['seed'].isin(seeds_dict['tickvaryNM8']))]
    df = df[(df['task_name'] != 'Bearing-MLP') | (df['seed'].isin(seeds_dict['Bearing-MLP']))]
    df = df[(df['task_name'] != 'Fraud-Detection') | (df['seed'].isin(seeds_dict['Fraud-Detection']))]
    return df


def tmp_handler_for_varynm(df: pd.DataFrame) -> pd.DataFrame:
    # for task that starts with tickvaryNM, set the baseline AFC as 1.2279942959778067
    # baseline is those rows with min_conf = 1.0
    varyNM_tasks = [f"tickvaryNM{i}" for i in range(1, 40)]
    for task_name in varyNM_tasks:
        # for those with min_conf=1.0, set BD:AFC as 1.2279942959778067, and update avg_latency accordingly
        # for those with min_conf<1.0, update speedup accordingly
        df_tmp = df[df["task_name"] == task_name].copy()
        if len(df_tmp) == 0:
            continue

        df_tmp_baseline = df_tmp[df_tmp["min_conf"] == 1.0].copy()
        old_afc = df_tmp_baseline["BD:AFC"].values[0]
        df_tmp_baseline["BD:AFC"] = 1.2279942959778067
        df_tmp_baseline["avg_latency"] += (1.2279942959778067 - old_afc)

        baseline_lat = df_tmp_baseline["avg_latency"].values[0]
        df_tmp_baseline["speedup"] = df_tmp_baseline["avg_latency"] / baseline_lat

        df_tmp_others = df_tmp[df_tmp["min_conf"] != 1.0].copy()
        df_tmp_others["speedup"] = baseline_lat / df_tmp_others["avg_latency"]

        df = df[df["task_name"] != task_name]
        df = pd.concat([df, df_tmp_baseline, df_tmp_others])
    return df


def main():
    args = EvalArgs().parse_args()
    raw_df = merge_csv(args)
    print(f'columns: {raw_df.columns}')
    useless_cols = ['run_shared', 'nocache', 'interpreter',
                    'min_confs', "avg_sample_query", "avg_qtime_query"]
    df = raw_df.drop(columns=useless_cols)
    df = seed_selection(df)
    df = tmp_handler_for_varynm(df)
    if args.avg:
        # seed,
        # agg_qids,task_home,
        # similarity-maxe,accuracy-maxe,sampling_rate,avg_latency,speedup,BD:AFC,BD:AMI,BD:Sobol,similarity,accuracy,similarity-r2,accuracy-r2,similarity-mse,accuracy-mse,similarity-mape,accuracy-mape,acc_loss,acc_loss_pct,acc_diff,similarity-auc,accuracy-auc,similarity-acc,accuracy-acc,similarity-f1,accuracy-f1
        keys = ['task_name', 'model_name', 'model',
                'task_home', 'agg_qids',
                'qinf', 'policy', 'nparts', 'ncfgs',
                'ncores', 'loading_mode',
                'scheduler_batch', 'scheduler_init',
                'max_error', 'min_conf', 'pest_nsamples']
        # deduplicate by the keys + ['seed'], keep the first
        print(f'total number of rows: {len(df)}')
        df = df.groupby(keys + ['seed']).first().reset_index()
        print(f'total number of rows: {len(df)}')
        df = df.groupby(keys).mean().reset_index()

    # add columns "avg_sample_query", "avg_qtime_query" back
    # the number of rows should be equal to df
    to_add = raw_df[raw_df['seed'] == 0]
    df = pd.merge(df, to_add[keys+['avg_sample_query', 'avg_qtime_query']],
                  on=keys, how='left')

    # add column naggs, which = len(agg_qids)
    df['naggs'] = df['agg_qids'].apply(lambda x: len(json.loads(x)))
    print(f'total number of rows: {len(df)}')
    df.to_csv(os.path.join(args.output_dir, args.output_filename), index=False)


if __name__ == "__main__":
    main()
