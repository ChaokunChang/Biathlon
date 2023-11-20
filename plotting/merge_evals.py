import os
import sys
import json
import pandas as pd

csv_dir = "./cache/evals"
# read all csv files in the directory, and merge them into one dataframe
# parse the filename to get the task name, model name, nparts, ncfgs, ncores, and max_error,
# and add them as columns to the dataframe
# save the dataframe to a csv file


def parse_filename(filename):
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
    elif task_name == "machinery":
        if model_name == "mlp":
            task_name = "machinery-v1"
        elif model_name == "dt":
            task_name = "machinery-v2"
        elif model_name == "knn":
            task_name = "machinery-v3"
    return task_name, model_name, nparts, ncfgs, ncores, max_error


def merge_csv(csv_dir):
    df = pd.DataFrame()
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv"):
            task_name, model_name, nparts, ncfgs, ncores, max_error = parse_filename(filename)
            df_tmp = pd.read_csv(os.path.join(csv_dir, filename))
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

            # set accuracy
            if task_name in ["tick-v1", "tick-v2",]:
                acc_type = "r2"
                if f"accuracy-{acc_type}" in df_tmp.columns:
                    df_tmp['accuracy'] = df_tmp[f"accuracy-{acc_type}"]
                    df_tmp['similarity'] = df_tmp[f"similarity-{acc_type}"]
            if task_name in ['cheaptrips', "machinery-v1", "machinery-v2", "machinery-v3"]:
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


def main():
    df = merge_csv(csv_dir)
    df.to_csv(os.path.join(csv_dir, "..", "evals.csv"), index=False)


if __name__ == "__main__":
    main()