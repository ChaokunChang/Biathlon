import os
import json
import pandas as pd
from tap import Tap


class EvalArgs(Tap):
    nparts: int = 2
    ncfgs: int = None
    ncores: int = 0
    model: str = "xgb"
    max_error: float = 1.0
    agg_qids: list[int] = [1, 2, 3]
    nocache: bool = False
    qinf: str = "sobol"
    load_only: bool = False


args = EvalArgs().parse_args()

nparts = args.nparts
if args.ncfgs is None:
    ncfgs = nparts
else:
    ncfgs = args.ncfgs
ncores = args.ncores
max_error = args.max_error
model = args.model
qinf = args.qinf

offline_nreqs = 100
if nparts >= 20:
    offline_nreqs = 50

agg_qids = args.agg_qids

command = f"python run.py --example trips --stage offline --task test/trips --model {model} --nparts {nparts} --nreqs {offline_nreqs} --ncfgs {ncfgs} --clear_cache --ncores {ncores}"
print(command)
if args.nocache:
    os.system(command=command)


cmd_prefix = f"python run.py --example trips --stage online --task test/trips --model {model} --nparts {nparts} --offline_nreqs {offline_nreqs} --ncfgs {ncfgs} --ncores {ncores}"
path_prefix = f"/home/ckchang/.cache/apxinf/xip/test/trips/seed-0/online/{model}/ncores-{ncores}/ldnthreads-1/nparts-{nparts}"

evals = []
results = []


def extract_result(all_info: dict, min_conf, base_time=None):
    avg_smpl_rate = 0.0
    for qid in agg_qids:
        avg_smpl_rate += all_info["avg_sample_query"][qid]
    avg_smpl_rate /= len(agg_qids)

    result = {
        "ncores": ncores,
        "nparts": nparts,
        "policy": "greedy",
        "max_error": max_error,
        "min_conf": min_conf,
        "sampling_rate": avg_smpl_rate,
        "similarity": all_info["evals_to_ext"]["r2"],
        "accuracy": all_info["evals_to_gt"]["r2"],
        "avg_latency": all_info["avg_ppl_time"],
        "speedup": base_time / all_info["avg_ppl_time"] if base_time is not None else 1.0,
        "BD:AFC": all_info["avg_query_time"],
        "BD:AMI": all_info["avg_pred_time"],
        "BD:Sobol": all_info["avg_scheduler_time"],

        "similarity-r2": all_info["evals_to_ext"]["r2"],
        "accuracy-r2": all_info["evals_to_gt"]["r2"],
        "similarity-mse": all_info["evals_to_ext"]["mse"],
        "accuracy-mse": all_info["evals_to_gt"]["mse"],
        "similarity-mape": all_info["evals_to_ext"]["mape"],
        "accuracy-mape": all_info["evals_to_gt"]["mape"],
        "similarity-maxe": all_info["evals_to_ext"]["maxe"],
        "accuracy-maxe": all_info["evals_to_gt"]["maxe"],
    }
    return result


command = f"{cmd_prefix} --exact --ncores {ncores}"
print(command)
if args.nocache:
    os.system(command=command)

eval_path = f"{path_prefix}/exact/evals_exact.json"
evals.append(extract_result(json.load(open(eval_path)), 1.0))
print(f"last eval: {evals[-1]}")

for min_conf in [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]:
    command = f"{cmd_prefix} --qinf {qinf} --pest_constraint error --max_error {max_error} --min_conf {min_conf}"
    if not args.load_only:
        os.system(command=command)

    eval_path = f"{path_prefix}/ncfgs-{ncfgs}/pest-error-MC-1000-0/qinf-{qinf}/scheduler-optimizer-1-1/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
    if os.path.exists(eval_path):
        evals.append(extract_result(json.load(open(eval_path)), min_conf, evals[0]["avg_latency"]))
    else:
        eval_path = f"{path_prefix}/ncfgs-{ncfgs}/pest-error-MC-1000-0/qinf-{qinf}/scheduler-optimizer-1/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
        if os.path.exists(eval_path):
            evals.append(extract_result(json.load(open(eval_path)), min_conf, evals[0]["avg_latency"]))
    print(f"last eval: {evals[-1]}")

# conver evals to pd.DataFrame and save as csv
evals = pd.DataFrame(evals)
evals.to_csv(f"./trips_{qinf}_{model}_{nparts}_{ncfgs}_{ncores}_{max_error}.csv", index=False)
