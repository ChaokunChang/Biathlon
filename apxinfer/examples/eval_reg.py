import os
import json
import pandas as pd
from tap import Tap

ALL_REG_TASKS = ["trips", "tripsfeast", "tick", "tickv2"]
ALL_CLS_TASKS = ["cheaptrips", "cheaptripsfeast", "machinery", "ccfraud", "machinerymulti"]

MachineryVaryNF = [f"machineryf{i}" for i in range(1, 8)] + [f"machineryxf{i}" for i in range(1, 8)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)] + [f"machinerymultixf{i}" for i in range(1, 8)]
ALL_CLS_TASKS += MachineryVaryNF
ALL_CLS_TASKS += MachineryMultiVaryNF

TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 8)]
ALL_REG_TASKS += TickVaryNMonths


class EvalArgs(Tap):
    interpreter: str = "python"
    task_home: str = "final"
    task_name: str = None
    model: str = None
    nparts: int = 20
    ncfgs: int = None
    ncores: int = 0
    loading_mode: int = 1
    max_error: float = 1.0
    agg_qids: list[int] = None
    run_shared: bool = False
    qinf: str = "sobol"
    policy: str = "optimizer"
    scheduler_init: int = 1
    scheduler_batch: int = 1
    nocache: bool = False
    seed: int = 0
    min_confs: list[float] = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]

    def process_args(self):
        assert self.task_name is not None
        assert self.model is not None
        assert self.agg_qids is not None


args = EvalArgs().parse_args()
args.process_args()

EVALS_HOME = "./evals"
interpreter = args.interpreter
if interpreter != "python":
    interpreter = f"sudo {interpreter}"
TASK_HOME = args.task_home
TASK_NAME = args.task_name
model = args.model
nparts = args.nparts
if args.ncfgs is None:
    ncfgs = nparts
else:
    ncfgs = args.ncfgs
ncores = args.ncores
loading_mode = args.loading_mode
max_error = args.max_error
min_confs = args.min_confs
qinf = args.qinf
policy = args.policy
scheduler_init = args.scheduler_init
scheduler_batch = args.scheduler_batch
seed = args.seed

offline_nreqs = 100
if nparts >= 20:
    offline_nreqs = 50

agg_qids = args.agg_qids


def extract_result(all_info: dict, min_conf, base_time=None):
    avg_smpl_rate = 0.0
    for qid in agg_qids:
        avg_smpl_rate += all_info["avg_sample_query"][qid]
    avg_smpl_rate /= len(agg_qids)

    result = {
        **args.as_dict(),
        "min_conf": min_conf,
        "sampling_rate": avg_smpl_rate,
        "avg_latency": all_info["avg_ppl_time"],
        "speedup": base_time / all_info["avg_ppl_time"]
        if base_time is not None
        else 1.0,
        "BD:AFC": all_info["avg_query_time"],
        "BD:AMI": all_info["avg_pred_time"],
        "BD:Sobol": all_info["avg_scheduler_time"],
        "avg_nrounds": all_info["avg_nrounds"],
        "avg_sample_query": all_info["avg_sample_query"],
        "avg_qtime_query": all_info["avg_qtime_query"],
    }
    if args.task_name in ALL_REG_TASKS:
        accs = {
            "similarity": all_info["evals_to_ext"]["r2"],
            "accuracy": all_info["evals_to_gt"]["r2"],
            "similarity-r2": all_info["evals_to_ext"]["r2"],
            "accuracy-r2": all_info["evals_to_gt"]["r2"],
            "similarity-mse": all_info["evals_to_ext"]["mse"],
            "accuracy-mse": all_info["evals_to_gt"]["mse"],
            "similarity-mape": all_info["evals_to_ext"]["mape"],
            "accuracy-mape": all_info["evals_to_gt"]["mape"],
            "similarity-maxe": all_info["evals_to_ext"]["maxe"],
            "accuracy-maxe": all_info["evals_to_gt"]["maxe"],
        }
    else:
        assert args.task_name in ALL_CLS_TASKS
        accs = {
            "similarity": all_info["evals_to_ext"]["acc"],
            "accuracy": all_info["evals_to_gt"]["acc"],
            "similarity-acc": all_info["evals_to_ext"]["acc"],
            "accuracy-acc": all_info["evals_to_gt"]["acc"],
            "similarity-f1": all_info["evals_to_ext"]["f1"],
            "accuracy-f1": all_info["evals_to_gt"]["f1"],
            "similarity-auc": all_info["evals_to_ext"]["auc"],
            "accuracy-auc": all_info["evals_to_gt"]["auc"],
        }
    result = {**result, **accs}
    return result


cmd_prefix = f"{interpreter} run.py --example {TASK_NAME} --stage online --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --offline_nreqs {offline_nreqs} --ncfgs {ncfgs} --ncores {ncores} --loading_mode {loading_mode} --seed {seed}"
results_prefix = f"/home/ckchang/.cache/apxinf/xip/{TASK_HOME}/{TASK_NAME}/seed-{seed}"

if args.run_shared:
    # offline
    command = f"{interpreter} run.py --example {TASK_NAME} --stage offline --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --nreqs {offline_nreqs} --ncfgs {ncfgs} --clear_cache --ncores {ncores} --loading_mode {loading_mode} --seed {seed}"
    print(command)
    qcm_path = f"{results_prefix}/offline/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/ncfgs-{ncfgs}/nreqs-{offline_nreqs}/model/xip_qcm.pkl"
    if not os.path.exists(qcm_path):
        os.system(command=command)

    # exact
    command = f"{cmd_prefix} --ncores {ncores} --exact"
    print(command)
    exact_path = f"{results_prefix}/online/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/exact/evals_exact.json"
    if not os.path.exists(exact_path):
        os.system(command=command)
else:
    evals = []
    path_prefix = (
        f"{results_prefix}/online/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}"
    )
    eval_path = f"{path_prefix}/exact/evals_exact.json"
    evals.append(extract_result(json.load(open(eval_path)), 1.0))
    print(f"last eval: {evals[-1]}")

    for min_conf in min_confs:
        cmd_prefix += (
            f" --scheduler {policy} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch}"
        )
        command = f"{cmd_prefix} --pest_constraint error --pest_seed {seed} --max_error {max_error} --min_conf {min_conf}"
        eval_path = f"{path_prefix}/ncfgs-{ncfgs}/pest-error-MC-1000-{seed}/qinf-{qinf}/scheduler-{policy}-{scheduler_init}-{scheduler_batch}/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
        if args.nocache or (not os.path.exists(eval_path)):
            os.system(command=command)
        evals.append(
            extract_result(
                json.load(open(eval_path)), min_conf, evals[0]["avg_latency"]
            )
        )
        print(f"last eval: {evals[-1]}")

    evals_dir = os.path.join(EVALS_HOME, f'seed-{seed}', f'mode-{loading_mode}')
    if not os.path.exists(evals_dir):
        os.makedirs(evals_dir, exist_ok=True)

    # conver evals to pd.DataFrame and save as csv
    evals = pd.DataFrame(evals)
    evals.to_csv(
        f"{evals_dir}/{TASK_NAME}_{qinf}-{policy}-{scheduler_init}-{scheduler_batch}_{model}_{nparts}_{ncfgs}_{ncores}_{max_error}.csv",
        index=False,
    )
