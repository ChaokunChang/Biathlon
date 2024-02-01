import os
import json
import pandas as pd
from tap import Tap

from apxinfer.core.config import EXP_HOME

ALL_REG_TASKS = ["trips", "tripsfeast", "tick", "tickv2",
                 "battery", "batteryv2",
                 "turbofan", "turbofanall"]
ALL_CLS_TASKS = ["cheaptrips", "cheaptripsfeast", "machinery",
                 "ccfraud", "machinerymulti",
                 "tdfraud", "tdfraudrandom", "tdfraudkaggle",
                 "student"]

StudentQNo = [f"studentqno{i}" for i in range(1, 19)]
ALL_CLS_TASKS += StudentQNo

StudentQNo18VaryNF = [f"studentqno18nf{i}" for i in range(1, 13)]
ALL_CLS_TASKS += StudentQNo18VaryNF

MachineryVaryNF = [f"machinerynf{i}" for i in range(1, 8)] + [f"machineryxf{i}" for i in range(1, 9)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)] + [f"machinerymultixf{i}" for i in range(1, 9)]
ALL_CLS_TASKS += MachineryVaryNF
ALL_CLS_TASKS += MachineryMultiVaryNF

TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
ALL_REG_TASKS += TickVaryNMonths

TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 30)]
ALL_CLS_TASKS += TripsFeastVaryWindow


class EvalArgs(Tap):
    interpreter: str = "python"
    task_home: str = "final"
    task_name: str = None
    model: str = None
    nparts: int = 100
    ncfgs: int = None
    ncores: int = 1
    loading_mode: int = 0
    max_error: float = 1.0
    agg_qids: list[int] = None
    run_shared: bool = False
    run_offline: bool = False
    run_baseline: bool = False
    qinf: str = "sobol"
    policy: str = "optimizer"
    scheduler_init: int = 1
    scheduler_batch: int = 1
    nocache: bool = False
    seed: int = 0
    min_confs: list[float] = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    pest_nsamples: int = 1000

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
pest_nsamples = args.pest_nsamples

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
        "meet_rate": all_info["meet_rate"],
        "avg_real_error": all_info["avg_real_error"],
    }
    dropped_keys = ['run_shared', 'run_offline', 'run_baseline', 'nocache', 'interpreter', 'min_confs']
    for key in dropped_keys:
        result.pop(key, None)
    for key in all_info["evals_to_ext"]:
        if key in ['time', 'size']:
            continue
        result[f"similarity-{key}"] = all_info["evals_to_ext"][key]
    for key in all_info["evals_to_gt"]:
        if key in ['time', 'size']:
            continue
        result[f"accuracy-{key}"] = all_info["evals_to_gt"][key]
    if args.task_name in ALL_REG_TASKS:
        result['similarity'] = result['similarity-r2']
        result['accuracy'] = result['accuracy-r2']
    else:
        assert args.task_name in ALL_CLS_TASKS
        result['similarity'] = result['similarity-f1']
        result['accuracy'] = result['accuracy-f1']
    return result


cmd_prefix = f"{interpreter} run.py --example {TASK_NAME} --stage online --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --offline_nreqs {offline_nreqs} --ncfgs {ncfgs} --ncores {ncores} --loading_mode {loading_mode} --seed {seed}"
results_prefix = f"{EXP_HOME}/{TASK_HOME}/{TASK_NAME}/seed-{seed}"

if args.run_offline or args.run_shared:
    # offline
    command = f"{interpreter} run.py --example {TASK_NAME} --stage offline --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts} --nreqs {offline_nreqs} --ncfgs {ncfgs} --clear_cache --ncores {ncores} --loading_mode {loading_mode} --seed {seed}"
    print(command)
    qcm_path = f"{results_prefix}/offline/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/ncfgs-{ncfgs}/nreqs-{offline_nreqs}/model/xip_qcm.pkl"
    os.system(command=command)
if args.run_baseline or args.run_shared:
    # exact
    command = f"{cmd_prefix} --ncores {ncores} --exact"
    print(command)
    exact_path = f"{results_prefix}/online/{model}/ncores-{ncores}/ldnthreads-{loading_mode}/nparts-{nparts}/exact/evals_exact.json"
    os.system(command=command)

if not args.run_shared and not args.run_offline and not args.run_baseline:
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
        command = f"{cmd_prefix} --pest_constraint error --pest_seed {seed} --pest_nsamples {pest_nsamples} --max_error {max_error} --min_conf {min_conf}"
        eval_path = f"{path_prefix}/ncfgs-{ncfgs}/pest-error-MC-{pest_nsamples}-{seed}/qinf-{qinf}/scheduler-{policy}-{scheduler_init}-{scheduler_batch}/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
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
    csv_path = f"{evals_dir}/{TASK_NAME}_{qinf}-{policy}-{scheduler_init}-{scheduler_batch}-{pest_nsamples}_{model}_{nparts}_{ncfgs}_{ncores}_{max_error}.csv"
    evals = pd.DataFrame(evals)
    evals.to_csv(csv_path, index=False)
