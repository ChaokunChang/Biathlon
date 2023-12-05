from tap import Tap
import os
import pandas as pd
import json

MachineryVaryNF = [f"machineryf{i}" for i in range(1, 8)]
MachineryVaryXNF = [f"machineryxf{i}" for i in range(1, 9)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)]
MachineryMultiVaryXNF = [f"machinerymultixf{i}" for i in range(1, 9)]
TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 1000)]


class ExpArgs(Tap):
    interpreter = "/home/ckchang/anaconda3/envs/apx/bin/python"
    task_home: str = "final2"
    exp: str = None
    model: str = None  # see each exp

    ncores: int = 1  # 1, 0
    loading_mode: int = 0  # 0, 1, 2, 5, 10
    nparts: int = 100
    ncfgs: int = 100
    seed: int = 0

    prep: bool = False
    skip_shared: bool = False

    default_only: bool = False
    collect_only: bool = False

    def process_args(self):
        assert self.exp is not None
        if not self.prep:
            assert self.model is not None


def run_prepare(args: ExpArgs):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter}"
    else:
        cmd = f"sudo {interpreter}"
    assert args.prep is True
    cmd = f"{cmd} prep.py --interpreter {interpreter} --task_home {args.task_home} --task_name {args.exp} --prepare_again --seed {args.seed}"
    os.system(cmd)


def extract_result(keys: dict, all_info: dict,
                   task_name: str, agg_qids: str,
                   base_time=None):
    agg_qids_list = agg_qids.split()
    avg_smpl_rate = 0.0
    for qid in agg_qids_list:
        avg_smpl_rate += all_info["avg_sample_query"][qid]
    avg_smpl_rate /= len(agg_qids_list)

    result = {
        "task_name": task_name,
        **keys,
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
    for key in all_info["evals_to_ext"]:
        if key in ['time', 'size']:
            continue
        result[f"similarity-{key}"] = all_info["evals_to_ext"][key]
    for key in all_info["evals_to_gt"]:
        if key in ['time', 'size']:
            continue
        result[f"accuracy-{key}"] = all_info["evals_to_gt"][key]
    if "similarity-r2" in result.keys():
        result['similarity'] = result['similarity-r2']
        result['accuracy'] = result['accuracy-r2']
    else:
        assert "similarity-f1" in result.keys()
        result['similarity'] = result['similarity-f1']
        result['accuracy'] = result['accuracy-f1']
    return result


def collect_results(args: ExpArgs, task_name: str, agg_qids: str, max_errors: list[float]) -> pd.DataFrame:
    TASK_HOME = args.task_home
    min_confs = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    qinf = "sobol"
    policy = "optimizer"
    pest_nsamples = 1000
    naggs = len(agg_qids.split())
    evals = []
    results_prefix = f"/home/ckchang/.cache/apxinf/xip/{TASK_HOME}/{task_name}/seed-{args.seed}"
    path_prefix = (
        f"{results_prefix}/online/{args.model}/ncores-{args.ncores}/ldnthreads-{args.loading_mode}/nparts-{args.nparts}"
    )
    eval_path = f"{path_prefix}/exact/evals_exact.json"

    keys = {"qinf": qinf, "policy": policy, "pest_nsamples": pest_nsamples,
            "agg_qids": agg_qids, "naggs": naggs, "model": args.model,
            "scheduler_init": 0, "scheduler_batch": 0,
            "max_error": 0.0, "min_conf": 1.0}
    evals.append(extract_result(keys, json.load(open(eval_path)), task_name, agg_qids))
    print(f"last eval: {evals[-1]}")

    for scheduler_init in [0, 1]:
        for scheduler_batch in [batch * naggs for batch in [1, 2, 5, 10, 30, 50, 70, 100]]:
            for max_error in max_errors:
                for min_conf in min_confs:
                    eval_path = f"{path_prefix}/ncfgs-{args.ncfgs}/pest-error-MC-{pest_nsamples}-{args.seed}/qinf-{qinf}/scheduler-{policy}-{scheduler_init}-{scheduler_batch}/evals_conf-0.05-{max_error}-{min_conf}-60.0-2048.0-1000.json"
                    if os.path.exists(eval_path):
                        keys = {"qinf": qinf, "policy": policy, "pest_nsamples": pest_nsamples,
                                "agg_qids": agg_qids, "naggs": naggs, "model": args.model,
                                "scheduler_init": scheduler_init, "scheduler_batch": scheduler_batch,
                                "max_error": max_error, "min_conf": min_conf}
                        evals.append(extract_result(keys, json.load(open(eval_path)), task_name, agg_qids, evals[0]["avg_latency"]))
                        print(f"last eval: {evals[-1]}")
    df = pd.DataFrame(evals)

    EVALS_HOME = "/home/ckchang/xip_evals"
    evals_dir = os.path.join(EVALS_HOME, f'seed-{args.seed}', f'mode-{args.loading_mode}')
    if not os.path.exists(evals_dir):
        os.makedirs(evals_dir, exist_ok=True)
    csv_path = f"{evals_dir}/{task_name}_{args.model}_{args.nparts}_{args.ncfgs}_{args.ncores}_evals.csv"
    df.to_csv(csv_path, index=False)
    return df


def get_base_cmd(args: ExpArgs, task_name: str, agg_qids: str):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter} eval_reg.py --seed {args.seed}"
    else:
        cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"

    cmd = f"{cmd} --interpreter {interpreter}  --task_home {args.task_home} --task_name {task_name} --agg_qids {agg_qids}"
    cmd = f"{cmd} --model {args.model} --nparts {args.nparts} --ncores {args.ncores} --loading_mode {args.loading_mode}"
    return cmd


def run_shared(args: ExpArgs, task_name: str, agg_qids: str):
    cmd = get_base_cmd(args, task_name, agg_qids)
    cmd = f"{cmd} --run_shared"
    if not args.skip_shared:
        os.system(cmd)


def run_eval_cmd(
    args: ExpArgs,
    task_name: str,
    agg_qids: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
    min_confs: list[float] = None,
):
    cmd = get_base_cmd(args, task_name, agg_qids)
    cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
    cmd = f"{cmd} --min_confs {' '.join([str(c) for c in min_confs])}"
    os.system(cmd)


def run_pipeline(args: ExpArgs, task_name: str, agg_qids: str,
                 max_errors: list[float]):
    """ run with different settings (scheduler_init, scheduler_batch, max_error, min_confs)
    1. default setting, (1, 1*naggs, 0.0, c), c \in [0.99, 0.98, 0.95]
    2. vary scheduler_batch, (1, n*naggs, 0.0, c), n \in [1, 2, 5, 10, 30, 50, 70, 100], c \in [0.99, 0.98, 0.95]
    3. vary max_error, (1, 1*naggs, e, c), e \in max_errors, c \in [0.99, 0.98, 0.95]
    4. vary min_confs, (1, 1*naggs, 0.0, c), c \in [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    5. sepecial scheduler_init, (0, n*naggs, 0.0, 0.95), n \in [1, 2, 5, 10, 30, 50, 70, 100]
    6. sepecial scheduler_init, (0, 1*naggs, 0.0, 0.99)
    """
    if args.collect_only:
        collect_results(args, task_name, agg_qids, max_errors)
        return

    model = args.model
    naggs = len(agg_qids.split())
    default_error = max_errors[0]
    run_shared(args, task_name, agg_qids)
    # 1. default setting
    run_eval_cmd(args, task_name, model, agg_qids, 1, 1*naggs, default_error, [0.99, 0.98, 0.95])

    if args.default_only:
        return

    # 2. vary scheduler_batch
    batches = [1, 2, 5, 10, 30, 50, 70, 100]
    for n in batches:
        run_eval_cmd(args, task_name, model, agg_qids, 1, n*naggs, default_error, [0.99, 0.98, 0.95])

    # 3. vary max_error
    for e in max_errors[1:]:
        run_eval_cmd(args, task_name, model, agg_qids, 1, 1*naggs, e, [0.99, 0.98, 0.95])

    # 4. vary min_confs
    min_confs = [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]:
    cmd = run_eval_cmd(args, task_name, agg_qids, 1, 1*naggs, default_error, min_confs)

    # 5. sepecial scheduler_init
    for n in batches:
        run_eval_cmd(args, task_name, model, agg_qids, 0, n*naggs, default_error, [0.95])

    # 6. sepecial scheduler_init
    run_eval_cmd(args, task_name, model, agg_qids, 0, 1*naggs, default_error, [0.99])


def run_machinery(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    run_pipeline(args, task_name, agg_qids, [0.0])


def run_machinerymulti(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinerymulti"
    agg_qids = "0 1 2 3 4 5 6 7"
    run_pipeline(args, task_name, agg_qids, [0.0])


def run_tdfraud(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraud"
    agg_qids = "1 2 3"
    run_pipeline(args, task_name, agg_qids, [0.0])


def run_tdfraudrandom(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudrandom"
    agg_qids = "1 2 3"
    run_pipeline(args, task_name, agg_qids, [0.0])


def run_tdfraudkaggle(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudkaggle"
    agg_qids = "1"
    run_pipeline(args, task_name, agg_qids, [0.0])


def run_tripsfeast(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsfeast"
    agg_qids = "1 2"
    max_errors = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, max_errors)
    run_pipeline(args, task_name, agg_qids, [0.5, 0.1])


def run_machinery_vary_nf(args: ExpArgs, nf: int):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = f"machineryxf{nf}"
    agg_qids = " ".join([f"{i}" for i in range(nf)])
    assert args.default_only is True, "must run default_only"
    run_pipeline(args, task_name, agg_qids, [0.0])


if __name__ == "__main__":
    args = ExpArgs().parse_args()
    if args.prep:
        run_prepare(args)
    elif args.exp == "machinery":
        run_machinery(args)
    elif args.exp == "tripsfeast":
        run_tripsfeast(args)
    elif args.exp == "machinerymulti":
        run_machinerymulti(args)
    elif args.exp in MachineryVaryXNF:
        nf = int(args.exp[len("machineryxf"):])
        run_machinery_vary_nf(args, nf)
    elif args.exp == "tdfraud":
        run_tdfraud(args)
    elif args.exp == "tdfraudrandom":
        run_tdfraudrandom(args)
    elif args.exp == "tdfraudkaggle":
        run_tdfraudkaggle(args)
    else:
        raise ValueError(f"invalid exp {args.exp}")