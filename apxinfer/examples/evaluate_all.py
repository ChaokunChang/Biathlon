from tap import Tap
import os


class ExpArgs(Tap):
    interpreter = "/home/ckchang/anaconda3/envs/apx/bin/python"
    exp: str = None
    model: str = None  # see each exp
    ncores: int = None  # 1, 0
    loading_mode: int = None  # 0, 1, 2, 5, 10
    nparts: int = 100
    ncfgs: int = 100
    seed: int = 0
    skip_shared: bool = False

    def process_args(self):
        assert self.exp is not None
        if self.exp != "prepare":
            assert self.model is not None
            assert self.ncores is not None
            assert self.loading_mode is not None


def get_scheduler_cfgs(args: ExpArgs, naggs: int = None):
    ncfgs = args.ncfgs
    if naggs is None:
        return [1, 5, 10, 20], [1, 5, 10, 20]
    if ncfgs == 100:
        inits = [1, 5, 10, 20]
        max_batch = naggs * ncfgs
        batches = [1, 5, 10, 20, 50]
        batches = batches + [i for i in range(100, max_batch + 1, 50)]
        return inits, batches
    else:
        raise ValueError(f"invalid ncfgs {ncfgs}")


def run_prepare(args: ExpArgs):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter}"
    else:
        cmd = f"sudo {interpreter}"
    for task in [
        "trips",
        "tick",
        "tickv2",
        "cheaptrips",
        "machinery",
        "ccfraud",
        "tripsfeast",
    ]:
        cmd = f"{cmd} prep.py --interpreter {interpreter} --task_name {task} --prepare_again --seed {args.seed}"
        os.system(cmd)


def get_base_cmd(args: ExpArgs, task_name: str, model: str, agg_qids: str):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter} eval_reg.py --seed {args.seed}"
    else:
        cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"

    cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
    cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores} --loading_mode {args.loading_mode}"
    cmd = f"{cmd} --run_shared"
    return cmd


def get_eval_cmd(
    args: ExpArgs,
    task_name: str,
    model: str,
    agg_qids: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter} eval_reg.py --seed {args.seed}"
    else:
        cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"

    cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
    cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores} --loading_mode {args.loading_mode}"
    cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
    return cmd


def run_machinery(args: ExpArgs):
    """
    must models = ["mlp", "dt", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = get_eval_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
            )
            os.system(cmd)


def run_cheaptrips(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptrips"
    agg_qids = "1 2 3"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = get_eval_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
            )
            os.system(cmd)


def run_ccfraud(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "xgb", "lgbm", "rf"]
    """
    task_name = "ccfraud"
    agg_qids = "3 4 5 6"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = get_eval_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
            )
            os.system(cmd)


def run_trips(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "trips"
    agg_qids = "1 2 3"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    max_errors = [0.5, 1.0, 2.0, 3.0]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = get_eval_cmd(
                    args,
                    task_name,
                    model,
                    agg_qids,
                    scheduler_init,
                    scheduler_batch,
                    max_error,
                )
                os.system(cmd)


def run_tripsfeast(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsfeast"
    agg_qids = "1 2"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    max_errors = [0.5, 1.0, 2.0, 3.0]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = get_eval_cmd(
                    args,
                    task_name,
                    model,
                    agg_qids,
                    scheduler_init,
                    scheduler_batch,
                    max_error,
                )
                os.system(cmd)


def run_cheaptripsfeast(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptripsfeast"
    agg_qids = "1 2"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = get_eval_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
            )
            os.system(cmd)


def run_tick_v1(args: ExpArgs):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = "tick"
    agg_qids = "1"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    max_errors = [0.001, 0.01, 0.05, 0.1]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = get_eval_cmd(
                    args,
                    task_name,
                    model,
                    agg_qids,
                    scheduler_init,
                    scheduler_batch,
                    max_error,
                )
                os.system(cmd)


def run_tick_v2(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickv2"
    agg_qids = "6"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    init_sizes, step_sizes = get_scheduler_cfgs(args, len(agg_qids))
    max_errors = [0.001, 0.01, 0.05, 0.1]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = get_eval_cmd(
                    args,
                    task_name,
                    model,
                    agg_qids,
                    scheduler_init,
                    scheduler_batch,
                    max_error,
                )
                os.system(cmd)


if __name__ == "__main__":
    args = ExpArgs().parse_args()
    if args.exp == "prepare":
        run_prepare(args)
    elif args.exp == "trips":
        run_trips(args)
    elif args.exp == "tick-v1":
        run_tick_v1(args)
    elif args.exp == "tick-v2":
        run_tick_v2(args)
    elif args.exp == "cheaptrips":
        run_cheaptrips(args)
    elif args.exp == "machinery":
        run_machinery(args)
    elif args.exp == "ccfraud":
        run_ccfraud(args)
    elif args.exp == "tripsfeast":
        run_tripsfeast(args)
    elif args.exp == "cheaptripsfeast":
        run_cheaptripsfeast(args)
    else:
        raise ValueError(f"invalid exp {args.exp}")
