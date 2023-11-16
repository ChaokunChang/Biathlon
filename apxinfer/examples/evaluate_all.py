from tap import Tap
import os

interpreter = "/home/ckchang/anaconda3/envs/apx/bin/python"


class ExpArgs(Tap):
    exp: str = None
    model: str = None  # see each exp
    ncores: int = None  # 1, 0
    loading_mode: int = None  # 0, 1, 2, 5, 10
    nparts: int = 100
    ncfgs: int = 100
    seed: int = 0

    def process_args(self):
        assert self.exp is not None
        if self.exp != "prepare":
            assert self.model is not None
            assert self.ncores is not None
            assert self.loading_mode is not None


def get_scheduler_cfgs(args: ExpArgs):
    ncfgs = args.ncfgs
    if ncfgs == 100:
        return [1, 5, 10, 20], [1, 5, 10, 20]
    else:
        raise ValueError(f'invalid ncfgs {ncfgs}')


def run_prepare(args: ExpArgs):
    for task in ['trips', 'tick', 'tickv2', 'machinery']:
        cmd = f"sudo {interpreter} prep.py --interpreter {interpreter} --task_name {task} --prepare_again --seed {args.seed}"
        os.system(cmd)


def run_machinery(args: ExpArgs):
    """
        must models = ["mlp", "dt", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    model = args.model
    init_sizes, step_sizes = get_scheduler_cfgs(args)
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"
            cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
            cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores}"
            cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error 0.0"
            os.system(cmd)


def run_cheaptrips(args: ExpArgs):
    """
        must models = ["xgb"]
        optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptrips"
    agg_qids = "1 2 3"
    model = args.model
    init_sizes, step_sizes = get_scheduler_cfgs(args)
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"
            cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
            cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores}"
            cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error 0.0"
            os.system(cmd)


def run_trips(args: ExpArgs):
    """
        must models = ["lgbm"]
        optional models = ["xgb", "dt", "rf"]
    """
    task_name = "trips"
    agg_qids = "1 2 3"
    model = args.model
    init_sizes, step_sizes = get_scheduler_cfgs(args)
    max_errors = [0.5, 1.0, 2.0, 3.0]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"
                cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
                cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores}"
                cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
                os.system(cmd)


def run_tick_v1(args: ExpArgs):
    """
        models = ["lr", "dt", "rf"]
    """
    task_name = "tick"
    agg_qids = "1"
    model = args.model
    init_sizes, step_sizes = get_scheduler_cfgs(args)
    max_errors = [0.001, 0.01, 0.05, 0.1]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"
                cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
                cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores}"
                cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
                os.system(cmd)


def run_tick_v2(args: ExpArgs):
    """
        must models = ["lr"]
        optional models = ["dt", "rf"]
    """
    task_name = "tickv2"
    agg_qids = "6"
    model = args.model
    init_sizes, step_sizes = get_scheduler_cfgs(args)
    max_errors = [0.001, 0.01, 0.05, 0.1]
    for scheduler_init in init_sizes:
        for scheduler_batch in step_sizes:
            for max_error in max_errors:
                cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"
                cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
                cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores}"
                cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
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
    else:
        raise ValueError(f'invalid exp {args.exp}')
