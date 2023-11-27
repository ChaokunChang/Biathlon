from tap import Tap
import os

MachineryVaryNF = [f"machineryf{i}" for i in range(1, 8)]
MachineryVaryXNF = [f"machineryxf{i}" for i in range(1, 9)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)]
MachineryMultiVaryXNF = [f"machinerymultixf{i}" for i in range(1, 9)]
TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 1000)]


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
    prep_single: str = None

    def process_args(self):
        assert self.exp is not None
        if self.exp != "prepare":
            assert self.model is not None
            assert self.ncores is not None
            assert self.loading_mode is not None


def get_scheduler_cfgs(args: ExpArgs, naggs: int = None):
    ncfgs = args.ncfgs
    if naggs is None:
        inits, batches = [1, 5, 10, 20], [1, 5, 10, 20]
    elif ncfgs == 100:
        inits = [1, 5, 10, 20]
        max_batch = naggs * ncfgs
        batches = [1, 5, 10, 20, 50]
        batches = batches + [i for i in range(100, max_batch + 1, 50)]
    else:
        raise ValueError(f"invalid ncfgs {ncfgs}")
    # create scheduler configs as pairs of (init, batch)
    cfgs = []
    for init in inits:
        for batch in batches:
            cfgs.append((init, batch))
    # move (5, 5), (5, 10), (10, 10), (10, 5) to the front
    cfgs = [(5, 5), (5, 10), (10, 10), (10, 5)] + [cfg for cfg in cfgs if cfg not in [(5, 5), (5, 10), (10, 10), (10, 5)]]
    cfgs = [(5, 5*naggs), (5, 3*naggs), (5, 2*naggs), (5, 1*naggs), (3, 3*naggs), (2, 2*naggs), (1, 1*naggs)] + cfgs
    return cfgs


def run_prepare(args: ExpArgs):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter}"
    else:
        cmd = f"sudo {interpreter}"
    if args.prep_single:
        tasks = [args.prep_single]
    else:
        tasks = [
            "trips",
            "tick",
            "tickv2",
            "cheaptrips",
            "machinery",
            "ccfraud",
            "tripsfeast",
            "machinerymulti",
            "tdfraud"
        ]
    for task in tasks:
        cmd = f"{cmd} prep.py --interpreter {interpreter} --task_name {task} --prepare_again --seed {args.seed}"
        if task in TickVaryNMonths:
            cmd = f"{cmd} --all_nparts 100"
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
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 8)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_machinerymulti(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinerymulti"
    agg_qids = "0 1 2 3 4 5 6 7"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 8)
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 3)
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 4)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_tdfraud(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraud"
    agg_qids = "1 2 3"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 4)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_tdfraudrandom(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudrandom"
    agg_qids = "1 2 3"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 4)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_tdfraudkaggle(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudkaggle"
    agg_qids = "1"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 4)
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 3)
    max_errors = [0.5, 1.0, 2.0, 3.0]
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 2)
    max_errors = [0.5, 1.0, 1.66, 2.0, 3.0]
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 2)
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 1)
    max_errors = [0.001, 0.01, 0.05, 0.1]
    for scheduler_init, scheduler_batch in cfgs:
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
    cfgs = get_scheduler_cfgs(args, 1)
    max_errors = [0.001, 0.01, 0.04, 0.05, 0.1]
    for scheduler_init, scheduler_batch in cfgs:
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


def get_scheduler_vary_cfgs(args: ExpArgs, naggs: int):
    cfgs = [(5, 5*naggs), (5, 3*naggs), (5, 1*naggs), (3, 3*naggs), (1, 1*naggs)]
    cfgs += [(5, 5), (5, 10), (5, 3), (5, 1)]
    return cfgs


def get_eval_vary_cmd(
    args: ExpArgs,
    task_name: str,
    model: str,
    agg_qids: str,
    scheduler_init: int,
    scheduler_batch: int,
    max_error: float,
):
    cmd = get_eval_cmd(
        args, task_name, model, agg_qids, scheduler_init, scheduler_batch, max_error
    )
    cmd = f"{cmd} --min_confs 0.95"
    return cmd


def run_machinery_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machineryxf{nf}"
    else:
        task_name = f"machineryf{nf}"
    agg_qids = " ".join([f"{i}" for i in range(nf)])
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)

    cfgs = get_scheduler_vary_cfgs(args, nf)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_vary_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_machinerymulti_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machinerymultixf{nf}"
    else:
        task_name = f"machinerymultif{nf}"
    agg_qids = " ".join([f"{i}" for i in range(nf)])
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)

    cfgs = get_scheduler_vary_cfgs(args, nf)
    for scheduler_init, scheduler_batch in cfgs:
        cmd = get_eval_vary_cmd(
            args, task_name, model, agg_qids, scheduler_init, scheduler_batch, 0.0
        )
        os.system(cmd)


def run_tick_vary_nmonths(args: ExpArgs, nmonths: int):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = f"tickvaryNM{nmonths}"
    agg_qids = "6"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)

    cfgs = get_scheduler_vary_cfgs(args, 1)
    max_errors = [0.01, 0.04]
    for scheduler_init, scheduler_batch in cfgs:
        for max_error in max_errors:
            cmd = get_eval_vary_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, max_error
            )
            os.system(cmd)


def run_tripsfeast_vary_window_size(args: ExpArgs, nmonths: int):
    """
    models = ["lgbm", "xgb", "dt", "rf"]
    """
    task_name = f"tripsfeastw{nmonths}"
    agg_qids = "1 2"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)

    cfgs = get_scheduler_vary_cfgs(args, 2)
    max_errors = [1.0, 1.66]
    for scheduler_init, scheduler_batch in cfgs:
        for max_error in max_errors:
            cmd = get_eval_vary_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, max_error
            )
            os.system(cmd)


def run_tick_price(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickvaryNM1"
    agg_qids = "6"
    model = args.model
    if not args.skip_shared:
        cmd = get_base_cmd(args, task_name, model, agg_qids)
        os.system(cmd)
    cfgs = get_scheduler_cfgs(args, 1)
    max_errors = [0.001, 0.01, 0.04, 0.05, 0.1]
    for scheduler_init, scheduler_batch in cfgs:
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


def run_vary_nsamples(args: ExpArgs):
    """
    args.exp = varynsamples-{task_name}
    """
    task_name = args.exp.split('-')[1]
    agg_qids = None
    if task_name == "tripsfeast":
        agg_qids = "1 2"
        max_errors = [1.0, 1.66]
    elif task_name in ["tickv2", "tickvaryNM1"]:
        agg_qids = "6"
        max_errors = [0.01, 0.04]
    elif task_name == "machinery":
        agg_qids = "0 1 2 3 4 5 6 7"
        max_errors = [0.0]
    elif task_name == "machinerymulti":
        agg_qids = "0 1 2 3 4 5 6 7"
        max_errors = [0.0]
    elif task_name == "tdfraud":
        agg_qids = "1 2 3"
        max_errors = [0.0]
    else:
        raise ValueError(f"invalid task_name {task_name}")
    model = args.model
    cfgs = get_scheduler_vary_cfgs(args, 1)
    for scheduler_init, scheduler_batch in cfgs:
        for max_error in max_errors:
            cmd = get_eval_vary_cmd(
                args, task_name, model, agg_qids, scheduler_init, scheduler_batch, max_error
            )
            for nsamples in [100, 1000, 5000, 10000, 50000]:
                os.system(f"{cmd} --pest_nsamples {nsamples}")


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
    elif args.exp == "machinerymulti":
        run_machinerymulti(args)
    elif args.exp in MachineryVaryNF:
        nf = int(args.exp[len("machineryf"):])
        run_machinery_vary_nf(args, nf)
    elif args.exp in MachineryMultiVaryNF:
        nf = int(args.exp[len("machinerymultif"):])
        run_machinerymulti_vary_nf(args, nf)
    elif args.exp in MachineryVaryXNF:
        nf = int(args.exp[len("machineryxf"):])
        run_machinery_vary_nf(args, nf, fixed=True)
    elif args.exp in MachineryMultiVaryXNF:
        nf = int(args.exp[len("machinerymultixf"):])
        run_machinerymulti_vary_nf(args, nf, fixed=True)
    elif args.exp in TickVaryNMonths:
        nmonths = int(args.exp[len("tickvaryNM"):])
        run_tick_vary_nmonths(args, nmonths)
    elif args.exp in TripsFeastVaryWindow:
        nmonths = int(args.exp[len("tripsfeastw"):])
        run_tripsfeast_vary_window_size(args, nmonths)
    elif args.exp == "tickprice":
        run_tick_price(args)
    elif args.exp == "tdfraud":
        run_tdfraud(args)
    elif args.exp == "tdfraudrandom":
        run_tdfraudrandom(args)
    elif args.exp.startswith('varynsamples'):
        run_vary_nsamples(args)
    elif args.exp == "tdfraudkaggle":
        run_tdfraudkaggle(args)
    else:
        raise ValueError(f"invalid exp {args.exp}")
