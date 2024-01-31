from tap import Tap
import os

StudentQNo = [f"studentqno{i}" for i in range(1, 19)]
StudentQNo18VaryNF = [f"studentqno18nf{i}" for i in range(1, 13)]
MachineryVaryNF = [f"machinerynf{i}" for i in range(1, 8)]
MachineryVaryXNF = [f"machineryxf{i}" for i in range(1, 9)]
MachineryMultiVaryNF = [f"machinerymultif{i}" for i in range(1, 8)]
MachineryMultiVaryXNF = [f"machinerymultixf{i}" for i in range(1, 9)]
TickVaryNMonths = [f"tickvaryNM{i}" for i in range(1, 30)]
TripsFeastVaryWindow = [f"tripsfeastw{i}" for i in range(1, 1000)]


class ExpArgs(Tap):
    interpreter = "/home/ckchang/anaconda3/envs/apx/bin/python"
    exp: str = None
    model: str = None  # see each exp
    ncores: int = 1  # 1, 0
    loading_mode: int = 0  # 0, 1, 2, 5, 10
    nparts: int = 100
    ncfgs: int = 100
    seed: int = 0
    skip_shared: bool = False
    prepare: bool = False
    warmup: bool = False

    def process_args(self):
        assert self.exp is not None
        if not self.prepare:
            assert self.model is not None


def get_default_min_confs(args: ExpArgs):
    return [0.95, 0.98]


def get_min_confs(args: ExpArgs):
    # return [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    return [0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]


def get_default_alphas(args: ExpArgs):
    return [5]


def get_default_betas(args: ExpArgs):
    return [1]


def get_all_quantiles(args: ExpArgs):
    return [1, 2, 3, 5] + [i for i in range(10, 100, 20)] + [100]


def get_default_scheduler_cfgs(args: ExpArgs, naggs: int):
    for alpha in get_default_alphas(args):
        for beta in get_default_betas(args):
            cfgs = [(alpha, beta * naggs)]
    return cfgs


def get_scheduler_cfgs(args: ExpArgs, naggs: int):
    quantiles = get_all_quantiles(args)
    default_alphas = get_default_alphas(args)
    default_betas = get_default_betas(args)
    cfgs = []

    # default apha and vary beta
    for alpha in default_alphas:
        for beta in quantiles:
            cfgs.append((alpha, beta * naggs))

    # default beta and vary alpha
    for beta in default_betas:
        for alpha in quantiles[:-1]:
            cfgs.append((alpha, beta * naggs))
    return cfgs


def list_to_option_str(values: list):
    return " ".join([f"{v}" for v in values])


def run_prepare(args: ExpArgs):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter}"
    else:
        cmd = f"sudo {interpreter}"
    cmd = f"{cmd} prep.py --interpreter {interpreter} --task_name {args.exp} --prepare_again --seed {args.seed}"
    os.system(cmd)


def get_base_cmd(args: ExpArgs, task_name: str, model: str, agg_qids: str):
    interpreter = args.interpreter
    if interpreter == "python":
        cmd = f"{interpreter} eval_reg.py --seed {args.seed}"
    else:
        cmd = f"sudo {interpreter} eval_reg.py --seed {args.seed}"

    cmd = f"{cmd} --interpreter {interpreter} --task_name {task_name} --agg_qids {agg_qids}"
    cmd = f"{cmd} --model {model} --nparts {args.nparts} --ncores {args.ncores} --loading_mode {args.loading_mode}"
    return cmd


def get_baseline_cmd(args: ExpArgs, task_name: str, model: str, agg_qids: str):
    cmd = get_base_cmd(args, task_name, model, agg_qids)
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
    cmd = get_base_cmd(args, task_name, model, agg_qids)
    cmd = f"{cmd} --scheduler_init {scheduler_init} --scheduler_batch {scheduler_batch} --max_error {max_error}"
    return cmd


def run_pipeline(
    args: ExpArgs,
    task_name: str,
    agg_qids: str,
    default_max_errors: list[float],
    max_errors: list[float],
    default_only: bool = False,
):
    naggs = len(agg_qids.split(" "))
    model = args.model

    if args.warmup:
        cmd = get_eval_cmd(args, task_name, model, agg_qids, 1000, 1000 * naggs, 0.0)
        cmd = f"{cmd} --min_confs 1.1 --nocache"
        os.system(cmd)

    # run offline and baseline
    if not args.skip_shared:
        cmd = get_baseline_cmd(args, task_name, model, agg_qids)
        os.system(cmd)

    default_min_confs_str = list_to_option_str(get_default_min_confs(args))
    min_confs_str = list_to_option_str(get_min_confs(args))

    default_cfgs = get_default_scheduler_cfgs(args, naggs)
    cfgs = get_scheduler_cfgs(args, naggs)

    # default only
    for default_init, default_batch in default_cfgs:
        for max_error in default_max_errors:
            cmd = get_eval_cmd(
                args,
                task_name,
                model,
                agg_qids,
                default_init,
                default_batch,
                max_error,
            )
            cmd = f"{cmd} --min_confs {default_min_confs_str}"
            os.system(cmd)
    if default_only:
        return

    # vary max_error only
    for default_init, default_batch in default_cfgs:
        for max_error in max_errors:
            cmd = get_eval_cmd(
                args,
                task_name,
                model,
                agg_qids,
                default_init,
                default_batch,
                max_error,
            )
            cmd = f"{cmd} --min_confs {default_min_confs_str}"
            os.system(cmd)

    # vary cfgs only
    for scheduler_init, scheduler_batch in cfgs:
        for max_error in default_max_errors:
            cmd = get_eval_cmd(
                args,
                task_name,
                model,
                agg_qids,
                scheduler_init,
                scheduler_batch,
                max_error,
            )
            cmd = f"{cmd} --min_confs {default_min_confs_str}"
            os.system(cmd)

    # vary min_conf
    for default_init, default_batch in default_cfgs:
        for max_error in default_max_errors:
            cmd = get_eval_cmd(
                args,
                task_name,
                model,
                agg_qids,
                default_init,
                default_batch,
                max_error,
            )
            cmd = f"{cmd} --min_confs {min_confs_str}"
            os.system(cmd)


def run_studentqno(args: ExpArgs, qno: int):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = f"studentqno{qno}"
    agg_qids = list_to_option_str([i for i in range(13)])
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_student(args: ExpArgs):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = "student"
    agg_qids = list_to_option_str([i + 1 for i in range(13)])
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machinery(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinery"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_machinerymulti(args: ExpArgs):
    """
    must models = ["mlp", "svm", "knn"]
    """
    task_name = "machinerymulti"
    agg_qids = "0 1 2 3 4 5 6 7"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_cheaptrips(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptrips"
    agg_qids = "1 2 3"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_ccfraud(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "xgb", "lgbm", "rf"]
    """
    task_name = "ccfraud"
    agg_qids = "3 4 5 6"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tdfraud(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraud"
    agg_qids = "1 2 3"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tdfraudrandom(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudrandom"
    agg_qids = "1 2 3"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tdfraudkaggle(args: ExpArgs):
    """
    must models = ["xgb"]
    """
    task_name = "tdfraudkaggle"
    agg_qids = "1"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_trips(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "trips"
    agg_qids = "1 2 3"
    default_max_errors = [0.1, 0.5, 1.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tripsfeast(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "dt", "rf"]
    """
    task_name = "tripsfeast"
    agg_qids = "1 2"
    default_max_errors = [0.1, 0.5, 1.0]
    max_errors = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_battery(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "battery"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [120, 300]
    max_errors = [60, 120, 300, 600, 900, 1200, 3600, 7200]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_batteryv2(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "batteryv2"
    agg_qids = "0 1 2 3 4"
    default_max_errors = [60, 120]
    max_errors = [30, 60, 120, 300, 600, 900, 1200, 1800, 2400, 3000, 3600, 4800, 7200]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofan(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofan"
    naggs = 9
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [1, 3, 6]
    max_errors = [1, 3, 6, 10, 20, 50, 80, 100]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_turbofanall(args: ExpArgs):
    """
    must models = ["lgbm"]
    optional models = ["xgb", "rf"]
    """
    task_name = "turbofanall"
    naggs = 44
    agg_qids = list_to_option_str([i for i in range(naggs)])
    default_max_errors = [1, 3, 6]
    max_errors = [1, 3, 6, 10, 20, 50, 80, 100]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_cheaptripsfeast(args: ExpArgs):
    """
    must models = ["xgb"]
    optional models = ["dt", "lgbm", "rf"]
    """
    task_name = "cheaptripsfeast"
    agg_qids = "1 2"
    default_max_errors = [0]
    max_errors = [0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_v1(args: ExpArgs):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = "tick"
    agg_qids = "1"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_v2(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickv2"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_studentqno18_vary_nf(args: ExpArgs, nf: int):
    """
    models = [lgbm, gbm, tfgbm]
    """
    task_name = f"studentqno18nf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_machinery_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machineryxf{nf}"
    else:
        task_name = f"machinerynf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_machinerymulti_vary_nf(args: ExpArgs, nf: int, fixed: bool = False):
    """
    must models = ["mlp", "svm", "knn"]
    """
    if fixed:
        task_name = f"machinerymultixf{nf}"
    else:
        task_name = f"machinerymultinf{nf}"
    agg_qids = list_to_option_str([i for i in range(nf)])
    run_pipeline(args, task_name, agg_qids, [0.0], [0.0], default_only=True)


def run_tick_vary_nmonths(args: ExpArgs, nmonths: int):
    """
    models = ["lr", "dt", "rf"]
    """
    task_name = f"tickvaryNM{nmonths}"
    agg_qids = "6"
    run_pipeline(
        args, task_name, agg_qids, [0.01, 0.05], [0.01, 0.05], default_only=True
    )


def run_tripsfeast_vary_window_size(args: ExpArgs, nmonths: int):
    """
    models = ["lgbm", "xgb", "dt", "rf"]
    """
    task_name = f"tripsfeastw{nmonths}"
    agg_qids = "1 2"
    run_pipeline(args, task_name, agg_qids, [1.0, 1.66], [1.0, 1.66], default_only=True)


def run_tick_price(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickvaryNM1"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_tick_price_middle(args: ExpArgs):
    """
    must models = ["lr"]
    optional models = ["dt", "rf"]
    """
    task_name = "tickvaryNM8"
    agg_qids = "6"
    default_max_errors = [0.01, 0.05]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(args, task_name, agg_qids, default_max_errors, max_errors)


def run_extreme_tick_price(args: ExpArgs):
    """
    must models = ["lr"]
    """
    task_name = "tickvaryNM8"
    agg_qids = "6"
    default_max_errors = [0.0001, 0.0005]
    max_errors = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    run_pipeline(
        args, task_name, agg_qids, default_max_errors, max_errors, default_only=True
    )


def run_vary_nsamples(args: ExpArgs):
    """
    args.exp = varynsamples-{task_name}
    """
    task_name = args.exp.split("-")[1]
    agg_qids = None
    if task_name == "tripsfeast":
        agg_qids = "1 2"
        naggs = 2
        default_max_errors = [1.0, 1.66]
    elif task_name in ["tickv2", "tickvaryNM1"]:
        agg_qids = "6"
        naggs = 1
        default_max_errors = [0.01, 0.05]
    elif task_name == "machinery":
        agg_qids = "0 1 2 3 4 5 6 7"
        naggs = 8
        default_max_errors = [0.0]
    elif task_name == "machinerymulti":
        agg_qids = "0 1 2 3 4 5 6 7"
        naggs = 8
        default_max_errors = [0.0]
    elif task_name == "tdfraud":
        agg_qids = "1 2 3"
        naggs = 3
        default_max_errors = [0.0]
    else:
        raise ValueError(f"invalid task_name {task_name}")
    model = args.model

    default_cfgs = get_default_scheduler_cfgs(args, naggs)
    default_min_confs_str = list_to_option_str(get_default_min_confs(args))
    nsamples_list = [50, 100, 256, 500, 768, 1000, 1024, 2048]
    for scheduler_init, scheduler_batch in default_cfgs:
        for max_error in default_max_errors:
            for nsamples in nsamples_list:
                cmd = get_eval_cmd(
                    args,
                    task_name,
                    model,
                    agg_qids,
                    scheduler_init,
                    scheduler_batch,
                    max_error,
                )
                cmd = f"{cmd} --min_confs {default_min_confs_str} --pest_nsamples {nsamples}"
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
    elif args.exp == "machinerymulti":
        run_machinerymulti(args)
    elif args.exp in StudentQNo18VaryNF:
        nf = int(args.exp[len("studentqno18nf") :])
        run_studentqno18_vary_nf(args, nf)
    elif args.exp in MachineryVaryNF:
        nf = int(args.exp[len("machinerynf") :])
        run_machinery_vary_nf(args, nf)
    elif args.exp in MachineryMultiVaryNF:
        nf = int(args.exp[len("machinerymultinf") :])
        run_machinerymulti_vary_nf(args, nf)
    elif args.exp in MachineryVaryXNF:
        nf = int(args.exp[len("machineryxf") :])
        run_machinery_vary_nf(args, nf, fixed=True)
    elif args.exp in MachineryMultiVaryXNF:
        nf = int(args.exp[len("machinerymultixf") :])
        run_machinerymulti_vary_nf(args, nf, fixed=True)
    elif args.exp in TickVaryNMonths:
        nmonths = int(args.exp[len("tickvaryNM") :])
        run_tick_vary_nmonths(args, nmonths)
    elif args.exp in TripsFeastVaryWindow:
        nmonths = int(args.exp[len("tripsfeastw") :])
        run_tripsfeast_vary_window_size(args, nmonths)
    elif args.exp == "tickprice":
        run_tick_price(args)
    elif args.exp == "tdfraud":
        run_tdfraud(args)
    elif args.exp == "tdfraudrandom":
        run_tdfraudrandom(args)
    elif args.exp.startswith("varynsamples"):
        run_vary_nsamples(args)
    elif args.exp == "tdfraudkaggle":
        run_tdfraudkaggle(args)
    elif args.exp == "tickpricemiddle":
        run_tick_price_middle(args)
    elif args.exp == "extremetickprice":
        run_extreme_tick_price(args)
    elif args.exp == "battery":
        run_battery(args)
    elif args.exp == "batteryv2":
        run_batteryv2(args)
    elif args.exp == "turbofan":
        run_turbofan(args)
    elif args.exp == "turbofanall":
        run_turbofanall(args)
    elif args.exp == "student":
        run_student(args)
    elif args.exp.startswith("studentqno"):
        qno = int(args.exp[len("studentqno") :])
        run_studentqno(args, qno)
    else:
        raise ValueError(f"invalid exp {args.exp}")
