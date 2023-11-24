from tap import Tap
import importlib
import os
import joblib

from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.config import BaseXIPArgs
from apxinfer.core.config import PrepareArgs, TrainerArgs
from apxinfer.core.config import OfflineArgs, OnlineArgs

from apxinfer.core.festimator import XIPFeatureEstimator, XIPFeatureErrorEstimator
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import MCPredictionEstimator
from apxinfer.core.qinfluence import XIPQInfEstimator, XIPQInfEstimatorByFInfs
from apxinfer.core.qinfluence import XIPQInfEstimatorSobol, XIPQInfEstimatorSTIndex
from apxinfer.core.qcost import XIPQCostModel
from apxinfer.core.scheduler import XIPSchedulerGreedy, XIPSchedulerOptimizer
from apxinfer.core.scheduler import XIPSchedulerWQCost, XIPSchedulerRandom
from apxinfer.core.scheduler import XIPSchedulerUniform, XIPSchedulerBalancedQCost
from apxinfer.core.scheduler import XIPSchedulerGradient, XIPSchedulerStepGradient
from apxinfer.core.pipeline import XIPPipeline, XIPPipelineSettings

from apxinfer.core.offline import OfflineExecutor
from apxinfer.core.online import OnlineExecutor


def get_func(name: str):
    module_names = {
        "trips": ["data", "query", "engine"],
        "machinery": ["data", "query", "engine"],
        # add more module names for other values of name
    }

    functions = {}
    for mod in module_names[name]:
        module = importlib.import_module(f"apxinfer.examples.{name}.{mod}")
        functions.update(vars(module))


def get_fengine(name: str, args: BaseXIPArgs):
    if name == "ccfraud":
        from apxinfer.examples.ccfraud.engine import get_ccfraud_engine
        fengine = get_ccfraud_engine(nparts=args.nparts, ncores=args.ncores, verbose=args.verbose)
    elif name == "tick":
        from apxinfer.examples.tick.engine import get_tick_engine
        fengine = get_tick_engine(nparts=args.nparts, ncores=args.ncores, verbose=args.verbose)
    elif name == "tickv2":
        from apxinfer.examples.tick.engine import get_tick_engine_v2
        fengine = get_tick_engine_v2(nparts=args.nparts, ncores=args.ncores, verbose=args.verbose)
    elif name == "tripsfeast":
        from apxinfer.examples.tripsfeast.engine import get_trips_feast_engine
        fengine = get_trips_feast_engine(nparts=args.nparts, ncores=args.ncores, verbose=args.verbose)
    elif name == "cheaptripsfeast":
        from apxinfer.examples.tripsfeast.engine import get_trips_feast_engine
        fengine = get_trips_feast_engine(nparts=args.nparts, ncores=args.ncores, verbose=args.verbose)
    elif name == "trips":
        from apxinfer.examples.trips.data import get_dloader
        from apxinfer.examples.trips.query import get_qps
        from apxinfer.examples.trips.engine import get_qengine
        dloader = get_dloader(nparts=args.nparts, verbose=args.verbose)
        qps = get_qps(dloader, args.verbose, version=1)
        fengine = get_qengine(qps, args.ncores, args.verbose)
    elif name == "cheaptrips":
        from apxinfer.examples.trips.data import get_dloader
        from apxinfer.examples.trips.query import get_qps
        from apxinfer.examples.trips.engine import get_qengine
        dloader = get_dloader(nparts=args.nparts, verbose=args.verbose)
        qps = get_qps(dloader, args.verbose, version=1)
        fengine = get_qengine(qps, args.ncores, args.verbose)
    elif name == "machinery" or name == "machinerymulti":
        from apxinfer.examples.machinery.data import get_dloader
        from apxinfer.examples.machinery.query import get_qps
        from apxinfer.examples.machinery.engine import get_qengine
        dloader = get_dloader(nparts=args.nparts, verbose=args.verbose)
        qps = get_qps(dloader, args.verbose)
        fengine = get_qengine(qps, args.ncores, args.verbose)
    elif name in ([f"machineryf{i}" for i in range(1, 8)] + [f"machinerymultif{i}" for i in range(1, 8)]):
        from apxinfer.examples.machinery.data import get_dloader
        from apxinfer.examples.machinery.query import get_qps
        from apxinfer.examples.machinery.engine import get_qengine
        dloader = get_dloader(nparts=args.nparts, verbose=args.verbose)
        qps = get_qps(dloader, args.verbose, nf=int(name[-1]))
        fengine = get_qengine(qps, args.ncores, args.verbose)
    elif name in ([f"machineryxf{i}" for i in range(1, 8)] + [f"machinerymultixf{i}" for i in range(1, 8)]):
        from apxinfer.examples.machinery.data import get_dloader
        from apxinfer.examples.machinery.query import get_qps_x
        from apxinfer.examples.machinery.engine import get_qengine
        dloader = get_dloader(nparts=args.nparts, verbose=args.verbose)
        qps = get_qps_x(dloader, args.verbose, nf=int(name[-1]))
        fengine = get_qengine(qps, args.ncores, args.verbose)

    for qry in fengine.queries:
        fest = XIPFeatureEstimator(err_module=XIPFeatureErrorEstimator(min_support=args.err_min_support,
                                                                       seed=args.seed,
                                                                       bs_type=args.bs_type,
                                                                       bs_nresamples=args.bs_nresamples,
                                                                       bs_max_nthreads=args.bs_nthreads,
                                                                       bs_feature_correction=args.bs_feature_correction,
                                                                       bs_bias_correction=args.bs_bias_correction,
                                                                       bs_for_var_std=args.bs_for_var_std))
        qry.set_estimator(fest)
        qry.set_loading_mode(args.loading_mode)
    return fengine


def run_ingest(name: str, args: BaseXIPArgs):
    if name == "trips" or name == "tripsfeast":
        from apxinfer.examples.trips.data import get_ingestor
        ingestor = get_ingestor(nparts=args.nparts, seed=args.seed)
        ingestor.run()
    elif name == "cheaptrips" or name == "cheaptripsfeast":
        from apxinfer.examples.trips.data import get_ingestor
        ingestor = get_ingestor(nparts=args.nparts, seed=args.seed)
        ingestor.run()
    elif name.startswith("machinery"):
        from apxinfer.examples.machinery.data import get_ingestor
        ingestor = get_ingestor(nparts=args.nparts, seed=args.seed)
        ingestor.run()
    elif name == "ccfraud":
        from apxinfer.examples.ccfraud.data import ingest
        ingest(nparts=args.nparts, seed=args.seed, verbose=args.verbose)
    elif name == "tick":
        from apxinfer.examples.tick.data import ingest
        ingest(nparts=args.nparts, seed=args.seed, verbose=args.verbose)


def run_prepare(name: str, args: PrepareArgs):
    if name == "trips" or name == "tripsfeast":
        from apxinfer.examples.trips.prepare import TripsPrepareWorker as Worker
        model_type = "regressor"
    elif name == "cheaptrips" or name == "cheaptripsfeast":
        from apxinfer.examples.cheaptrips.prepare import CheapTripsPrepareWorker as Worker
        model_type = "classifier"
    elif name == "machinery" or (name in [f"machineryf{i}" for i in range(1, 8)]) or (name in [f"machineryxf{i}" for i in range(1, 8)]):
        from apxinfer.examples.machinery.prepare import MachineryBinaryClassPrepareWorker as Worker
        model_type = "classifier"
    elif name == "machinerymulti" or (name in [f"machinerymultif{i}" for i in range(1, 8)]) or (name in [f"machinerymultixf{i}" for i in range(1, 8)]):
        from apxinfer.examples.machinery.prepare import MachineryMultiClassPrepareWorker as Worker
        model_type = "classifier"
    elif name == "ccfraud":
        from apxinfer.examples.ccfraud.prepare import CCFraudPrepareWorker as Worker
        model_type = "classifier"
    elif name == "tick" or name == "tickv2":
        from apxinfer.examples.tick.prepare import TickPrepareWorker as Worker
        model_type = "regressor"

    worker: XIPPrepareWorker = Worker(
        DIRHelper.get_prepare_dir(args),
        get_fengine(name, args),
        args.max_requests,
        args.train_ratio,
        args.valid_ratio,
        model_type,
        args.model,
        args.seed,
        args.nparts
    )
    worker.run(args.skip_dataset)


def run_trainer(name: str, args: TrainerArgs):
    if name == "trips" or name == "tripsfeast":
        from apxinfer.examples.trips.trainer import TripsTrainer as Trainer
        model_type = "regressor"
    elif name == "cheaptrips" or name == "cheaptripsfeast":
        from apxinfer.examples.cheaptrips.trainer import CheapTripsTrainer as Trainer
        model_type = "classifier"
    elif name.startswith("machinery"):
        from apxinfer.examples.machinery.trainer import MachineryTrainer as Trainer
        model_type = "classifier"
    elif name == "ccfraud":
        from apxinfer.examples.ccfraud.trainer import CCFraudTrainer as Trainer
        model_type = "classifier"
    elif name == "tick" or name == "tickv2":
        from apxinfer.examples.tick.trainer import TickTrainer as Trainer
        model_type = "regressor"

    trainer = Trainer(
        DIRHelper.get_prepare_dir(args),
        model_type,
        args.model,
        args.seed,
        scaler_type=args.scaler_type,
        multi_class=(name == "machinerymulti")
    )
    trainer.run()


def run_offline(name: str, args: OfflineArgs):
    # load test data
    test_set = LoadingHelper.load_dataset(args, "valid", args.nreqs)
    verbose = args.verbose and len(test_set) <= 10

    # load xip model
    model: XIPModel = LoadingHelper.load_model(args)

    # create a feature engine for this task
    fengine = get_fengine(name, args)

    executor = OfflineExecutor(
        working_dir=DIRHelper.get_offline_dir(args),
        fextractor=fengine,
        model=model,
        nparts=args.nparts,
        ncfgs=args.ncfgs,
        verbose=verbose,
    )
    executor.run(test_set, args.clear_cache)


def load_xip_qcm(args: OnlineArgs) -> XIPQCostModel:
    ofl_args = OfflineArgs().from_dict({**args.as_dict(), "nreqs": args.offline_nreqs})
    model_dir = DIRHelper.get_qcost_model_dir(ofl_args)
    model_path = os.path.join(model_dir, "xip_qcm.pkl")
    model: XIPQCostModel = joblib.load(model_path)
    return model


def run_online(name: str, args: OnlineArgs):
    # load test data
    test_set = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    verbose = args.verbose and len(test_set) <= 10

    # load xip model
    model: XIPModel = LoadingHelper.load_model(args)

    # create a feature engine for this task
    fengine = get_fengine(name, args)

    # create a prediction estimator for this task
    if args.pest == "MC":
        constraint = args.pest_constraint
        if constraint == "conf":
            constraint_value = args.min_conf
        elif constraint == "error":
            constraint_value = args.max_error
        elif constraint == "relative_error":
            constraint_value = args.max_relative_error
        pred_estimator = MCPredictionEstimator(
            constraint_type=constraint,
            constraint_value=constraint_value,
            seed=args.pest_seed,
            n_samples=args.pest_nsamples,
            pest_point=args.pest_point,
            verbose=verbose,
        )
    else:
        raise ValueError("Invalid prediction estimator")

    # create qinf estimator for this task
    if args.qinf == "direct":
        qinf_estimator = XIPQInfEstimator(
            pred_estimator=pred_estimator, verbose=verbose
        )
    elif args.qinf == "by_finf":
        qinf_estimator = XIPQInfEstimatorByFInfs(
            pred_estimator=pred_estimator, verbose=verbose
        )
    elif args.qinf == "sobol":
        qinf_estimator = XIPQInfEstimatorSobol(
            pred_estimator=pred_estimator, verbose=verbose
        )
    elif args.qinf == "sobolT":
        qinf_estimator = XIPQInfEstimatorSTIndex(
            pred_estimator=pred_estimator, verbose=verbose
        )
    else:
        raise ValueError("Invalid qinf estimator")

    # create qcost estimator for this task
    qcost_model = load_xip_qcm(args)

    # create a scheduler for this task
    step_sizes = [round(1.0 / args.ncfgs, 3)] * fengine.num_queries
    scheduler_args = {
        "fextractor": fengine,
        "model": model,
        "pred_estimator": pred_estimator,
        "qinf_estimator": qinf_estimator,
        "qcost_estimator": qcost_model,
        "sample_grans": step_sizes,
        "min_qsamples": [args.scheduler_init * i for i in step_sizes], 
        "batch_size": args.scheduler_batch,
        "min_card": args.err_min_support,
        "verbose": verbose,
    }
    if args.scheduler == "greedy":
        scheduler = XIPSchedulerGreedy(**scheduler_args)
    elif args.scheduler == "greedy_plus":
        scheduler = XIPSchedulerWQCost(**scheduler_args)
    elif args.scheduler == "random":
        scheduler = XIPSchedulerRandom(**scheduler_args)
    elif args.scheduler == "uniform":
        scheduler = XIPSchedulerUniform(**scheduler_args)
    elif args.scheduler == "blqcost":
        scheduler = XIPSchedulerBalancedQCost(**scheduler_args)
    elif args.scheduler == "optimizer":
        scheduler = XIPSchedulerOptimizer(**scheduler_args)
    elif args.scheduler == "gradient":
        scheduler = XIPSchedulerGradient(**scheduler_args)
    elif args.scheduler == "stepgradient":
        scheduler = XIPSchedulerStepGradient(**scheduler_args)
    else:
        raise ValueError("Invalid scheduler")

    # create a pipeline for this task
    ppl_settings = XIPPipelineSettings(
        termination_condition=args.termination_condition,
        max_relative_error=args.max_relative_error,
        max_error=args.max_error,
        min_conf=args.min_conf,
        max_time=args.max_time,
        max_memory=args.max_memory,
        max_rounds=args.max_rounds,
    )
    ppl = XIPPipeline(
        fextractor=fengine,
        model=model,
        pred_estimator=pred_estimator,
        scheduler=scheduler,
        settings=ppl_settings,
        verbose=verbose,
    )

    # run pipline to serve online requests
    online_dir = DIRHelper.get_online_dir(args)
    executor = OnlineExecutor(ppl=ppl, working_dir=online_dir, verbose=verbose)
    executor.run(test_set, args.exact)


class RunArgs(Tap):
    example: str = "trips"
    stage: str = "prepare"


if __name__ == "__main__":
    args = RunArgs().parse_args(known_only=True)
    print(f"args= {args}")
    if args.stage == "ingest":
        staget_args = BaseXIPArgs().parse_args(known_only=True)
        run_ingest(args.example, staget_args)
    elif args.stage == "prepare":
        staget_args = PrepareArgs().parse_args(known_only=True)
        # print(f"run {args.stage} with {staget_args}")
        run_prepare(args.example, staget_args)
    elif args.stage == "train":
        staget_args = TrainerArgs().parse_args(known_only=True)
        # print(f"run {args.stage} with {staget_args}")
        run_trainer(args.example, staget_args)
    elif args.stage == "offline":
        staget_args = OfflineArgs().parse_args(known_only=True)
        # print(f"run {args.stage} with {staget_args}")
        run_offline(args.example, staget_args)
    elif args.stage == "online":
        staget_args = OnlineArgs().parse_args(known_only=True)
        # print(f"run {args.stage} with {staget_args}")
        run_online(args.example, staget_args)
    else:
        raise ValueError(f"unsupported stage {args.stage}")
