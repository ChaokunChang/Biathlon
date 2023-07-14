import os
import pandas as pd
import joblib

from apxinfer.core.utils import XIPQType
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import MCPredictionEstimator
from apxinfer.core.qinfluence import XIPQInfEstimator, XIPQInfEstimatorByFInfs
from apxinfer.core.qcost import XIPQCostModel, QueryCostModel
from apxinfer.core.scheduler import XIPScheduler, XIPSchedulerGreedy
from apxinfer.core.scheduler import XIPSchedulerWQCost, XIPSchedulerRandom
from apxinfer.core.scheduler import XIPSchedulerUniform, XIPSchedulerBalancedQCost
from apxinfer.core.pipeline import XIPPipeline, XIPPipelineSettings
from apxinfer.core.config import OnlineArgs, DIRHelper, OfflineArgs

from apxinfer.core.online import OnlineExecutor

from apxinfer.examples.taxi.feature import get_fextractor


def load_model(args: OnlineArgs) -> XIPModel:
    model_path = DIRHelper.get_model_path(args)
    model = joblib.load(model_path)
    return model


def load_dataset(args: OnlineArgs, name: str, nreqs: int = 0) -> pd.DataFrame:
    dataset_dir = DIRHelper.get_dataset_dir(args)
    ds_path = os.path.join(dataset_dir, f"{name}_set.csv")
    dataset = pd.read_csv(ds_path)
    if nreqs > 0:
        dataset = dataset[:nreqs]
    return dataset


def load_xip_qcm(args: OnlineArgs) -> XIPQCostModel:
    ofl_args = OfflineArgs().from_dict({**args.as_dict(), "nreqs": 10})
    model_dir = DIRHelper.get_qcost_model_dir(ofl_args)
    model_path = os.path.join(model_dir, "xip_qcm.pkl")
    model: XIPQCostModel = joblib.load(model_path)
    return model


class TaxiOnlineArgs(OnlineArgs):
    plus: bool = False


if __name__ == "__main__":
    args = TaxiOnlineArgs().parse_args()

    # load test data
    test_set = load_dataset(args, "test", args.nreqs)
    verbose = args.verbose and len(test_set) <= 10

    # load xip model
    model = load_model(args)

    # create a feature extractor for this task
    fextractor = get_fextractor(
        args.nparts,
        args.seed,
        disable_sample_cache=args.disable_sample_cache,
        disable_query_cache=args.disable_query_cache,
        plus=args.plus,
        loading_nthreads=args.loading_nthreads,
    )

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
            point_pest=args.pest_point,
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
    else:
        raise ValueError("Invalid qinf estimator")

    # create qcost estimator for this task
    qcost_model = load_xip_qcm(args)

    # create a scheduler for this task
    if args.scheduler == "greedy":
        scheduler = XIPSchedulerGreedy(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 3)] * fextractor.num_queries,
            batch_size=args.scheduler_batch,
            verbose=verbose,
        )
    elif args.scheduler == "greedy_plus":
        scheduler = XIPSchedulerWQCost(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 3)] * fextractor.num_queries,
            batch_size=args.scheduler_batch,
            verbose=verbose,
        )
    elif args.scheduler == "random":
        scheduler = XIPSchedulerRandom(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 3)] * fextractor.num_queries,
            batch_size=args.scheduler_batch,
            verbose=verbose,
        )
    elif args.scheduler == "uniform":
        scheduler = XIPSchedulerUniform(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 3)] * fextractor.num_queries,
            batch_size=args.scheduler_batch,
            verbose=verbose,
        )
    elif args.scheduler == "blqcost":
        scheduler = XIPSchedulerBalancedQCost(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 3)] * fextractor.num_queries,
            batch_size=args.scheduler_batch,
            verbose=verbose,
        )
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
        fextractor=fextractor,
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
