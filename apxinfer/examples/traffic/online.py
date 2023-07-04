import os
import pandas as pd
import joblib

from apxinfer.core.config import OnlineArgs, DIRHelper
from apxinfer.core.model import XIPModel
from apxinfer.core.prediction import MCPredictionEstimator
from apxinfer.core.qinfluence import XIPQInfEstimator, XIPQInfEstimatorByFInfs
from apxinfer.core.qcost import XIPQCostModel
from apxinfer.core.scheduler import XIPScheduler
from apxinfer.core.pipeline import XIPPipeline, XIPPipelineSettings

from apxinfer.core.online import OnlineExecutor

from apxinfer.examples.traffic.feature import get_fextractor


def load_model(args: OnlineArgs) -> XIPModel:
    model_path = DIRHelper.get_model_path(args)
    model = joblib.load(model_path)
    return model


def load_dataset(args: OnlineArgs, name: str, num_requests: int = 0) -> pd.DataFrame:
    dataset_dir = DIRHelper.get_dataset_dir(args)
    ds_path = os.path.join(dataset_dir, f"{name}_set.csv")
    dataset = pd.read_csv(ds_path)
    if num_requests > 0:
        dataset = dataset[:num_requests]
    return dataset


class TrafficOnlineArgs(OnlineArgs):
    plus: bool = False


if __name__ == "__main__":
    args = TrafficOnlineArgs().parse_args()

    # load test data
    test_set = load_dataset(args, "test", args.num_requests)
    verbose = args.verbose_execution and len(test_set) <= 10

    # load xip model
    model = load_model(args)

    # create a feature extractor for this task
    fextractor = get_fextractor(
        args.max_nchunks,
        args.seed,
        disable_sample_cache=args.disable_sample_cache,
        disable_query_cache=args.disable_query_cache,
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
        )
    else:
        raise ValueError("Invalid prediction estimator")

    # create qinf estimator for this task
    if args.qinf == "direct":
        qinf_estimator = XIPQInfEstimator(pred_estimator=pred_estimator)
    elif args.qinf == "by_finf":
        qinf_estimator = XIPQInfEstimatorByFInfs(pred_estimator=pred_estimator)
    else:
        raise ValueError("Invalid qinf estimator")

    # create qcost estimator for this task
    qcost_model = XIPQCostModel()

    # create a scheduler for this task
    if args.scheduler == "greedy":
        scheduler = XIPScheduler(
            fextractor=fextractor,
            model=model,
            pred_estimator=pred_estimator,
            qinf_estimator=qinf_estimator,
            qcost_estimator=qcost_model,
            sample_grans=[round(1.0 / args.ncfgs, 2)] * fextractor.num_queries,
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
    )

    # run pipline to serve online requests
    online_dir = DIRHelper.get_online_dir(args)
    OnlineExecutor(ppl=ppl, working_dir=online_dir, verbose=verbose).run(
        test_set, args.exact
    )
