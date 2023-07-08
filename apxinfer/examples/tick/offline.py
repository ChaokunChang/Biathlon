from apxinfer.core.config import DIRHelper, LoadingHelper
from apxinfer.core.config import OfflineArgs
from apxinfer.core.offline import OfflineExecutor

from apxinfer.examples.tick.feature import get_fextractor


if __name__ == "__main__":
    args = OfflineArgs().parse_args()

    # load test data
    test_set = LoadingHelper.load_dataset(args, "valid", args.nreqs)
    verbose = args.verbose and len(test_set) <= 10

    # load xip model
    model = LoadingHelper.load_model(args)

    # create a feature extractor for this task
    fextractor = get_fextractor(
        args.nparts,
        args.seed,
        disable_sample_cache=False,
        disable_query_cache=False,
    )

    executor = OfflineExecutor(
        working_dir=DIRHelper.get_offline_dir(args),
        fextractor=fextractor,
        nparts=args.nparts,
        ncfgs=args.ncfgs,
        verbose=verbose,
    )
    executor.run(test_set, args.clear_cache)
