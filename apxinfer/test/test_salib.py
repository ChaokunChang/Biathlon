import numpy as np
import pandas as pd
import time
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from apxinfer.core.config import LoadingHelper
from apxinfer.core.config import OnlineArgs
from apxinfer.core.model import XIPModel

profiles = []


def compute_sobol_main_effect_index(fvals: list, fdists: list, fests: list,
                                    groups: list, model: XIPModel,
                                    N: int = 1000, seed: int = 0,
                                    calc_second_order: bool = False):
    n_features = len(fdists)
    # print(fvec)
    bounds = []
    dists = []
    for i in range(n_features):
        if fdists[i] == 'fixed':
            bounds.append([fvals[i], 1e-9])
            dists.append('norm')
        elif fdists[i] in ['normal', 'r-normal', 'l-normal']:
            bounds.append([fvals[i], max(fests[i], 1e-9)])
            dists.append('norm')
        else:
            raise ValueError(f"Unknown distribution {dists[i]}")
    problem = {
        "num_vars": n_features,
        "groups": groups,
        "names": [f'f{i}' for i in range(n_features)],
        "bounds": bounds,
        "dists": dists
    }
    st = time.time()
    param_values = sobol_sample.sample(problem, N,
                                       calc_second_order=calc_second_order,
                                       seed=seed)
    sampling_time = time.time() - st

    st = time.time()
    preds = model.predict(param_values)
    prediction_time = time.time() - st

    st = time.time()
    Si = sobol_analyze.analyze(problem, preds,
                               calc_second_order=calc_second_order,
                               seed=seed)
    analysis_time = time.time() - st

    profiles.append({
        "N": N,
        "sampling_time": sampling_time,
        "prediction_time": prediction_time,
        "analysis_time": analysis_time,
        "total_time": sampling_time + prediction_time + analysis_time
    })

    return Si["S1"]


if __name__ == "__main__":
    args = OnlineArgs().parse_args(known_only=True)
    model: XIPModel = LoadingHelper.load_model(args)

    dataset = LoadingHelper.load_dataset(
        args, "test", args.nreqs, offset=args.nreqs_offset
    )
    cols = dataset.columns.tolist()
    req_cols = [col for col in cols if col.startswith("req_")]
    fcols = [col for col in cols if col.startswith("f_")]
    label_col = "label"

    groups = []
    fvals = []
    fdists = []
    fests = []
    for fcol in fcols:
        # f_{Type}_xxx_{group}
        ftype = fcol.split("_")[1]
        group = fcol.split("_")[-1]

        groups.append(group)
        # val is the mean of that feature column in dataset
        fvals.append(dataset[fcol].mean())
        if ftype == "AGG":
            # dist is normal, and est is the std of that feature column in dataset
            fdists.append("normal")
            fests.append(dataset[fcol].std())
        else:
            # dist is fixed, and est is 0
            fdists.append("fixed")
            fests.append(0)

    qinfs = compute_sobol_main_effect_index(fvals, fdists, fests,
                                            groups, model,
                                            N=args.pest_nsamples,
                                            seed=args.seed,
                                            calc_second_order=False)
    print(f"qinfs = {qinfs}")

    profiles = []
    for N in [100, 200, 300, 500, 1000, 2000, 5000]:
        qinfs = compute_sobol_main_effect_index(fvals, fdists, fests,
                                                groups, model,
                                                N=args.pest_nsamples,
                                                seed=args.seed,
                                                calc_second_order=False)
        print(f"qinfs = {qinfs}")

    profile_df = pd.DataFrame(profiles)
    # sort by N
    profile_df = profile_df.sort_values(by='N')
    print(profile_df)
