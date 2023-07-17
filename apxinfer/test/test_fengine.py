from typing import List, TypedDict, Union, Callable
import logging
import numpy as np
import pandas as pd
import time
import json
from tap import Tap
import asyncio
import datetime as dt
from tap import Tap


from apxinfer.examples.taxi.data import TaxiTripRequest
from apxinfer.core.utils import XIPRequest, XIPQType, XIPQueryConfig
from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.utils import merge_fvecs, is_same_float
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.festimator import XIPFeatureErrorEstimator, XIPFeatureEstimator
from apxinfer.core.query import XIPQOperatorDescription, XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.test.test_query import get_request, get_dloader
from apxinfer.test.test_query import get_qps, get_qcfgs


class FEngineTestArgs(Tap):
    verbose: bool = False
    qsample: float = 0.1
    ld_nthreads: int = 0
    cp_nthreads: int = 0
    ncores: int = 0


def get_fengine(args: FEngineTestArgs, data_loader: XIPDataLoader):
    qps = get_qps(data_loader, verbose=args.verbose)
    fengine = XIPFEngine(qps, args.ncores, verbose=args.verbose)
    return fengine


if __name__ == "__main__":
    args = FEngineTestArgs().parse_args()
    print(f"run with args: {args}")

    request = get_request()
    data_loader: XIPDataLoader = get_dloader(args.verbose)

    qps = get_qps(data_loader, verbose=args.verbose)
    qcfgs = get_qcfgs(qps, args.qsample, 0)
    fengine = XIPFEngine(qps, args.ncores, verbose=args.verbose)
    ret_seq = fengine.run(request, qcfgs, "sequential")

    qps = get_qps(data_loader, verbose=args.verbose)
    qcfgs = get_qcfgs(qps, args.qsample, 0)
    fengine = XIPFEngine(qps, args.ncores, verbose=args.verbose)
    ret_asy = fengine.run(request, qcfgs, "async")
