from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.trips.data import get_dloader
from apxinfer.examples.tripsfeast.query import TripsQP0, TripsQP1, TripsQP2
from apxinfer.examples.tripsfeast.query import TripsQP3, TripsQP4


def get_trips_feast_engine(nparts: int, ncores: int = 0,
                           seed: int = 0, verbose: bool = False):
    data_loader: XIPDataLoader = get_dloader(nparts, seed=seed, verbose=verbose)
    qp0 = TripsQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TripsQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TripsQP2(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_trips_feast_engine_vary(nparts: int, rate: int,
                                ncores: int = 0, seed: int = 0,
                                verbose: bool = False):
    data_loader: XIPDataLoader = get_dloader(nparts, seed, verbose=verbose)
    qp0 = TripsQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TripsQP3(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        window=1.0*rate,
        fnames=None,
        verbose=verbose,
    )
    qp2 = TripsQP4(
        qname="q2",
        qtype=XIPQType.AGG,
        data_loader=data_loader,
        window=0.5*rate,
        fnames=None,
        verbose=verbose,
    )
    qps: List[XIPQueryProcessor] = [qp0, qp1, qp2]

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
