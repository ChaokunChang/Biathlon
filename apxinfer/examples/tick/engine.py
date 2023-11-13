from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.tick.query import TickQP0
from apxinfer.examples.tick.query import TickQP1, TickQP2
from apxinfer.examples.tick.query import TickQP3, TickQP4


def get_tick_engine(nparts: int, ncores: int = 0, verbose: bool = False):
    tick_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table=f"tick_{nparts}",
        seed=0,
        enable_cache=False,
    )

    tick_fstore_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table="tick_fstore_hour",
        seed=0,
        enable_cache=False,
    )

    qp0 = TickQP0(
        qname="q0",
        qtype=XIPQType.NORMAL,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qp1 = TickQP1(
        qname="q1",
        qtype=XIPQType.AGG,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    )
    qps: List[XIPQueryProcessor] = [qp0, qp1]
    for offset in range(1, 7):
        qps.append(TickQP2(
            qname=f"q{len(qps)}",
            qtype=XIPQType.FSTORE,
            data_loader=tick_fstore_loader,
            offset=offset,
            fnames=None,
            verbose=verbose,
        ))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_tick_engine_v2(nparts: int, ncores: int = 0, verbose: bool = False):
    tick_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table=f"tick_{nparts}",
        seed=0,
        enable_cache=False,
    )

    tick_fstore_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table="tick_fstore_hour",
        seed=0,
        enable_cache=False,
    )

    qps: List[XIPQueryProcessor] = []
    for offset in range(1, 7):
        qps.append(TickQP2(
            qname=f"q{len(qps)}",
            qtype=XIPQType.FSTORE,
            data_loader=tick_fstore_loader,
            offset=7 - offset,
            fnames=None,
            verbose=verbose,
        ))
    qps.append(TickQP1(
        qname=f"q{len(qps)}",
        qtype=XIPQType.AGG,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    ))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_tick_engine_v3(nparts: int, ncores: int = 0, verbose: bool = False):
    tick_data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table=f"tick_{nparts}",
        seed=0,
        enable_cache=False,
    )

    tick_fstore_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table="tick_fstore_hour",
        seed=0,
        enable_cache=False,
    )

    qps: List[XIPQueryProcessor] = []
    for offset in range(1, 7):
        qps.append(TickQP2(
            qname=f"q{len(qps)}",
            qtype=XIPQType.FSTORE,
            data_loader=tick_fstore_loader,
            offset=7 - offset,
            fnames=None,
            verbose=verbose,
        ))
    qps.append(TickQP1(
        qname=f"q{len(qps)}",
        qtype=XIPQType.AGG,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    ))

    for offset in range(1, 7):
        qps.append(TickQP4(
            qname=f"q{len(qps)}",
            qtype=XIPQType.FSTORE,
            data_loader=tick_fstore_loader,
            offset=7 - offset,
            fnames=None,
            verbose=verbose,
        ))
    qps.append(TickQP3(
        qname=f"q{len(qps)}",
        qtype=XIPQType.AGG,
        data_loader=tick_data_loader,
        fnames=None,
        verbose=verbose,
    ))
    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
