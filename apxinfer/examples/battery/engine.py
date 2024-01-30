from typing import List
from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor
from apxinfer.core.fengine import XIPFEngine

from apxinfer.examples.battery.query import BatteryQPAgg, BatteryQPNonAGG


def get_battery_engine(nparts: int, ncores: int = 0, verbose: bool = False):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table=f"battery_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = ["Voltage_measured", "Current_measured", "Temperature_measured",
               "Current_load", "Voltage_load"]
    aggops = ['avg', 'stdPop', 'skew', 'kurtosis']
    aggops = ['avg', 'stdPop']
    # aggops = ['min', 'max', 'avg', 'stdPop', 'skew', 'kurtosis']
    for col in columns:
        qps.append(BatteryQPAgg(
            qname=f"q-{len(qps)}",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            dcol=col,
            dcol_ops=aggops,
            verbose=verbose)
        )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine


def get_batteryv2_engine(nparts: int, ncores: int = 0, verbose: bool = False):
    data_loader: XIPDataLoader = XIPDataLoader(
        backend="clickhouse",
        database="xip",
        table=f"battery_{nparts}",
        seed=0,
        enable_cache=False,
    )
    if verbose:
        print(f"tsize ={data_loader.statistics['tsize']}")
        print(f"nparts={data_loader.statistics['nparts']}")

    qps: List[XIPQueryProcessor] = []
    columns = ["Voltage_measured", "Current_measured", "Temperature_measured",
               "Current_load", "Voltage_load"]
    aggops = ['avg', 'stdPop', 'skew', 'kurtosis']
    aggops = ['avg', 'stdPop']
    # aggops = ['min', 'max', 'avg', 'stdPop', 'skew', 'kurtosis']
    for col in columns:
        qps.append(BatteryQPAgg(
            qname=f"q-{len(qps)}",
            qtype=XIPQType.AGG,
            data_loader=data_loader,
            dcol=col,
            dcol_ops=aggops,
            verbose=verbose)
        )
    qps.append(BatteryQPNonAGG(
        qname=f"q-{len(qps)}",
        qtype=XIPQType.NORMAL,
        data_loader=data_loader,
        verbose=verbose)
    )

    fengine = XIPFEngine(qps, ncores, verbose=verbose)
    return fengine
