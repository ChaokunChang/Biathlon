from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.tdfraud.data import TDFraudRequest


class TDFraudQP0(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_app", "req_device", "req_os",
                 "req_channel", "req_click_time"]
        dcol_aggs = [
            [lambda x: int(x[0][0])],
            [lambda x: int(x[0][0])],
            [lambda x: int(x[0][2])],
            [lambda x: int(x[0][3])],
            [lambda x: pd.to_datetime(x[0][0]).hour,
             lambda x: pd.to_datetime(x[0][0]).day],
        ]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TDFraudQP1(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        window_size = 1
        req_dt = pd.to_datetime(request["req_click_time"])
        from_dt = req_dt + dt.timedelta(days=-window_size)
        ip = request["req_ip"]
        and_list = [f"ip = {ip}",
                    f"click_time >= '{from_dt}'",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TDFraudQP2(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        req_dt = pd.to_datetime(request["req_click_time"])
        ip = request["req_ip"]
        app = request["req_app"]
        and_list = [f"ip = {ip}",
                    f"app = {app}",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class TDFraudQP3(XIPQueryProcessor):
    def __init__(self, qname: str, qtype: XIPQType, data_loader: XIPDataLoader,
                 fnames: List[str] = None, verbose: bool = False) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TDFraudRequest) -> str:
        req_dt = pd.to_datetime(request["req_click_time"])
        ip = request["req_ip"]
        app = request["req_app"]
        os = request["req_os"]
        and_list = [f"ip = {ip}",
                    f"app = {app}",
                    f"os = {os}",
                    f"click_time < '{req_dt}'"]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ['channel']
        dcol_aggs = [["count"]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops
