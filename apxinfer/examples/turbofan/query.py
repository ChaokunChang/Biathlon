from typing import List

from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.turbofan.data import TurbofanRequest


class TurbofanQPAgg(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        dcol: str,
        dcol_ops: List[str],
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        self.dcol = dcol
        self.dcol_ops = dcol_ops
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_condition(self, request: TurbofanRequest) -> str:
        req_name = request["req_name"]
        req_unit = request["req_unit"]
        req_cycle = request["req_cycle"]
        qcond = f"name = '{req_name}' AND unit = {req_unit} AND cycle = {req_cycle}"
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [self.dcol]
        dcol_aggs = [self.dcol_ops]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops
