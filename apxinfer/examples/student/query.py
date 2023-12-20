from typing import List
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPQType
from apxinfer.core.data import XIPDataLoader
from apxinfer.core.query import XIPQueryProcessor, XIPQOperatorDescription

from apxinfer.examples.student.data import StudentRequest


def get_qury_group(q_no: int) -> str:
    # Select level group for the question based on the q_no.
    if q_no <= 3:
        grp = "0-4"
    elif q_no <= 13:
        grp = "5-12"
    elif q_no <= 22:
        grp = "13-22"
    return grp


class StudentQP0(XIPQueryProcessor):
    def __init__(
        self,
        qname: str,
        qtype: XIPQType,
        data_loader: XIPDataLoader,
        fnames: List[str] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(qname, qtype, data_loader, fnames, verbose)

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = ["req_qno"]
        dcol_aggs = [[lambda x: int(x[0][0])]]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops


class StudentQP1(XIPQueryProcessor):
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

    def get_query_condition(self, request: StudentRequest) -> str:
        req_session_id = request["req_session_id"]
        req_qno = request["req_qno"]
        level_group = get_qury_group(req_qno)

        and_list = [
            f"session_id = {req_session_id}",
            f"level_group = '{level_group}'",
            f"isNotNull({self.dcol})"
        ]
        qcond = " AND ".join(and_list)
        return qcond

    def get_query_ops(self) -> List[XIPQOperatorDescription]:
        dcols = [self.dcol]
        dcol_aggs = [self.dcol_ops]
        qops = [
            XIPQOperatorDescription(dcol=dcol, dops=dcol_aggs[i])
            for i, dcol in enumerate(dcols)
        ]
        return qops
