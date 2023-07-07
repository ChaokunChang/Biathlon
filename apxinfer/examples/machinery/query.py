from typing import List

from apxinfer.core.utils import XIPQueryConfig, XIPFeatureVec, XIPQType
from apxinfer.core.utils import merge_fvecs
from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import FEstimatorHelper

from apxinfer.examples.machinery.data import MachineryRequest
from apxinfer.examples.machinery.data import MachineryLoader


class MachineryQuery(XIPQuery):
    def __init__(
        self,
        qname: str,
        dcols: List[str],
        aggs: List[str] = ["avg"],
        enable_cache: bool = False,
        nparts: int = 100,
        seed: int = 0,
    ) -> None:
        self.aggs = aggs
        self.dcols = dcols
        fnames = [f"{agg}_{dcol}" for agg in aggs for dcol in dcols]
        data_loader = MachineryLoader(
            backend="clickhouse",
            database="xip",
            table="mach_imbalance",
            seed=seed,
            enable_cache=enable_cache,
            nparts=nparts,
        )
        super().__init__(qname, XIPQType.AGG, data_loader, fnames, enable_cache)

    def run(self, request: MachineryRequest, qcfg: XIPQueryConfig) -> XIPFeatureVec:
        req_data = self.data_loader.load_data(request, qcfg, self.dcols)
        if req_data is None or len(req_data) == 0:
            return self.get_default_fvec(request, qcfg)
        else:
            fvecs = []
            for agg in self.aggs:
                fvec: XIPFeatureVec = FEstimatorHelper.SUPPORTED_AGGS[agg](
                    req_data, qcfg["qsample"], self.data_loader.statistics["tsize"]
                )
                fvecs.append(fvec)
            fvec = merge_fvecs(fvecs, new_names=self.fnames)
        return fvec
