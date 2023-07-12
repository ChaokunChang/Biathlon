import pandas as pd
import os

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, DIRHelper

from apxinfer.examples.machinery.data import MachineryIngestor
from apxinfer.examples.machinery.feature import get_fextractor


class MachineryMultiClassPrepareWorker(XIPPrepareWorker):
    def __init__(
        self,
        working_dir: str,
        fextractor: XIPFeatureExtractor,
        max_requests: int,
        train_ratio: float,
        valid_ratio: float,
        model_type: str,
        model_name: str,
        seed: int,
    ) -> None:
        super().__init__(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
        )
        self.database = "xip"
        self.table = "mach_imbalance"

    def get_requests(self) -> pd.DataFrame:
        sql = f"""
            SELECT bid
            FROM {self.database}.{self.table}
            GROUP BY bid
            ORDER BY bid
        """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        if self.max_requests > 0 and self.max_requests < len(requests):
            requests = requests.sample(self.max_requests, replace=False)
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        sql = f"""
            SELECT bid as req_bid, label
            FROM (
                SELECT bid, label
                FROM {self.database}.{self.table}
                GROUP BY (bid, label)
                ORDER BY (bid, label)
                )
            WHERE bid IN ({','.join([str(x) for x in requests['req_bid'].values])})
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels = requests.merge(df, on="req_bid", how="left")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["label"]


class MachineryBinaryClassPrepareWorker(MachineryMultiClassPrepareWorker):
    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        labels = super().get_labels(requests)
        labels = labels.apply(lambda x: 1 if x > 0 else 0)
        return labels


class MachineryPrepareArgs(PrepareArgs):
    plus: bool = False
    multiclass: bool = False


def ingest_data(nparts: int = 100, seed: int = 0) -> None:
    dsrc = "/mnt/hddraid/clickhouse-data/user_files/machinery"
    if not os.path.exists(dsrc):
        dsrc = "/public/ckchang/db/clickhouse/user_files/machinery"
    if not os.path.exists(dsrc):
        dsrc = "/mnt/sdb/dataset/machinery"
    ingestor = MachineryIngestor(
        dsrc_type="csv_dir",
        dsrc=dsrc,
        database="xip",
        table="mach_imbalance",
        nparts=nparts,
        seed=seed,
    )
    ingestor.run()


if __name__ == "__main__":
    args = MachineryPrepareArgs().parse_args()
    nparts = args.nparts
    skip_dataset = args.skip_dataset
    max_requests = args.max_requests
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    model_name = args.model
    model_type = "classifier"
    seed = args.seed
    working_dir = DIRHelper.get_prepare_dir(args)

    ingest_data(nparts=nparts, seed=seed)

    fextractor = get_fextractor(
        nparts,
        seed,
        disable_sample_cache=True,
        disable_query_cache=True,
        plus=args.plus,
        loading_nthreads=args.loading_nthreads
    )

    multiclass = args.multiclass
    if multiclass:
        worker = MachineryMultiClassPrepareWorker(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
        )
    else:
        worker = MachineryBinaryClassPrepareWorker(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
        )
    worker.run(skip_dataset=skip_dataset)
