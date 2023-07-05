import pandas as pd
import numpy as np
import datetime as dt
import os
from tqdm import tqdm

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, DIRHelper

from apxinfer.examples.tick.feature import get_fextractor


class TickPrepareWorker(XIPPrepareWorker):
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

    def get_requests(self) -> pd.DataFrame:
        self.logger.info("Getting requests")
        sql = """SELECT toString(min(tick_dt)) from xip.tick"""
        min_dt: str = self.db_client.command(sql)
        min_dt = dt.datetime.strptime(min_dt, "%Y-%m-%d %H:%M:%S.%f")
        start_dt = min_dt + dt.timedelta(hours=7)

        max_dt: str = self.db_client.command(
            """SELECT toString(max(tick_dt)) from xip.tick"""
        )
        max_dt = dt.datetime.strptime(max_dt, "%Y-%m-%d %H:%M:%S.%f")
        end_dt = max_dt - dt.timedelta(hours=1)

        sql = f""" WITH makeDateTime64(year, month, day, hour, 0, 0) as dt
                SELECT count()
                FROM xip.tick_fstore_hour
                WHERE dt > '{start_dt}' AND dt < '{end_dt}'
            """
        cnt: int = self.db_client.command(sql)
        self.logger.info(f"number of possible requests: {cnt}")
        rate = 1
        if self.max_requests > 0 and self.max_requests < cnt:
            rate = int(max(rate, cnt // self.max_requests))
        self.logger.info(f"requests sampling: {rate}")

        sql = f"""
                WITH makeDateTime64(year, month, day, hour, 0, 0) as clk_dt
                SELECT cpair, toString(clk_dt) as dt
                FROM xip.tick_fstore_hour
                WHERE (cityHash64(clk_dt) % {int(rate)}) == 0
                        AND clk_dt > '{start_dt}'
                        AND clk_dt < '{end_dt}'
                ORDER BY (cpair, year, month, day, hour)
                """
        df: pd.DataFrame = self.db_client.query_df(sql)
        df["dt"] = df["dt"].apply(lambda x: pd.to_datetime(x).to_pydatetime())
        self.logger.info(f"Got of {len(df)}x of requests")
        df.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return df

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")

        labels = []
        for req in tqdm(
            requests.to_dict(orient="records"),
            desc="Getting labels",
            total=len(requests),
        ):
            cpair: str = req["req_cpair"]
            req_dt: dt.datetime = req["req_dt"]

            target_dt = req_dt + dt.timedelta(hours=1)
            year = target_dt.year
            month = target_dt.month
            day = target_dt.day
            hour = target_dt.hour
            avg_bid = self.db_client.command(
                f""" SELECT avg_bid
                    FROM xip.tick_fstore_hour
                    WHERE cpair = '{cpair}'
                    AND year = {year} AND month = {month}
                    AND day = {day} AND hour = {hour}
                """
            )
            labels.append(avg_bid)
        labels_pds = pd.Series(labels)
        labels_pds.replace("", np.nan, inplace=True)
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds


if __name__ == "__main__":
    # Configurations
    args = PrepareArgs().parse_args()
    max_nchunks = args.max_nchunks
    skip_dataset = args.skip_dataset
    max_requests = args.max_requests
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    model_name = args.model
    model_type = "regressor"
    seed = args.seed
    # working_dir = f'/home/ckchang/.cache/apxinf/tmp/{model_name}/seed-{seed}/prepare'
    # os.makedirs(working_dir, exist_ok=True)
    working_dir = DIRHelper.get_prepare_dir(args)

    fextractor = get_fextractor(
        max_nchunks, seed, disable_sample_cache=True, disable_query_cache=True
    )
    pworker = TickPrepareWorker(
        working_dir,
        fextractor,
        max_requests,
        train_ratio,
        valid_ratio,
        model_type,
        model_name,
        seed,
    )
    pworker.run(skip_dataset=skip_dataset)
