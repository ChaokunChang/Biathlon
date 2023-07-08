import pandas as pd
import os
from tqdm import tqdm

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, DIRHelper

from apxinfer.examples.ccfraud.data import (
    CCFraudCardsIngestor,
    CCFraudTxnsIngestor,
    CCFraudUsersIngestor,
)
from apxinfer.examples.ccfraud.feature import get_fextractor


class CCFraudPrepareWorker(XIPPrepareWorker):
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
        self.table = "cc_fraud_txns"

    def get_requests(self) -> pd.DataFrame:
        # for each user, we extract the most recent 5 fraudlent transaction
        # and 5 non-fraudulent transaction
        self.logger.info("Getting requests")
        sql = f"""
            SELECT *
            FROM (
                SELECT uid, card_index, count() AS cnt,
                        countIf(is_fraud=1) as cnt_fraud,
                        cnt_fraud / cnt as fraud_rate
                FROM {self.database}.{self.table}
                GROUP BY uid, card_index
                ORDER BY cnt DESC
                )
            """
        uid_cards_df: pd.DataFrame = self.db_client.query_df(sql)
        uid_cards_df.to_csv(
            os.path.join(self.working_dir, "uid_cards.csv"), index=False
        )
        uid_cards = uid_cards_df.values
        self.logger.info(f"Number of (uid, card_index): {len(uid_cards)}")
        self.logger.info(
            f"mean count: {uid_cards[:, 2].mean()}, "
            f"mean fraud_cnt: {uid_cards[:, 3].mean()}"
        )

        requests = None
        for uid, card_index, cnt, cnt_fraud, fraud_rate in tqdm(
            uid_cards[:1000], desc="get requests", total=1000
        ):
            group_cnt = min(cnt_fraud, 5)
            sql = f"""
                SELECT txn_id, uid, card_index,
                        toString(txn_datetime) as txn_datetime,
                        amount, use_chip, merchant_name,
                        merchant_city, merchant_state, zip_code,
                        mcc, errors
                FROM {self.database}.{self.table}
                WHERE uid = {uid} AND card_index = {card_index} AND is_fraud = 1
                ORDER BY {self.database}.{self.table}.txn_datetime DESC
                LIMIT {group_cnt}
                """
            df_fraudulent: pd.DataFrame = self.db_client.query_df(sql)
            group_cnt = min(cnt - cnt_fraud, 5)
            sql = f"""
                SELECT txn_id, uid, card_index,
                        toString(txn_datetime) as txn_datetime,
                        amount, use_chip, merchant_name,
                        merchant_city, merchant_state, zip_code,
                        mcc, errors
                FROM {self.database}.{self.table}
                WHERE uid = {uid} AND card_index = {card_index} AND is_fraud = 0
                ORDER BY {self.database}.{self.table}.txn_datetime DESC
                LIMIT {group_cnt}
                """
            df_non_fraudulent: pd.DataFrame = self.db_client.query_df(sql)
            df = pd.concat(
                [df_fraudulent, df_non_fraudulent], axis=0, ignore_index=True
            )
            if len(df) > 0:
                if requests is None:
                    requests = df
                else:
                    requests = pd.concat([requests, df], axis=0, ignore_index=True)
        self.logger.info(f"Got {len(requests)}x of requests")

        if self.max_requests > 0 and self.max_requests < len(requests):
            # select max_requests in the middle
            start = int((len(requests) - self.max_requests) / 2)
            end = start + self.max_requests
            requests = requests.iloc[start:end]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        txn_ids = [str(x) for x in requests["req_txn_id"].values]
        sql = f"""
            SELECT txn_id, is_fraud
            FROM {self.database}.{self.table}
            WHERE txn_id IN ({','.join(txn_ids)})
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels_pds = requests.merge(
            df, left_on="req_txn_id", right_on="txn_id", how="left"
        )["is_fraud"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        return super().get_features(requests)


class CCFraudPrepareArgs(PrepareArgs):
    plus: bool = False


def ingest_data(nparts: int = 100, seed: int = 0):
    txns_src = "file('credit-card-transactions/credit_card_transactions-ibm_v2.csv', CSVWithNames)"
    txns_ingestor = CCFraudTxnsIngestor(
        dsrc_type="user_files",
        dsrc=txns_src,
        database="xip",
        table="cc_fraud_txns",
        nparts=nparts,
        seed=seed,
    )
    txns_ingestor.run()

    cards_src = "file('credit-card-transactions/sd254_cards.csv', CSVWithNames)"
    cards_ingestor = CCFraudCardsIngestor(
        dsrc_type="user_files",
        dsrc=cards_src,
        database="xip",
        table="cc_fraud_cards",
        nparts=nparts,
        seed=seed,
    )
    cards_ingestor.run()

    users_src = "file('credit-card-transactions/sd254_users.csv', CSVWithNames)"
    users_ingestor = CCFraudUsersIngestor(
        dsrc_type="user_files",
        dsrc=users_src,
        database="xip",
        table="cc_fraud_users",
        nparts=nparts,
        seed=seed,
    )
    users_ingestor.run()


if __name__ == "__main__":
    # Configurations
    args = CCFraudPrepareArgs().parse_args()
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
    )
    pworker = CCFraudPrepareWorker(
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
