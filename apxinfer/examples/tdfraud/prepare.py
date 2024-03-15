import pandas as pd
import os
from tqdm import tqdm
from typing import Tuple

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker


class TDFraudPrepareWorker(XIPPrepareWorker):
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
        nparts: int,
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
        self.database = f"xip_{seed}"
        self.table = f"tdfraud_{nparts}"

    def get_requests(self) -> pd.DataFrame:
        # for each user, we extract the most recent 5 fraudlent transaction
        # and 5 non-fraudulent transaction

        self.logger.info("Getting requests")
        sql = f"""
            SELECT *
            FROM (
                SELECT ip, count() AS cnt,
                        countIf(is_attributed=1) as cnt_fraud,
                        cnt_fraud / cnt as fraud_rate
                FROM {self.database}.{self.table}
                GROUP BY ip
                ORDER BY cnt DESC
                )
            """
        ip_fraud_df: pd.DataFrame = self.db_client.query_df(sql)
        ip_fraud_df.to_csv(os.path.join(self.working_dir, "ip_fraud.csv"), index=False)
        ip_fraud = ip_fraud_df.values
        self.logger.info(f"Number of (ip): {len(ip_fraud)}")
        self.logger.info(
            f"mean count: {ip_fraud[:, 1].mean()}, "
            f"mean fraud_cnt: {ip_fraud[:, 2].mean()}"
        )

        requests = None
        for ip, cnt, cnt_fraud, fraud_rate in tqdm(
            ip_fraud[:1000], desc="get requests", total=1000
        ):
            group_cnt = min(cnt_fraud, 5)
            sql = f"""
                SELECT txn_id, ip, app, device, os, channel,
                        toString(click_time) as click_time
                FROM {self.database}.{self.table}
                WHERE ip = {ip} AND is_attributed = 1
                ORDER BY {self.database}.{self.table}.click_time DESC
                LIMIT {group_cnt}
                """
            df_fraudulent: pd.DataFrame = self.db_client.query_df(sql)
            group_cnt = min(cnt - cnt_fraud, 5)
            sql = f"""
                SELECT txn_id, ip, app, device, os, channel,
                        toString(click_time) as click_time
                FROM {self.database}.{self.table}
                WHERE ip = {ip} AND is_attributed = 0
                ORDER BY {self.database}.{self.table}.click_time DESC
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
        requests.to_csv(os.path.join(self.working_dir, "all_requests.csv"), index=False)

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
            SELECT txn_id, is_attributed
            FROM {self.database}.{self.table}
            WHERE txn_id IN ({','.join(txn_ids)})
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels_pds = requests.merge(
            df, left_on="req_txn_id", right_on="txn_id", how="left"
        )["is_attributed"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        return super().get_features(requests)


class TDFraudRandomPrepareWorker(TDFraudPrepareWorker):
    def get_train_samples(self) -> pd.DataFrame:
        possible_dsrcs = [
            "/public/ckchang/db/clickhouse/user_files/talkingdata/adtracking-fraud",
            "/mnt/sdb/dataset/talkingdata/adtracking-fraud",
            "/mnt/hddraid/clickhouse-data/user_files/talkingdata/adtracking-fraud",
            "/var/lib/clickhouse/user_files/talkingdata/adtracking-fraud",
        ]
        dsrc = None
        for src in possible_dsrcs:
            if os.path.exists(src):
                dsrc = src
                print(f"dsrc path: {dsrc}")
                break
        if dsrc is None:
            raise RuntimeError("no valid dsrc!")
        clkh_db_dir = dsrc
        sample_path = os.path.join(clkh_db_dir, "train_sample.csv")
        train_samples = pd.read_csv(sample_path)
        return train_samples

    def get_requests(self) -> pd.DataFrame:
        req_path = os.path.join(self.working_dir, "requests.csv")
        if os.path.exists(req_path):
            self.logger.info(f"Loading requests from {req_path}")
            requests = pd.read_csv(req_path)
            return requests
        self.logger.info("Getting requests")
        train_samples = self.get_train_samples()
        requests = train_samples.drop(columns=["is_attributed", "attributed_time"])
        # add a column called txn_id, which is the row number, on the first column
        requests.insert(0, "txn_id", range(len(requests)))
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        train_samples = self.get_train_samples()
        labels_pds = train_samples["is_attributed"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds


class TDFraudKagglePrepareWorker(TDFraudPrepareWorker):
    def get_train_samples(self) -> pd.DataFrame:
        possible_dsrcs = [
            "/public/ckchang/db/clickhouse/user_files/talkingdata/adtracking-fraud",
            "/mnt/sdb/dataset/talkingdata/adtracking-fraud",
            "/mnt/hddraid/clickhouse-data/user_files/talkingdata/adtracking-fraud",
            "/var/lib/clickhouse/user_files/talkingdata/adtracking-fraud",
        ]
        dsrc = None
        for src in possible_dsrcs:
            if os.path.exists(src):
                dsrc = src
                print(f"dsrc path: {dsrc}")
                break
        if dsrc is None:
            raise RuntimeError("no valid dsrc!")
        clkh_db_dir = dsrc
        sample_path = os.path.join(clkh_db_dir, "train_sample.csv")
        train_samples = pd.read_csv(sample_path)
        return train_samples

    def get_requests(self) -> pd.DataFrame:
        req_path = os.path.join(self.working_dir, "requests.csv")
        if os.path.exists(req_path):
            self.logger.info(f"Loading requests from {req_path}")
            requests = pd.read_csv(req_path)
            return requests
        self.logger.info("Getting requests")
        train_samples = self.get_train_samples()
        requests = train_samples.drop(columns=["is_attributed", "attributed_time"])
        # add a column called txn_id, which is the row number, on the first column
        requests.insert(0, "txn_id", range(len(requests)))
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f"Getting labels for {len(requests)}x requests")
        train_samples = self.get_train_samples()
        labels_pds = train_samples["is_attributed"]
        labels_pds.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels_pds


class TDFraudRalfPrepareWorker(TDFraudPrepareWorker):
    def _extract_requests(
        self,
        start_dt: str = "2017-11-08 16:00:00",
        end_dt: str = "2017-11-09 16:00:00",
        # start_dt: str = "2017-11-08 00:00:00",
        # end_dt: str = "2017-11-08 01:00:00",
        sampling_rate: float = 0.01,
        max_num: int = 0,
    ) -> pd.DataFrame:
        # for each user, we extract the most recent 5 fraudlent transaction
        # and 5 non-fraudulent transaction
        self.logger.info("Getting requests")
        sql = f"""
            SELECT *
            FROM (
                SELECT ip, count() AS cnt,
                        countIf(is_attributed=1) as cnt_fraud,
                        cnt_fraud / cnt as fraud_rate
                FROM {self.database}.{self.table}
                GROUP BY ip
                ORDER BY (cnt, ip) DESC
                )
            """
        ip_fraud_df: pd.DataFrame = self.db_client.query_df(sql)
        ip_fraud_df.to_csv(os.path.join(self.working_dir, "ip_fraud.csv"), index=False)
        self.logger.info(f"Number of (ip): {len(ip_fraud_df)}")
        self.logger.info(
            f"mean count: {ip_fraud_df['cnt'].mean()}, "
            f"mean fraud_cnt: {ip_fraud_df['cnt_fraud'].mean()}"
        )

        ip_fraud_df = pd.read_csv(os.path.join(self.working_dir, "ip_fraud.csv"))
        ip_fraud_df = ip_fraud_df[ip_fraud_df['fraud_rate'] >= 0.01]
        selected_ips = ip_fraud_df.sample(frac=sampling_rate,
                                          random_state=0)["ip"].tolist()
        self.logger.info(f"Selected {len(selected_ips)}x of ips")

        selected_ips = [str(x) for x in selected_ips]

        sql = f"""
            SELECT toString(click_time) as ts,
                toString(attributed_time) as label_ts,
                txn_id, ip, app, device, os, channel,
                toString(click_time) as click_time
            FROM {self.database}.{self.table}
            WHERE ip IN ({','.join(selected_ips)})
                AND click_time >= '{start_dt}'
                AND click_time <= '{end_dt}'
            ORDER BY click_time, ip
        """
        requests = self.db_client.query_df(sql)

        requests["ts"] = pd.to_datetime(requests["ts"]).astype(int) // 10**9
        requests["label_ts"] = pd.to_datetime(requests["label_ts"]).astype(int) // 10**9

        # if label_ts < ts, set label_ts = ts
        requests["label_ts"] = requests["label_ts"].where(
            requests["label_ts"] > requests["ts"], requests["ts"]
        )

        self.logger.info(f"Got {len(requests)}x of requests")

        if max_num > 0 and max_num < len(requests):
            requests = requests[:max_num]

        return requests

    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests()
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n_train = len(dataset) // 2
        n_valid = 100
        train_set = dataset[:n_train]
        test_set = dataset[n_train:]
        valid_set = test_set[:n_valid]
        return train_set, valid_set, test_set


class TDFraudRalfTestPrepareWorker(TDFraudRalfPrepareWorker):
    def get_requests(self) -> pd.DataFrame:
        requests = self._extract_requests(start_dt="2017-11-09 15:00:00")
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests
