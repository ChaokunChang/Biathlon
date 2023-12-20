import pandas as pd
import os
import tqdm
from typing import Tuple

from apxinfer.core.fengine import XIPFEngine as XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker

from apxinfer.examples.student.data import get_dsrc


class StudentPrepareWorker(XIPPrepareWorker):
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
        self.database = "xip"
        self.table = f"student_{nparts}"

        self._cached_df = None

    def extract_request_and_labels(self) -> pd.DataFrame:
        if self._cached_df is not None:
            return self._cached_df
        dsrc = get_dsrc()
        # df = pd.read_csv(os.path.join(dsrc, 'train.csv'))
        labels = pd.read_csv(os.path.join(dsrc, "train_labels.csv"))
        labels["qno"] = labels["session_id"].apply(lambda x: int(x.split("_")[-1][1:]))
        labels["session_id"] = labels["session_id"].apply(
            lambda x: int(x.split("_")[0])
        )
        df = labels
        self._cached_df = df
        df.to_csv(os.path.join(self.working_dir, "extracted_df.csv"), index=False)
        return self._cached_df

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        requests = df[["session_id", "qno"]]
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        df = self.extract_request_and_labels()
        df = requests.merge(
            df,
            left_on=["req_session_id", "req_qno"],
            right_on=["session_id", "qno"],
            how="left",
        )
        labels = df[["correct"]]
        self.logger.info(f"Getting labels for {len(labels)}x requests")
        labels.to_csv(os.path.join(self.working_dir, "labels.csv"), index=False)
        return labels["correct"]

    def split_dataset(
        self, dataset: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_set, valid_set, test_set = super().split_dataset(dataset)
        test_set = test_set.sample(len(test_set) // 10, random_state=0)
        if test_set > 1000:
            test_set = test_set.sample(len(test_set) // 10, random_state=0)
            if test_set > 1000:
                test_set = test_set.sample(1000, random_state=0)
        return train_set, valid_set, test_set


class StudentQNoPrepareWorker(StudentPrepareWorker):
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
        qno: int,
    ) -> None:
        self.qno = qno
        super().__init__(
            working_dir,
            fextractor,
            max_requests,
            train_ratio,
            valid_ratio,
            model_type,
            model_name,
            seed,
            nparts,
        )

    def get_requests(self) -> pd.DataFrame:
        df = self.extract_request_and_labels()
        df = df[df["qno"] == self.qno]
        requests = df[["session_id", "qno"]]
        if self.max_requests > 0 and self.max_requests < len(requests):
            requests = requests.sample(self.max_requests, random_state=0)
        self.logger.info(f"Extracted {len(requests)}x of requests")
        requests.to_csv(os.path.join(self.working_dir, "requests.csv"), index=False)
        return requests
