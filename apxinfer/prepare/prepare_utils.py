import os
import os.path as osp
from typing import List, Tuple, Callable
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from tap import Tap
import glob
import json
import time
from tqdm import tqdm
import logging
import warnings

import apxinfer.utils as xutils
from apxinfer.utils import DBConnector
# create_model, evaluate_regressor, get_global_feature_importance

EXP_HOME = '/home/ckchang/.cache/apxinf'

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


class PrepareStageArgs(Tap):
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation
    multi_class: bool = False  # whether the task is multi-class classification
    all_features: bool = False  # whether to use all features

    max_requests: int = 2000  # maximum number of requests
    train_ratio: float = 0.5  # ratio of training data
    valid_ratio: float = 0.3  # ratio of validation data

    shuffle_table: bool = False  # whether to shuffle the table


def get_exp_dir(task: str, args: PrepareStageArgs) -> str:
    task_dir = os.path.join(EXP_HOME, task)
    model_dir = os.path.join(task_dir, args.model)
    exp_dir = os.path.join(model_dir, f'seed-{args.seed}')
    return exp_dir


class DBWorker:
    def __init__(self, database: str, tables: List[str],
                 data_src: str, src_type: str,
                 max_nchunks: int, seed: int) -> None:
        self.database = database
        self.tables = tables
        self.seed = seed
        self.data_src = data_src
        self.src_type = src_type
        self.db_client = DBConnector().client
        self.max_nchunks = max_nchunks
        self.logger = logging.getLogger(__name__)

    def check_table_not_exists(self, table: str) -> bool:
        query = f"SELECT * FROM system.tables WHERE database = '{self.database}' AND name = '{table}'"
        return len(self.db_client.command(query)) == 0

    def check_table_empty(self, table: str) -> bool:
        if self.check_table_not_exists(table):
            return True
        query = f"SELECT * FROM {self.database}.{table} LIMIT 1"
        return len(self.db_client.command(query)) == 0

    def create_tables(self) -> None:
        pass

    def ingest_data(self) -> None:
        pass

    def work(self) -> None:
        for table in self.tables:
            if (not self.check_table_not_exists(table)) or (not self.check_table_empty(table)):
                self.logger.info(f'Table {table} exists or not empty, skip ingesting')
                return None
        self.create_tables()
        self.ingest_data()


def train_valid_test_split(dataset: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # shuffle the data
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    # calculate the number of rows for each split
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # n_test = n_total - n_train - n_valid

    # split the data
    train_set = dataset[:n_train]
    valid_set = dataset[n_train:n_train + n_valid]
    test_set = dataset[n_train + n_valid:]

    return train_set, valid_set, test_set


class DatasetWorker:
    """ Dataset Worker prepare dataset for model training and evaluation
    It will prepare requests, labels, queries, and features for training, validation and testing
    """
    def __init__(self, working_dir: str, dbworker: DBWorker,
                 max_requests: int,
                 train_ratio: float, valid_ratio: float,
                 model_type: str, model_name: str,
                 seed: int) -> None:
        self.working_dir = working_dir

        self.dbworker = dbworker
        self.database = dbworker.database
        self.tables = dbworker.tables
        self.db_client = dbworker.db_client

        self.max_requests = max_requests
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        self.model_type = model_type
        self.model_name = model_name

        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def get_requests(self) -> pd.DataFrame:
        self.logger.info(f'Getting requests for from {self.database}.{self.tables}')
        raise NotImplementedError

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f'Getting labels for {len(requests)} requests')
        raise NotImplementedError

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f'Getting features for {len(requests)} requests')
        raise NotImplementedError

    def create_dataset(self) -> Tuple[pd.DataFrame, List[str], str]:
        # return dataset, fnames, label_name
        self.logger.info(f'Creating dataset for {self.model_type} {self.model_name}')
        requests = self.get_requests()
        labels = self.get_labels(requests)
        features = self.get_features(requests)
        dataset = pd.concat([requests, features, labels], axis=1)
        fnames = list(features.columns)
        label_name = labels.name
        return dataset, fnames, label_name

    def build_pipeline(self, X: pd.DataFrame, y: pd.Series) -> Pipeline:
        self.logger.info(f'Building pipeline for {self.model_type} {self.model_name}')
        model = xutils.create_model(self.model_type, self.model_name, random_state=self.seed)
        ppl = Pipeline(
            [
                ("model", model)
            ]
        )
        ppl.fit(X, y)
        joblib.dump(ppl, osp.join(self.working_dir, "pipeline.pkl"))
        return ppl

    def work(self) -> None:
        dataset, fnames, label_name = self.create_dataset()
        train_set, valid_set, test_set = train_valid_test_split(dataset=dataset, train_ratio=self.train_ratio,
                                                                valid_ratio=self.valid_ratio, seed=self.seed)
        ppl = self.build_pipeline(train_set[fnames], train_set[label_name])

        # save global feature importance of the model
        self.logger.info(f'Calculating global feature importance for {self.model_type} {self.model_name}')
        global_feature_importance = xutils.get_global_feature_importance(ppl, fnames)
        gfimps: pd.DataFrame = pd.DataFrame({'fname': fnames, 'importance': global_feature_importance}, columns=['fname', 'importance'])
        gfimps.to_csv(osp.join(self.working_dir, 'global_feature_importance.csv'), index=False)
        self.logger.info(gfimps)

        # save the dataset
        self.logger.info(f'Saving dataset for {self.model_type} {self.model_name}')
        train_set['ppl_pred'] = ppl.predict(train_set[fnames])
        valid_set['ppl_pred'] = ppl.predict(valid_set[fnames])
        test_set['ppl_pred'] = ppl.predict(test_set[fnames])
        train_set.to_csv(osp.join(self.working_dir, 'train_set.csv'), index=False)
        valid_set.to_csv(osp.join(self.working_dir, 'valid_set.csv'), index=False)
        test_set.to_csv(osp.join(self.working_dir, 'test_set.csv'), index=False)

        # save dataset statistics
        self.logger.info(f'Saving dataset statistics for {self.model_type} {self.model_name}')
        train_set.describe().to_csv(osp.join(self.working_dir, 'train_set_stats.csv'))
        valid_set.describe().to_csv(osp.join(self.working_dir, 'valid_set_stats.csv'))
        test_set.describe().to_csv(osp.join(self.working_dir, 'test_set_stats.csv'))

        # save evaluations
        self.logger.info(f'Saving evaluations for {self.model_type} {self.model_name}')
        train_evals = xutils.evaluate_pipeline(ppl, train_set[fnames].values, train_set[label_name].values)
        valid_evals = xutils.evaluate_pipeline(ppl, valid_set[fnames].values, valid_set[label_name].values)
        test_evals = xutils.evaluate_pipeline(ppl, test_set[fnames].values, test_set[label_name].values)
        all_evals = {
            'train': train_evals,
            'valid': valid_evals,
            'test': test_evals
        }
        with open(osp.join(self.working_dir, 'evals.json'), 'w') as f:
            json.dump(all_evals, f, indent=4)

        # for classification pipeline, we print and save the classification report
        if self.model_type == 'classification':
            self.logger.info(f'Saving classification reports for {self.model_type} {self.model_name}')
            train_report = metrics.classification_report(train_set[label_name], train_set['ppl_pred'])
            valid_report = metrics.classification_report(valid_set[label_name], valid_set['ppl_pred'])
            test_report = metrics.classification_report(test_set[label_name], test_set['ppl_pred'])
            self.logger.info(train_report)
            self.logger.info(valid_report)
            self.logger.info(test_report)
            with open(osp.join(self.working_dir, 'classification_reports.txt'), 'w') as f:
                f.write(f"train_report: \n{train_report}\n")
                f.write(f"valid_report: \n{valid_report}\n")
                f.write(f"test_report: \n{test_report}\n")


if __name__ == "__main__":
    dbworker = DBWorker(database='xip', tables=['sensors', 'machinery_shuffle'],
                        data_src='/tmp', src_type='file',
                        max_nchunks=10, seed=0)
    dbworker.work()
