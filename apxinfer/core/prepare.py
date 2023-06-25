import numpy as np
import pandas as pd
import time
import os
import os.path as osp
import joblib
import logging
import json
from tqdm import tqdm
from typing import List, Tuple
from sklearn import metrics

from apxinfer.core.utils import XIPFeatureVec
from apxinfer.core.data import DBHelper
# from apxinfer.core.query import XIPQuery
from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.model import XIPModel, create_model, evaluate_model

logging.basicConfig(level=logging.INFO)


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


class XIPPrepareWorker:
    """ This Worker prepares dataset for model training and evaluation
    It will prepare requests, labels, queries, and features for training, validation and testing
    """
    def __init__(self, working_dir: str,
                 fextractor: XIPFeatureExtractor,
                 max_requests: int,
                 train_ratio: float, valid_ratio: float,
                 model_type: str, model_name: str,
                 seed: int) -> None:
        self.working_dir = working_dir

        self.fextractor = fextractor
        self.db_client = DBHelper.get_db_client()

        self.max_requests = max_requests
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        self.model_type = model_type
        self.model_name = model_name

        self.seed = seed
        self.logger = logging.getLogger('DatasetCreator')

    def get_requests(self) -> pd.DataFrame:
        self.logger.info(f'Getting requests for {osp.basename(self.working_dir)}')
        raise NotImplementedError

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f'Getting labels for {len(requests)}x requests')
        raise NotImplementedError

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        num_requests = len(requests)
        self.logger.info(f'Getting features for {num_requests}x requests')
        fnames = []
        qfeatures_list = []
        qcosts = []
        for qid, query in enumerate(self.fextractor.queries):
            st = time.time()
            num_qf = len(query.fnames)
            fnames.extend(query.fnames)

            qfeatures = np.zeros((num_requests, num_qf))
            self.logger.info(f'Extracting features {query.fnames}')
            final_qcfg = query.cfg_pools[-1]
            for rid, req in tqdm(enumerate(requests.to_dict(orient='records')),
                                 desc=f'Extracting {qid}',
                                 total=num_requests):
                fvec: XIPFeatureVec = query.run(req, final_qcfg)
                # print(fvec)
                qfeatures[rid] = fvec['fvals']
            self.logger.info(f'Extracted features {query.fnames}')
            qfeatures_list.append(qfeatures)
            qcosts.append(time.time() - st)
        features = np.concatenate(qfeatures_list, axis=1)
        features = pd.DataFrame(features, columns=fnames)
        with open(osp.join(self.working_dir, 'dataset', 'qcosts.json'), 'w') as f:
            json.dump({'num_requests': num_requests, 'qcosts': qcosts}, f, indent=4)
        features.to_csv(osp.join(self.working_dir, 'dataset', 'features.csv'), index=False)
        return features

    def create_dataset(self) -> Tuple[pd.DataFrame, List[str], str]:
        # return dataset, fnames, label_name
        self.logger.info(f'Creating dataset for {self.model_type} {self.model_name}')
        requests = self.get_requests()
        requests = requests.add_prefix('req_')
        # add request_id column
        requests.insert(0, 'req_id', range(len(requests)))

        features = self.get_features(requests)
        features = features.add_prefix('f_')

        labels = self.get_labels(requests)
        labels = labels.rename('label')

        dataset = pd.concat([requests.reset_index(drop=True),
                             features.reset_index(drop=True),
                             labels.reset_index(drop=True)], axis=1)

        # remove the requests that have no features or labels
        dataset = dataset.dropna()
        self.logger.info(f'droped {len(dataset) - len(requests)}x requests')

        fnames = list(features.columns)
        label_name = labels.name
        return dataset, fnames, label_name

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        self.logger.info(f'Building pipeline for {self.model_type} {self.model_name}')
        model = create_model(self.model_type, self.model_name, random_state=self.seed)
        model.fit(X.values, y.values)
        return model

    def prepare_dirs(self):
        dataset_dir = osp.join(self.working_dir, 'dataset')
        model_dir = osp.join(self.working_dir, 'model')
        for d in [dataset_dir, model_dir]:
            os.makedirs(d, exist_ok=True)

    def run(self, skip_dataset: bool = False) -> None:
        self.prepare_dirs()
        if skip_dataset:
            dataset = pd.read_csv(osp.join(self.working_dir, 'dataset', 'dataset.csv'))
            cols = list(dataset.columns)
            fnames = [col for col in cols if col.startswith('f_')]
            label_name = cols[-1]
        else:
            dataset, fnames, label_name = self.create_dataset()
            dataset.to_csv(osp.join(self.working_dir, 'dataset', 'dataset.csv'), index=False)

        train_set, valid_set, test_set = train_valid_test_split(dataset=dataset, train_ratio=self.train_ratio,
                                                                valid_ratio=self.valid_ratio, seed=self.seed)
        # save the dataset
        self.logger.info(f'Saving dataset for {self.model_type} {self.model_name}')
        train_set.to_csv(osp.join(self.working_dir, 'dataset', 'train_set.csv'), index=False)
        valid_set.to_csv(osp.join(self.working_dir, 'dataset', 'valid_set.csv'), index=False)
        test_set.to_csv(osp.join(self.working_dir, 'dataset', 'test_set.csv'), index=False)

        # save dataset statistics
        self.logger.info(f'Saving dataset statistics for {self.model_type} {self.model_name}')
        train_set.describe().to_csv(osp.join(self.working_dir, 'dataset', 'train_set_stats.csv'))
        valid_set.describe().to_csv(osp.join(self.working_dir, 'dataset', 'valid_set_stats.csv'))
        test_set.describe().to_csv(osp.join(self.working_dir, 'dataset', 'test_set_stats.csv'))
