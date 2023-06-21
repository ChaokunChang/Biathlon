import pandas as pd
import os
from tqdm import tqdm

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.model import XIPModel, XIPRegressor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, DIRHelper

from apxinfer.examples.traffic.data import req_to_dt
from apxinfer.examples.traffic.feature import get_fextractor


class TrafficPrepareWorker(XIPPrepareWorker):
    def __init__(self, working_dir: str, fextractor: XIPFeatureExtractor, max_requests: int, train_ratio: float, valid_ratio: float, model_type: str, model_name: str, seed: int) -> None:
        super().__init__(working_dir, fextractor, max_requests, train_ratio, valid_ratio, model_type, model_name, seed)
        self.database = 'xip'
        self.table = 'traffic'

    def get_requests(self) -> pd.DataFrame:
        self.logger.info('Getting requests')
        sql = f"""
            SELECT year, month, day, hour, borough
            FROM {self.database}.{self.table}
            GROUP BY year, month, day, hour, borough
            ORDER BY year, month, day, hour, borough
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        if self.max_requests > 0 and self.max_requests < len(df):
            # select max_requests in the middle
            start = int((len(df) - self.max_requests) / 2)
            end = start + self.max_requests
            df = df.iloc[start:end]
        self.logger.info(f'Got of {len(df)}x of requests')
        df.to_csv(os.path.join(self.working_dir, 'requests.csv'), index=False)
        return df

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f'Getting labels for {len(requests)}x requests')
        sql = f"""
            SELECT year, month, day, hour, borough, count() AS cnt
            FROM {self.database}.{self.table}
            GROUP BY year, month, day, hour, borough
            ORDER BY year, month, day, hour, borough
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        self.logger.info(f'Number of rows: {len(df)}')
        print(f'Columns: {df.columns}')
        labels = []
        for req in tqdm(requests.to_dict(orient='records'),
                        desc='Getting labels',
                        total=len(requests)):
            req_dt = req_to_dt(req)
            next_hour = req_dt + pd.Timedelta(hours=1)
            tmp = df[(df['year'] == next_hour.year) & (df['month'] == next_hour.month) & (df['day'] == next_hour.day) & (df['hour'] == next_hour.hour) & (df['borough'] == req['req_borough'])]['cnt'].values
            if len(tmp) > 0:
                labels.append(tmp[0])
            else:
                labels.append(None)
        labels_pds = pd.Series(labels)
        labels_pds.to_csv(os.path.join(self.working_dir, 'labels.csv'), index=False)
        return labels_pds

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        return super().get_features(requests)

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> XIPModel:
        if self.model_name == 'mlp':
            self.logger.info(f'Building pipeline for {self.model_type} {self.model_name}')
            from sklearn.neural_network import MLPRegressor
            model = XIPRegressor(MLPRegressor(hidden_layer_sizes=(100, 50, 100),
                                              random_state=self.seed,
                                              learning_rate_init=0.01,
                                              max_iter=1000,
                                              verbose=True))
            model.fit(X.values, y.values)
            return model
        else:
            return super().build_model(X, y)


class TrafficPrepareArgs(PrepareArgs):
    pass


if __name__ == '__main__':
    # Configurations
    args = TrafficPrepareArgs().parse_args()
    max_nchunks = args.max_nchunks
    n_cfgs = max_nchunks
    skip_dataset = args.skip_dataset
    max_requests = args.max_requests
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    model_name = args.model
    model_type = 'regressor'
    seed = args.seed
    # working_dir = f'/home/ckchang/.cache/apxinf/tmp/{model_name}/seed-{seed}/prepare'
    # os.makedirs(working_dir, exist_ok=True)
    working_dir = DIRHelper.get_prepare_dir(args)

    fextractor = get_fextractor(max_nchunks, seed, n_cfgs,
                                disable_sample_cache=True,
                                disable_query_cache=True)
    pworker = TrafficPrepareWorker(working_dir, fextractor, max_requests,
                                   train_ratio, valid_ratio,
                                   model_type, model_name, seed)
    pworker.run(skip_dataset=skip_dataset)
