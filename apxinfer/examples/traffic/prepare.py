import pandas as pd
import os
from tqdm import tqdm

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, get_prepare_dir

from apxinfer.examples.traffic.data import TrafficDataIngestor, TrafficHourDataLoader
from apxinfer.examples.traffic.data import TrafficFStoreIngestor, TrafficFStoreLoader
from apxinfer.examples.traffic.data import req_to_dt
from apxinfer.examples.traffic.feature import TrafficQP0, TrafficQP1, TrafficQP2
from apxinfer.examples.traffic.feature import TrafficQP3, TrafficQP4


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
        df = self.db_client.query_df(sql)
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
            tmp = df[(df['year'] == next_hour.year) & (df['month'] == next_hour.month) & (df['day'] == next_hour.day) & (df['hour'] == next_hour.hour) & (df['borough'] == req['borough'])]['cnt'].values
            if len(tmp) > 0:
                labels.append(tmp[0])
            else:
                labels.append(None)
        label_name = 'request_label'
        labels_pds = pd.Series(labels, name=label_name)
        labels_pds.to_csv(os.path.join(self.working_dir, f'{label_name}.csv'), index=False)
        return labels_pds

    def get_features(self, requests: pd.DataFrame) -> pd.DataFrame:
        return super().get_features(requests)


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
    working_dir = get_prepare_dir(args)

    # ingestors
    dt_ingestor = TrafficDataIngestor(dsrc_type='user_files', dsrc="file('DOT_Traffic_Speeds_NBE.csv', 'CSVWithNames')",
                                      database='xip', table='traffic',
                                      max_nchunks=max_nchunks, seed=seed)
    fs_ingestor_hour = TrafficFStoreIngestor(dsrc_type='clickhouse',
                                             dsrc=f'{dt_ingestor.database}.{dt_ingestor.table}',
                                             database='xip', table='traffic_fstore_hour',
                                             granularity='hour')
    fs_ingestor_day = TrafficFStoreIngestor(dsrc_type='clickhouse',
                                            dsrc=f'{dt_ingestor.database}.{dt_ingestor.table}',
                                            database='xip', table='traffic_fstore_day',
                                            granularity='day')

    # ingest data
    dt_ingestor.run()
    fs_ingestor_hour.run()
    fs_ingestor_day.run()

    # data loader
    dt_loader = TrafficHourDataLoader(dt_ingestor)
    fs_loader_hour = TrafficFStoreLoader(fs_ingestor_hour)
    fs_loader_day = TrafficFStoreLoader(fs_ingestor_day)

    # Create dataset
    qp0 = TrafficQP0(key='query_0')
    qp1 = TrafficQP1(key='query_1', data_loader=fs_loader_hour)
    qp2 = TrafficQP2(key='query_2', data_loader=dt_loader, n_cfgs=n_cfgs)
    qp3 = TrafficQP3(key='query_3', data_loader=fs_loader_day)
    qp4 = TrafficQP4(key='query_4', data_loader=fs_loader_hour)
    queries = [qp0, qp1, qp2, qp3, qp4]
    fextractor = XIPFeatureExtractor(queries)
    creator = TrafficPrepareWorker(working_dir, fextractor, max_requests,
                                   train_ratio, valid_ratio,
                                   model_type, model_name, seed)
    creator.run(skip_dataset=skip_dataset)
