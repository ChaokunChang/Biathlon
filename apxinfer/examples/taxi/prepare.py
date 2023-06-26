import pandas as pd
import os

from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.core.prepare import XIPPrepareWorker
from apxinfer.core.config import PrepareArgs, DIRHelper

from apxinfer.examples.taxi.data import TaxiTripIngestor
from apxinfer.examples.taxi.feature import get_fextractor


class TaxiTripPrepareWorker(XIPPrepareWorker):
    def __init__(self, working_dir: str, fextractor: XIPFeatureExtractor, max_requests: int, train_ratio: float, valid_ratio: float, model_type: str, model_name: str, seed: int) -> None:
        super().__init__(working_dir, fextractor, max_requests, train_ratio, valid_ratio, model_type, model_name, seed)
        self.database = 'xip'
        self.table = 'trips'

    def get_requests(self) -> pd.DataFrame:
        trips_from = '2015-08-01 00:00:00'
        trips_to = '2015-08-15 00:00:00'
        total_num = self.db_client.command(f""" SELECT count()
                                                FROM {self.database}.{self.table}
                                                WHERE pickup_datetime >= '{trips_from}' AND pickup_datetime < '{trips_to}'
                                            """)
        self.logger.info(f'Total number of trips: {total_num}')
        # as there is too much trips (around 5M), we should sample some of them
        sampling_rate = 0.01
        sql = f""" SELECT
                        trip_id, pickup_datetime,
                        pickup_ntaname, dropoff_ntaname,
                        pickup_longitude, pickup_latitude,
                        dropoff_longitude, dropoff_latitude,
                        passenger_count, trip_distance
                    FROM {self.database}.{self.table}
                    WHERE pickup_datetime >= '{trips_from}' AND pickup_datetime < '{trips_to}'
                          AND intHash64(trip_id) % ({int(1.0/sampling_rate)}) == 0
                """
        requests: pd.DataFrame = self.db_client.query_df(sql)
        requests['pickup_datetime'] = pd.to_datetime(requests['pickup_datetime']) + pd.Timedelta(hours=8)
        # drop requests with invalid pickup/dropoff locations or ntaname is ''
        before_size = len(requests)
        requests = requests[requests['pickup_ntaname'] != '']
        requests = requests[requests['dropoff_ntaname'] != '']
        self.logger.info(f'Dropped {before_size - len(requests)}x of requests with invalid pickup/dropoff locations')

        if self.max_requests > 0 and self.max_requests < len(requests):
            # select max_requests in the middle
            start = int((len(requests) - self.max_requests) / 2)
            end = start + self.max_requests
            requests = requests.iloc[start:end]
        self.logger.info(f'Extracted {len(requests)}x of requests')
        requests.to_csv(os.path.join(self.working_dir, 'requests.csv'), index=False)
        return requests

    def get_labels(self, requests: pd.DataFrame) -> pd.Series:
        self.logger.info(f'Getting labels for {len(requests)}x requests')
        sql = f"""
            SELECT trip_id, fare_amount
            FROM {self.database}.{self.table}
            WHERE trip_id IN ({','.join([str(x) for x in requests['req_trip_id'].values])})
            """
        df: pd.DataFrame = self.db_client.query_df(sql)
        labels_pds = requests.merge(df, left_on='req_trip_id', right_on='trip_id', how='left')['fare_amount']
        labels_pds.to_csv(os.path.join(self.working_dir, 'labels.csv'), index=False)
        return labels_pds


class TaxiTripPrepareArgs(PrepareArgs):
    plus: bool = False


def ingest_data(max_nchunks: int = 100, seed: int = 0):
    ingestor = TaxiTripIngestor(dsrc_type='clickhouse',
                                dsrc="default.trips",
                                database='xip',
                                table='trips',
                                max_nchunks=max_nchunks,
                                seed=seed)
    ingestor.run()


if __name__ == "__main__":
    args = TaxiTripPrepareArgs().parse_args()
    max_nchunks = args.max_nchunks
    n_cfgs = max_nchunks
    skip_dataset = args.skip_dataset
    max_requests = args.max_requests
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    model_name = args.model
    model_type = 'regressor'
    seed = args.seed
    working_dir = DIRHelper.get_prepare_dir(args)

    ingest_data(max_nchunks=max_nchunks, seed=seed)

    fextractor = get_fextractor(max_nchunks, seed, n_cfgs,
                                disable_sample_cache=True,
                                disable_query_cache=True,
                                plus=args.plus)
    pworker = TaxiTripPrepareWorker(working_dir, fextractor, max_requests,
                                    train_ratio, valid_ratio,
                                    model_type, model_name, seed)
    pworker.run(skip_dataset=skip_dataset)
