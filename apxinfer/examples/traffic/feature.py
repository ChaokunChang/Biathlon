from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.traffic.data import TrafficDataIngestor, TrafficHourDataLoader
from apxinfer.examples.traffic.data import TrafficFStoreIngestor, TrafficFStoreLoader
from apxinfer.examples.traffic.query import TrafficQP0, TrafficQP1, TrafficQP2
from apxinfer.examples.traffic.query import TrafficQP3, TrafficQP4


def get_fextractor(max_nchunks: int, seed: int, n_cfgs: int, disable_sample_cache: bool) -> XIPFeatureExtractor:
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
    dt_loader = TrafficHourDataLoader(dt_ingestor, not disable_sample_cache)
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
    return fextractor
