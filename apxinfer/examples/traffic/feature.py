from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.traffic.data import TrafficDataIngestor, TrafficHourDataLoader
from apxinfer.examples.traffic.data import TrafficFStoreIngestor, TrafficFStoreLoader
from apxinfer.examples.traffic.query import TrafficQP0, TrafficQP1, TrafficQP2
from apxinfer.examples.traffic.query import TrafficQP3, TrafficQP4


def get_fextractor(
    max_nchunks: int,
    seed: int,
    n_cfgs: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
) -> XIPFeatureExtractor:
    # ingestors
    dt_ingestor = TrafficDataIngestor(
        dsrc_type="user_files",
        dsrc="file('DOT_Traffic_Speeds_NBE.csv', 'CSVWithNames')",
        database="xip",
        table="traffic",
        max_nchunks=max_nchunks,
        seed=seed,
    )
    fs_ingestor_hour = TrafficFStoreIngestor(
        dsrc_type="clickhouse",
        dsrc=f"{dt_ingestor.database}.{dt_ingestor.table}",
        database="xip",
        table="traffic_fstore_hour",
        granularity="hour",
    )
    fs_ingestor_day = TrafficFStoreIngestor(
        dsrc_type="clickhouse",
        dsrc=f"{dt_ingestor.database}.{dt_ingestor.table}",
        database="xip",
        table="traffic_fstore_day",
        granularity="day",
    )

    # ingest data
    dt_ingestor.run()
    fs_ingestor_hour.run()
    fs_ingestor_day.run()

    # data loader
    qp1_loader = TrafficFStoreLoader(fs_ingestor_hour, not disable_sample_cache)
    qp2_loader = TrafficHourDataLoader(dt_ingestor, not disable_sample_cache)
    qp3_loader = TrafficFStoreLoader(fs_ingestor_day, not disable_sample_cache)
    qp4_loader = TrafficFStoreLoader(
        fs_ingestor_hour, not disable_sample_cache
    )  # avoid cache influence

    # Create dataset
    qp0 = TrafficQP0(qname="query_0", enable_cache=not disable_query_cache)
    qp1 = TrafficQP1(
        qname="query_1", data_loader=qp1_loader, enable_cache=not disable_query_cache
    )
    qp2 = TrafficQP2(
        qname="query_2",
        data_loader=qp2_loader,
        enable_cache=not disable_query_cache,
        n_cfgs=n_cfgs,
    )
    qp3 = TrafficQP3(
        qname="query_3", data_loader=qp3_loader, enable_cache=not disable_query_cache
    )
    qp4 = TrafficQP4(
        qname="query_4", data_loader=qp4_loader, enable_cache=not disable_query_cache
    )
    queries = [qp0, qp1, qp2, qp3, qp4]
    fextractor = XIPFeatureExtractor(queries)
    return fextractor
