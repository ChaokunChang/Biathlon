from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.taxi.query import TaxiTripQ0, TaxiTripQ1, TaxiTripQ2, TaxiTripQ3
from apxinfer.examples.taxi.query import TaxiTripAGGFull


def get_fextractor(
    max_nchunks: int,
    seed: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
    plus: bool = False,
) -> XIPFeatureExtractor:
    # create queries
    q0 = TaxiTripQ0(qname="query_0", enable_cache=not disable_query_cache)
    queries = [q0]
    if plus:
        for hour in [1, 24, 24 * 7]:
            queries.append(
                TaxiTripAGGFull(
                    qname=f"query_{len(queries)}",
                    window_hours=hour,
                    condition_cols=["pickup_ntaname"],
                    finished_only=False,
                    dcols=["trip_distance", "passenger_count"],
                    aggs=["count", "avg", "sum", "max", "min", "median", "unique"],
                    enable_cache=not disable_query_cache,
                    max_nchunks=max_nchunks,
                    seed=seed,
                )
            )
            queries.append(
                TaxiTripAGGFull(
                    qname=f"query_{len(queries)}",
                    window_hours=hour,
                    condition_cols=["dropoff_ntaname"],
                    finished_only=False,
                    dcols=["trip_distance", "passenger_count"],
                    aggs=["count", "avg", "sum", "max", "min", "median", "unique"],
                    enable_cache=not disable_query_cache,
                    max_nchunks=max_nchunks,
                    seed=seed,
                )
            )
            queries.append(
                TaxiTripAGGFull(
                    qname=f"query_{len(queries)}",
                    window_hours=hour,
                    condition_cols=["pickup_ntaname", "dropoff_ntaname"],
                    finished_only=False,
                    dcols=["trip_distance", "passenger_count"],
                    aggs=["count", "avg", "sum", "max", "min", "median", "unique"],
                    enable_cache=not disable_query_cache,
                    max_nchunks=max_nchunks,
                    seed=seed,
                )
            )
            queries.append(
                TaxiTripAGGFull(
                    qname=f"query_{len(queries)}",
                    window_hours=hour,
                    condition_cols=[
                        "pickup_ntaname",
                        "dropoff_ntaname",
                        "passenger_count",
                    ],
                    finished_only=True,
                    dcols=[
                        "trip_distance",
                        "fare_amount",
                        "tip_amount",
                        "trip_duration",
                    ],
                    aggs=["count", "avg", "sum", "max", "min", "median", "unique"],
                    enable_cache=not disable_query_cache,
                    max_nchunks=max_nchunks,
                    seed=seed,
                )
            )
    else:
        q1 = TaxiTripQ1(
            qname="query_1",
            enable_cache=not disable_query_cache,
            max_nchunks=max_nchunks,
            seed=seed,
        )
        q2 = TaxiTripQ2(
            qname="query_2",
            enable_cache=not disable_query_cache,
            max_nchunks=max_nchunks,
            seed=seed,
        )
        q3 = TaxiTripQ3(
            qname="query_3",
            enable_cache=not disable_query_cache,
            max_nchunks=max_nchunks,
            seed=seed,
        )
        queries.extend([q1, q2, q3])
    # create fextractor
    fextractor = XIPFeatureExtractor(
        queries=queries, enable_cache=not disable_sample_cache
    )

    return fextractor
