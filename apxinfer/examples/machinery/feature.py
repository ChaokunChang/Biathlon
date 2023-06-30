from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.machinery.query import MachineryQuery


def get_fextractor(
    max_nchunks: int,
    seed: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
    plus: bool = False,
) -> XIPFeatureExtractor:
    queries = []
    for col in [f"sensor_{i}" for i in range(8)]:  # 8 sensors
        queries.append(
            MachineryQuery(
                qname=f"query_{len(queries)}",
                dcols=[col],
                aggs=["avg", "std", "min", "max", "median"] if plus else ["avg"],
                enable_cache=not disable_query_cache,
                max_nchunks=max_nchunks,
                seed=seed,
            )
        )
    fextractor = XIPFeatureExtractor(
        queries=queries, enable_cache=not disable_sample_cache
    )
    return fextractor
