from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.examples.tick1000.query import TickQP0, TickQP1, TickQP2


def get_fextractor(
    nparts: int,
    seed: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
    plus: bool = True,
    loading_nthreads: int = 1
) -> XIPFeatureExtractor:
    queries = []
    if plus:
        queries.append(
            TickQP0(qname=f"query_{len(queries)}", enable_cache=not disable_query_cache)
        )
    queries.append(
        TickQP1(
            qname=f"query_{len(queries)}",
            enable_cache=not disable_query_cache,
            seed=seed,
        )
    )
    for offset in range(1, 7):
        queries.append(
            TickQP2(
                qname=f"query_{len(queries)}",
                enable_cache=not disable_query_cache,
                seed=seed,
                offset=offset,
            )
        )

    # create fextractor
    fextractor = XIPFeatureExtractor(
        queries=queries, enable_cache=not disable_sample_cache, loading_nthreads=loading_nthreads
    )

    return fextractor
