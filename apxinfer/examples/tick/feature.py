from apxinfer.core.feature import XIPFeatureExtractor
from apxinfer.examples.tick.query import TickQP0, TickQP1, TickQP2


def get_fextractor(
    max_nchunks: int,
    seed: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
    plus: bool = False,
) -> XIPFeatureExtractor:
    q0 = TickQP0(qname="query_0", enable_cache=not disable_query_cache)
    queries = [q0]
    if plus:
        raise NotImplementedError
    else:
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
        queries=queries, enable_cache=not disable_sample_cache
    )

    return fextractor
