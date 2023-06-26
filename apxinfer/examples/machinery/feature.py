from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.machinery.query import MachineryQuery


def get_fextractor(max_nchunks: int, seed: int, n_cfgs: int,
                   disable_sample_cache: bool,
                   disable_query_cache: bool = False,
                   plus: bool = False) -> XIPFeatureExtractor:
    queries = []
    for col in [f'sensor_{i}' for i in range(8)]:  # 8 sensors
        queries.append(MachineryQuery(key=f'query_{len(queries)}',
                                      dcols=[col],
                                      aggs=['avg', 'std', 'min', 'max', 'median'] if plus else ['avg'],
                                      enable_cache=not disable_query_cache,
                                      max_nchunks=max_nchunks,
                                      seed=seed,
                                      n_cfgs=n_cfgs)
                       )
    fextractor = XIPFeatureExtractor(queries=queries,
                                     enable_cache=not disable_sample_cache)
    return fextractor
