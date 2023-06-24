from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.ccfraud.data import CCFraudTxnsIngestor, CCFraudTxnsLoader
from apxinfer.examples.ccfraud.data import CCFraudCardsIngestor, CCFraudCardsLoader
from apxinfer.examples.ccfraud.data import CCFraudUsersIngestor, CCFraudUsersLoader
from apxinfer.examples.ccfraud.query import CCFraudQ0, CCFraudQ1, CCFraudQ2
from apxinfer.examples.ccfraud.query import CCFraudQ3  # , CCFraudQ4, CCFraudQ5


def get_fextractor(max_nchunks: int, seed: int, n_cfgs: int,
                   disable_sample_cache: bool,
                   disable_query_cache: bool = False) -> XIPFeatureExtractor:
    # ingestors
    txns_src = "file('credit-card-transactions/credit_card_transactions-ibm_v2.csv', CSVWithNames)"
    txns_ingestor = CCFraudTxnsIngestor(dsrc_type='user_files',
                                        dsrc=txns_src,
                                        database='xip',
                                        table='cc_fraud_txns',
                                        max_nchunks=100,
                                        seed=0)
    txns_ingestor.run()

    cards_src = "file('credit-card-transactions/sd254_cards.csv', CSVWithNames)"
    cards_ingestor = CCFraudCardsIngestor(dsrc_type='user_files',
                                          dsrc=cards_src,
                                          database='xip',
                                          table='cc_fraud_cards',
                                          max_nchunks=100,
                                          seed=0)
    cards_ingestor.run()

    users_src = "file('credit-card-transactions/sd254_users.csv', CSVWithNames)"
    users_ingestor = CCFraudUsersIngestor(dsrc_type='user_files',
                                          dsrc=users_src,
                                          database='xip',
                                          table='cc_fraud_users',
                                          max_nchunks=100,
                                          seed=0)
    users_ingestor.run()

    # data loader
    # q0_loader = None
    q1_loader = CCFraudCardsLoader(cards_ingestor, enable_cache=not disable_sample_cache)
    q2_loader = CCFraudUsersLoader(users_ingestor, enable_cache=not disable_sample_cache)
    q3_loader = CCFraudTxnsLoader(txns_ingestor, enable_cache=not disable_sample_cache)
    # q4_loader = CCFraudTxnsLoader(CCFraudTxnsIngestor(dsrc_type='user_files',
    #                                                   dsrc=txns_src,
    #                                                   database='xip',
    #                                                   table='cc_fraud_txns',
    #                                                   max_nchunks=100,
    #                                                   seed=0),
    #                               window_size=30 * 12,
    #                               condition_cols=['uid', 'card_index'],
    #                               enable_cache=not disable_sample_cache)

    # Create dataset
    q0 = CCFraudQ0(key='query_0', database='xip', table='cc_fraud_txns', enable_cache=not disable_query_cache)
    q1 = CCFraudQ1(key='query_1', data_loader=q1_loader, enable_cache=not disable_query_cache)
    q2 = CCFraudQ2(key='query_2', data_loader=q2_loader, enable_cache=not disable_query_cache)
    q3 = CCFraudQ3(key='query_3', data_loader=q3_loader, enable_cache=not disable_query_cache, n_cfgs=n_cfgs)
    queries = [q0, q1, q2, q3]
    # q4 = CCFraudQ3(key='query_4', data_loader=q4_loader, enable_cache=not disable_query_cache, n_cfgs=n_cfgs)
    # queries = [q0, q1, q2, q3, q4]
    fextractor = XIPFeatureExtractor(queries)
    return fextractor
