from apxinfer.core.feature import XIPFeatureExtractor

from apxinfer.examples.ccfraud.data import CCFraudTxnsIngestor, CCFraudTxnsLoader
from apxinfer.examples.ccfraud.data import CCFraudCardsIngestor, CCFraudCardsLoader
from apxinfer.examples.ccfraud.data import CCFraudUsersIngestor, CCFraudUsersLoader
from apxinfer.examples.ccfraud.query import CCFraudQ0, CCFraudQ1, CCFraudQ2
from apxinfer.examples.ccfraud.query import CCFraudQ3, CCFraudQ4, CCFraudQ5


def get_fextractor(
    nparts: int,
    seed: int,
    disable_sample_cache: bool,
    disable_query_cache: bool = False,
    plus: bool = False,
    loading_nthreads: int = 1,
) -> XIPFeatureExtractor:
    # ingestors
    fpath = "credit-card-transactions/credit_card_transactions-ibm_v2.csv"
    txns_src = f"file('{fpath}', CSVWithNames)"
    txns_ingestor = CCFraudTxnsIngestor(
        dsrc_type="user_files",
        dsrc=txns_src,
        database="xip",
        table="cc_fraud_txns",
        nparts=nparts,
        seed=seed,
    )
    txns_ingestor.run()

    cards_src = "file('credit-card-transactions/sd254_cards.csv', CSVWithNames)"
    cards_ingestor = CCFraudCardsIngestor(
        dsrc_type="user_files",
        dsrc=cards_src,
        database="xip",
        table="cc_fraud_cards",
        nparts=nparts,
        seed=seed,
    )
    cards_ingestor.run()

    users_src = "file('credit-card-transactions/sd254_users.csv', CSVWithNames)"
    users_ingestor = CCFraudUsersIngestor(
        dsrc_type="user_files",
        dsrc=users_src,
        database="xip",
        table="cc_fraud_users",
        nparts=nparts,
        seed=seed,
    )
    users_ingestor.run()

    # data loader
    # q0_loader = None
    q1_loader = CCFraudCardsLoader(
        cards_ingestor, enable_cache=not disable_sample_cache
    )
    q2_loader = CCFraudUsersLoader(
        users_ingestor, enable_cache=not disable_sample_cache
    )
    q3_loader = CCFraudTxnsLoader(txns_ingestor, enable_cache=not disable_sample_cache)

    # Create dataset
    q0 = CCFraudQ0(
        qname="query_0",
        database="xip",
        table="cc_fraud_txns",
        enable_cache=not disable_query_cache,
    )
    q1 = CCFraudQ1(
        qname="query_1", data_loader=q1_loader, enable_cache=not disable_query_cache
    )
    q2 = CCFraudQ2(
        qname="query_2", data_loader=q2_loader, enable_cache=not disable_query_cache
    )
    q3 = CCFraudQ3(
        qname="query_3", data_loader=q3_loader, enable_cache=not disable_query_cache
    )
    queries = [q0, q1, q2, q3]

    if plus:
        q4_loader = CCFraudTxnsLoader(
            CCFraudTxnsIngestor(
                dsrc_type="user_files",
                dsrc=txns_src,
                database="xip",
                table="cc_fraud_txns",
                nparts=nparts,
                seed=seed,
            ),
            window_size=30 * 12,
            condition_cols=["uid", "card_index"],
            enable_cache=not disable_sample_cache,
        )
        q4 = CCFraudQ3(
            qname="query_4", data_loader=q4_loader, enable_cache=not disable_query_cache
        )

        q5_loader = CCFraudTxnsLoader(
            CCFraudTxnsIngestor(
                dsrc_type="user_files",
                dsrc=txns_src,
                database="xip",
                table="cc_fraud_txns",
                nparts=nparts,
                seed=seed,
            ),
            window_size=30 * 12,
            condition_cols=["uid", "card_index"],
            enable_cache=not disable_sample_cache,
        )
        q5 = CCFraudQ4(
            qname="query_5", data_loader=q5_loader, enable_cache=not disable_query_cache
        )

        q6_loader = CCFraudTxnsLoader(
            CCFraudTxnsIngestor(
                dsrc_type="user_files",
                dsrc=txns_src,
                database="xip",
                table="cc_fraud_txns",
                nparts=nparts,
                seed=seed,
            ),
            window_size=30 * 12,
            condition_cols=["uid", "card_index"],
            enable_cache=not disable_sample_cache,
        )
        q6 = CCFraudQ5(
            qname="query_6", data_loader=q6_loader, enable_cache=not disable_query_cache
        )

        queries = [q0, q1, q2, q3, q4, q5, q6]

    fextractor = XIPFeatureExtractor(queries, loading_nthreads=loading_nthreads)
    return fextractor
