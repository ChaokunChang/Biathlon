from typing import List
import numpy as np
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class CCFraudRequest(XIPRequest):
    req_txn_id: int
    req_uid: int
    req_card_index: int
    req_txn_datetime: str
    req_amount: float
    req_use_chip: str
    req_merchant_name: str
    req_merchant_city: str
    req_merchant_state: str
    req_zip_code: int
    req_mcc: int
    req_errors: str


class CCFraudTxnsIngestor(XIPDataIngestor):
    def __init__(
        self,
        dsrc_type: str,
        dsrc: str,
        database: str,
        table: str,
        nparts: int,
        seed: int,
    ) -> None:
        super().__init__(dsrc_type, dsrc, database, table, nparts, seed)
        self.db_client = DBHelper.get_db_client()

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    txn_id UInt32,
                    uid UInt32,
                    card_index UInt32,
                    txn_datetime DateTime,
                    year UInt32,
                    month UInt32,
                    day UInt32,
                    hour UInt32,
                    minute UInt32,
                    amount Float32,
                    use_chip String,
                    merchant_name String,
                    merchant_city String,
                    merchant_state String,
                    zip_code UInt32,
                    mcc UInt32,
                    errors String,
                    is_fraud UInt8,
                    pid UInt32 -- partition key, used for sampling
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY (uid, card_index, txn_datetime)
                SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def create_aux_table(self, aux_table: str) -> int:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{aux_table} (
                    uid UInt32,
                    card_index UInt32,
                    txn_datetime DateTime,
                    year UInt32,
                    month UInt32,
                    day UInt32,
                    hour UInt32,
                    minute UInt32,
                    amount Float32,
                    use_chip String,
                    merchant_name String,
                    merchant_city String,
                    merchant_state String,
                    zip_code UInt32,
                    mcc UInt32,
                    errors String,
                    is_fraud UInt8
            ) ENGINE = MergeTree()
            ORDER BY (uid, card_index, txn_datetime)
            """
        self.db_client.command(sql)
        if DBHelper.table_empty(self.db_client, self.database, aux_table):
            self.logger.info(
                f"Ingesting data from {self.dsrc} into table {aux_table} in database {self.database}"
            )
            sql = f"""
                INSERT INTO {self.database}.{aux_table}
                SELECT User AS uid, Card AS card_index,
                    parseDateTimeBestEffort(concat(toString(Year), '-', toString(Month), '-', toString(Day), ' ', toString(Time))) as txn_datetime,
                    Year AS year, Month AS month, Day AS day,
                    toHour(txn_datetime) AS hour,
                    toMinute(txn_datetime) AS minute,
                    toFloat64(replaceAll(Amount, '$', '')) AS amount,
                    `Use Chip` AS use_chip,
                    toString(`Merchant Name`) AS merchant_name,
                    `Merchant City` AS merchant_city,
                    `Merchant State` AS merchant_state,
                    Zip AS zip_code, MCC AS mcc,
                    `Errors?` AS errors,
                    (`Is Fraud?` == 'Yes') AS is_fraud
                FROM {self.dsrc}
                FORMAT CSVWithNames
                """
            self.db_client.command(sql)
        return DBHelper.get_table_size(self.db_client, self.database, aux_table)

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        assert (
            self.dsrc_type == "user_files"
        ), f"Unsupported data source type {self.dsrc_type}"

        # we first create an auxiliary table to store the data
        aux_table = f"{self.table}_aux"
        nrows = self.create_aux_table(aux_table)
        print(f"nrows = {nrows}")

        # we then insert the data into the main table
        self.logger.info(
            f"Ingesting data from {aux_table} into table {self.database}.{self.table}"
        )
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SELECT tmp1.*, tmp2.pid
            FROM
            (
                SELECT rowNumberInAllBlocks() as txn_id, *
                FROM {self.database}.{aux_table}
            ) as tmp1
            JOIN
            (
                SELECT rowNumberInAllBlocks() as txn_id, value % {self.nparts} as pid
                FROM generateRandom('value UInt32', {self.seed})
                LIMIT {nrows}
            ) as tmp2
            ON tmp1.txn_id = tmp2.txn_id
        """
        self.db_client.command(sql)

        # we drop the auxiliary table
        self.drop_aux_table(aux_table)

    def drop_aux_table(self, aux_table: str) -> None:
        DBHelper.drop_table(self.db_client, self.database, aux_table)

    def drop_table(self) -> None:
        DBHelper.drop_table(self.db_client, self.database, self.table)
        self.drop_aux_table(f"{self.table}_aux")

    def clear_aux_table(self, aux_table: str) -> None:
        DBHelper.clear_table(self.db_client, self.database, aux_table)

    def clear_table(self) -> None:
        DBHelper.clear_table(self.db_client, self.database, self.table)
        self.clear_aux_table(f"{self.table}_aux")


class CCFraudTxnsLoader(XIPDataLoader):
    def __init__(
        self,
        ingestor: CCFraudTxnsIngestor,
        window_size: int = 30,
        condition_cols: List[str] = ["uid"],
        enable_cache: bool = False,
    ) -> None:
        super().__init__(
            "clickhouse",
            ingestor.database,
            ingestor.table,
            ingestor.seed,
            enable_cache=enable_cache,
        )
        self.ingestor = ingestor
        self.db_client = ingestor.db_client
        self.nparts = ingestor.nparts
        self.window_size = window_size  # window size in days
        self.condition_cols = condition_cols

    def load_data(
        self, request: CCFraudRequest, qcfg: XIPQueryConfig,
        cols: List[str], loading_nthreads: int = 1
    ) -> np.ndarray:
        from_pid = self.nparts * qcfg.get("qoffset", 0)
        to_pid = self.nparts * qcfg["qsample"]
        req_dt = pd.to_datetime(request["req_txn_datetime"])
        from_dt = req_dt + dt.timedelta(days=-self.window_size)
        conditon_values = [request[f"req_{col}"] for col in self.condition_cols]
        condtions = [
            f"{col} = '{val}'" for col, val in zip(self.condition_cols, conditon_values)
        ]
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE pid >= {int(from_pid)} AND pid < {int(to_pid)}
                AND txn_datetime >= '{from_dt}' AND txn_datetime < '{req_dt}'
                AND {' AND '.join(condtions)}
            SETTINGS max_threads = {loading_nthreads}
        """
        return self.db_client.query_np(sql)


class CCFraudCardsIngestor(CCFraudTxnsIngestor):
    def create_database(self) -> None:
        return super().create_database()

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                uid UInt32,
                card_index UInt32,
                card_brand String,
                card_type String,
                card_number Int64,
                exp_date String,
                cvv Int16,
                has_chip UInt8,
                cards_issued Int64,
                credit_limit Int64,
                acct_open_date String,
                pin_last_changed UInt32,
                card_on_dark_web UInt8,
            ) ENGINE = MergeTree()
            ORDER BY (uid, card_index)
            """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        # ingest data into feature store, i.e. table {self.table}
        sql = f"""
            INSERT INTO {self.database}.{self.table}
                SELECT `User` AS uid, `CARD INDEX` AS card_index,
                    `Card Brand` AS card_brand, `Card Type` AS card_type,
                    `Card Number` AS card_number, `Expires` AS exp_date,
                    `CVV` AS cvv, `Has Chip` AS has_chip,
                    `Cards Issued` AS cards_issued,
                    toFloat64(replaceAll(`Credit Limit`, '$', '')) AS credit_limit,
                    `Acct Open Date` AS acct_open_date, `Year PIN last Changed` AS pin_last_changed,
                    (`Card on Dark Web` == 'Yes') AS card_on_dark_web
                FROM {self.dsrc}
                FORMAT CSVWithNames
        """
        self.db_client.command(sql)


class CCFraudCardsLoader(XIPDataLoader):
    def __init__(
        self, ingestor: CCFraudCardsIngestor, enable_cache: bool = False
    ) -> None:
        super().__init__(
            "clickhouse",
            ingestor.database,
            ingestor.table,
            ingestor.seed,
            enable_cache=enable_cache,
        )
        self.ingestor = ingestor
        self.db_client = ingestor.db_client

    def load_data(
        self, request: CCFraudRequest, qcfg: XIPQueryConfig,
        cols: List[str], loading_nthreads: int = 1
    ) -> np.ndarray:
        uid = request["req_uid"]
        card_index = request["req_card_index"]
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE uid = {uid} AND card_index = {card_index}
            SETTINGS max_threads = {loading_nthreads}
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        if df.empty:
            self.logger.warning(f"No data found for request {request}")
            return np.zeros(len(cols))
        else:
            if len(df) == 1:
                return df.values[0]
            else:
                print(f"req={request}, cols={cols}, df={df}")
                raise ValueError("feature aggregation is not supported yet")


class CCFraudUsersIngestor(CCFraudTxnsIngestor):
    def create_database(self) -> None:
        return super().create_database()

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                uid UInt32,
                uname String,
                current_age UInt8,
                retirement_age UInt8,
                birth_year UInt16,
                birth_month UInt8,
                gender UInt8,
                address String,
                apartment String,
                city String,
                state String,
                zipcode String,
                latitude Float64,
                longitude Float64,
                per_capita_income UInt32,
                yearly_income UInt32,
                total_debt UInt32,
                fico_score UInt16,
                num_credit_cards UInt8,
            ) ENGINE = MergeTree()
            ORDER BY (uid)
            """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        # ingest data into feature store, i.e. table {self.table}
        sql = f"""
            INSERT INTO {self.database}.{self.table}
                SELECT rowNumberInAllBlocks() as uid,
                    `Person` AS uname,
                    `Current Age` AS current_age,
                    `Retirement Age` AS retirement_age,
                    `Birth Year` AS birth_year,
                    `Birth Month` AS birth_month,
                    `Gender` AS gender,
                    `Address` AS address,
                    `Apartment` AS apartment,
                    `City` AS city,
                    `State` AS state,
                    `Zipcode` AS zipcode,
                    toFloat64(`Latitude`) AS latitude,
                    toFloat64(`Longitude`) AS longitude,
                    toFloat64(replaceAll(`Per Capita Income - Zipcode`, '$', '')) AS per_capita_income,
                    toFloat64(replaceAll(`Yearly Income - Person`, '$', '')) AS yearly_income,
                    toFloat64(replaceAll(`Total Debt`, '$', '')) AS total_debt,
                    toFloat64(`FICO Score`) AS fico_score,
                    toInt32(`Num Credit Cards`) AS num_credit_cards
                FROM {self.dsrc}
                FORMAT CSVWithNames
        """
        self.db_client.command(sql)


class CCFraudUsersLoader(XIPDataLoader):
    def __init__(
        self, ingestor: CCFraudUsersIngestor, enable_cache: bool = False
    ) -> None:
        super().__init__(
            "clickhouse",
            ingestor.database,
            ingestor.table,
            ingestor.seed,
            enable_cache=enable_cache,
        )
        self.ingestor = ingestor
        self.db_client = ingestor.db_client

    def load_data(
        self, request: CCFraudRequest, qcfg: XIPQueryConfig,
        cols: List[str], loading_nthreads: int = 1
    ) -> np.ndarray:
        uid = request["req_uid"]
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE uid = {uid}
            SETTINGS max_threads = {loading_nthreads}
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        if df.empty:
            self.logger.warning(f"No data found for request {request}")
            return np.zeros(len(cols))
        else:
            if len(df) == 1:
                return df.values[0]
            else:
                raise ValueError("feature aggregation is not supported yet")
