from typing import List
import numpy as np
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class TrafficRequest(XIPRequest):
    req_year: int
    req_month: int
    req_day: int
    req_hour: int
    req_borough: str


class TrafficQConfig(XIPQueryConfig, total=False):
    pass


def req_to_dt(request: TrafficRequest) -> dt.datetime:
    datetime = dt.datetime(year=request['req_year'], month=request['req_month'],
                           day=request['req_day'], hour=request['req_hour'])
    return datetime


def dt_to_req(datetime: dt.datetime, borough: str) -> TrafficRequest:
    request = TrafficRequest(req_year=datetime.year, req_month=datetime.month,
                             req_day=datetime.day, req_hour=datetime.hour, req_borough=borough)
    return request


class TrafficDataIngestor(XIPDataIngestor):
    def __init__(self, dsrc_type: str, dsrc: str, database: str, table: str, max_nchunks: int, seed: int) -> None:
        super().__init__(dsrc_type, dsrc, database, table, max_nchunks, seed)
        self.db_client = DBHelper.get_db_client()

    def create_database(self) -> None:
        self.logger.info(f'Creating database {self.database}')
        if DBHelper.database_exists(self.db_client, self.database):
            self.logger.info(f'Database {self.database} already exists')
            return
        self.db_client.command(f'CREATE DATABASE IF NOT EXISTS {self.database}')

    def create_table(self) -> None:
        self.logger.info(f'Creating table {self.table} in database {self.database}')
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(f'Table {self.table} already exists in database {self.database}')
            return
        sql = f""" CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    trip_id UInt32, -- also row id
                    link_id UInt32, -- TRANSCOM Link ID
                    speed Float32, -- Average speed a vehicle traveled between end points on the link in the most recent interval
                    travel_time Float32, -- Time the average vehicle took to traverse the link
                    data_as_of DateTime, -- Last time data was received from link
                    link_points String, -- WKT representation of the link
                    encoded_poly_line String, -- Encoded polyline representation of the link
                    encoded_poly_line_lvls String, -- Encoded polyline levels representation of the link
                    owner String, -- Owner of the link
                    borough String, -- Borough the link is located in
                    link_name String, -- Street name of the link
                    -- derived features
                    year UInt16, -- year of data_as_of
                    month UInt8, -- month of data_as_of
                    day UInt8, -- day of data_as_of
                    hour UInt8, -- hour of data_as_of
                    minute UInt8, -- minute of data_as_of
                    pid UInt32 -- partition key, used for sampling
                ) ENGINE = MergeTree()
                ORDER BY (pid, borough, data_as_of)
                SETTINGS index_granularity = 32
        """
        self.db_client.command(sql)

    def create_aux_table(self, aux_table: str) -> int:
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{aux_table} (
                ID UInt32, -- TRANSCOM Link ID
                SPEED Float32, -- Average speed a vehicle traveled between end points on the link in the most recent interval
                TRAVEL_TIME Float32, -- Time the average vehicle took to traverse the link
                DATA_AS_OF DateTime, -- Last time data was received from link
                LINK_POINTS String, -- WKT representation of the link
                ENCODED_POLY_LINE String, -- Encoded polyline representation of the link
                ENCODED_POLY_LINE_LVLS String, -- Encoded polyline levels representation of the link
                OWNER String, -- Owner of the link
                BOROUGH String, -- Borough the link is located in
                LINK_NAME String -- Street name of the link
            ) ENGINE = MergeTree()
            ORDER BY (ID, DATA_AS_OF)
            """
        self.db_client.command(sql)
        if DBHelper.table_empty(self.db_client, self.database, aux_table):
            self.logger.info(f'Ingesting data from {self.dsrc} into table {aux_table} in database {self.database}')
            sql = f"""
                INSERT INTO {self.database}.{aux_table}
                SELECT ID AS ID, SPEED AS SPEED, TRAVEL_TIME AS TRAVEL_TIME,
                    parseDateTimeBestEffort(DATA_AS_OF) AS DATA_AS_OF, LINK_POINTS AS LINK_POINTS,
                    ENCODED_POLY_LINE AS ENCODED_POLY_LINE,
                    ENCODED_POLY_LINE_LVLS AS ENCODED_POLY_LINE_LVLS,
                    OWNER AS OWNER, BOROUGH AS BOROUGH, LINK_NAME AS LINK_NAME
                FROM {self.dsrc}
                FORMAT CSVWithNames
                """
            self.db_client.command(sql)
        return DBHelper.get_table_size(self.db_client, self.database, aux_table)

    def ingest_data(self) -> None:
        self.logger.info(f'Ingesting data from {self.dsrc} into table {self.table} in database {self.database}')
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f'Table {self.table} in database {self.database} is not empty')
            return
        assert self.dsrc_type == 'user_files', f'Unsupported data source type {self.dsrc_type}'

        # we first create an auxiliary table to store the data
        aux_table = f'{self.table}_aux'
        nrows = self.create_aux_table(aux_table)

        # we then insert the data into the main table
        self.logger.info(f'Ingesting data from {aux_table} into table {self.table} in database {self.database}')
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SELECT tmp1.*, tmp2.pid
            FROM
            (
                SELECT rowNumberInAllBlocks() as trip_id, ID AS link_id,
                    SPEED AS speed, TRAVEL_TIME AS travel_time,
                    DATA_AS_OF AS data_as_of, LINK_POINTS AS link_points,
                    ENCODED_POLY_LINE AS encoded_poly_line,
                    ENCODED_POLY_LINE_LVLS AS encoded_poly_line_lvls,
                    OWNER AS owner, BOROUGH AS borough, LINK_NAME AS link_name,
                    toYear(data_as_of) AS year, toMonth(data_as_of) AS month, toDayOfMonth(data_as_of) AS day,
                    toHour(data_as_of) AS hour, toMinute(data_as_of) AS minute
                FROM {self.database}.{aux_table}
            ) as tmp1
            JOIN
            (
                SELECT rowNumberInAllBlocks() as trip_id, value % {self.max_nchunks} as pid
                FROM generateRandom('value UInt32', {self.seed})
                LIMIT {nrows}
            ) as tmp2
            ON tmp1.trip_id = tmp2.trip_id
        """
        self.db_client.command(sql)

    def drop_aux_table(self, aux_table: str) -> None:
        DBHelper.drop_table(self.db_client, self.database, aux_table)

    def drop_table(self) -> None:
        DBHelper.drop_table(self.db_client, self.database, self.table)
        self.drop_aux_table(f'{self.table}_aux')

    def clear_aux_table(self, aux_table: str) -> None:
        DBHelper.clear_table(self.db_client, self.database, aux_table)

    def clear_table(self) -> None:
        DBHelper.clear_table(self.db_client, self.database, self.table)
        self.clear_aux_table(f'{self.table}_aux')


class TrafficHourDataLoader(XIPDataLoader):
    def __init__(self, ingestor: TrafficDataIngestor, enable_cache: bool = False) -> None:
        super().__init__('clickhouse', ingestor.database, ingestor.table,
                         ingestor.seed, enable_cache=enable_cache)
        self.ingestor = ingestor
        self.db_client = ingestor.db_client
        self.max_nchunks = ingestor.max_nchunks

    def load_data(self, request: TrafficRequest, qcfg: TrafficQConfig, cols: List[str]) -> np.ndarray:
        from_pid = 0
        to_pid = self.max_nchunks * qcfg['qsample']
        req_dt = req_to_dt(request)
        req_dt_plus_1h = req_dt + dt.timedelta(hours=1)
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE pid BETWEEN {from_pid} AND {to_pid}
                AND borough = '{request["req_borough"]}'
                AND data_as_of BETWEEN '{req_dt}' AND '{req_dt_plus_1h}'
        """
        return self.db_client.query_np(sql)


class TrafficFStoreIngestor(TrafficDataIngestor):
    def __init__(self, dsrc_type: str, dsrc: str,
                 database: str, table: str,
                 granularity: str) -> None:
        super().__init__(dsrc_type, dsrc, database, table, None, None)
        self.all_granularities = ['year', 'month', 'day', 'hour', 'minute']
        self.granularity = granularity
        self.keys = self.all_granularities[:self.all_granularities.index(self.granularity) + 1]

    def create_database(self) -> None:
        return super().create_database()

    def create_table(self) -> None:
        self.logger.info(f'Creating table {self.table} in database {self.database}')
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(f'Table {self.table} already exists in database {self.database}')
            return
        keys_w_type = [key + " UInt16" for key in self.keys]
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                borough String,
                {', '.join(keys_w_type)},
                cnt UInt32, -- number of records
                avg_speed Float32,
                avg_travel_time Float32,
                std_speed Float32,
                std_travel_time Float32,
                min_speed Float32,
                min_travel_time Float32,
                max_speed Float32,
                max_travel_time Float32,
                median_speed Float32,
                median_travel_time Float32
            ) ENGINE = MergeTree()
            PARTITION BY borough
            ORDER BY (borough, {', '.join(self.keys)})
            """
        self.db_client.command(sql)

    def ingest_data(self) -> None:
        self.logger.info(f'Ingesting data from {self.dsrc} into table {self.table} in database {self.database}')
        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f'Table {self.table} in database {self.database} is not empty')
            return
        assert self.dsrc_type == 'clickhouse', f'Unsupported data source type {self.dsrc_type}'
        assert self.granularity in self.all_granularities, f'Unsupported granularity {self.granularity}'
        assert self.granularity != 'minute', f'Granularity {self.granularity} is not supported'

        # ingest data into feature store, i.e. table {self.table}
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SELECT borough, {', '.join(self.keys)},
                count() AS cnt,
                avg(speed) AS avg_speed,
                avg(travel_time) AS avg_travel_time,
                stddevPop(speed) AS std_speed,
                stddevPop(travel_time) AS std_travel_time,
                min(speed) AS min_speed,
                min(travel_time) AS min_travel_time,
                max(speed) AS max_speed,
                max(travel_time) AS max_travel_time,
                median(speed) AS median_speed,
                median(travel_time) AS median_travel_time
            FROM {self.dsrc}
            GROUP BY borough, {', '.join(self.keys)}
        """
        self.db_client.command(sql)


class TrafficFStoreLoader(XIPDataLoader):
    def __init__(self, ingestor: TrafficFStoreIngestor,
                 enable_cache: bool = False) -> None:
        super().__init__('clickhouse', ingestor.database, ingestor.table,
                         ingestor.seed, enable_cache=enable_cache)
        self.ingestor = ingestor
        self.db_client = ingestor.db_client
        self.granularity = self.ingestor.granularity
        self.keys = self.ingestor.keys

        # self.all_granularities = ['year', 'month', 'day', 'hour']
        # self.granularity = granularity
        # self.keys = self.all_granularities[:self.all_granularities.index(self.granularity) + 1]

    def load_data(self, request: TrafficRequest, qcfg: TrafficQConfig, cols: List[str]) -> np.ndarray:
        key_values = [request[f'req_{key}'] for key in self.keys]
        conditions = [f'{key} = {value}' for key, value in zip(self.keys, key_values)]
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE borough = '{request["req_borough"]}'
                AND {' AND '.join(conditions)}
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        if df.empty:
            self.logger.warning(f'No data found for request {request}')
            return np.zeros(len(cols))
        else:
            if len(df) == 1:
                return df.values[0]
            else:
                if self.granularity == self.ingestor.granularity:
                    self.logger.warning(f'More than one record found for request {request}')
                else:
                    # TODO: feature aggregation
                    raise ValueError('feature aggregation is not supported yet')
