from typing import List
import numpy as np
import pandas as pd
import datetime as dt

from apxinfer.core.utils import XIPRequest, XIPQueryConfig
from apxinfer.core.data import DBHelper, XIPDataIngestor, XIPDataLoader


class TaxiTripRequest(XIPRequest):
    req_trip_id: int
    req_pickup_datetime: str
    req_pickup_ntaname: str
    req_dropoff_ntaname: str
    req_pickup_longitude: float
    req_pickup_latitude: float
    req_dropoff_longitude: float
    req_dropoff_latitude: float
    req_passenger_count: int
    req_trip_distance: float


class TaxiTripIngestor(XIPDataIngestor):
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

    def create_table(self) -> None:
        self.logger.info(f"Creating table {self.database}.{self.table}")
        if DBHelper.table_exists(self.db_client, self.database, self.table):
            self.logger.info(
                f"Table {self.table} already exists in database {self.database}"
            )
            return
        sql = f"""CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                    trip_id UInt32,
                    pickup_datetime DateTime,
                    dropoff_datetime DateTime,
                    pickup_longitude Nullable(Float64),
                    pickup_latitude Nullable(Float64),
                    dropoff_longitude Nullable(Float64),
                    dropoff_latitude Nullable(Float64),
                    passenger_count UInt8,
                    trip_distance Float32,
                    fare_amount Float32,
                    extra Float32,
                    tip_amount Float32,
                    tolls_amount Float32,
                    total_amount Float32,
                    payment_type Enum(
                        'CSH' = 1,
                        'CRE' = 2,
                        'NOC' = 3,
                        'DIS' = 4,
                        'UNK' = 5
                    ),
                    pickup_ntaname LowCardinality(String),
                    dropoff_ntaname LowCardinality(String),
                    trip_duration Float32,
                    pid UInt32 -- partition key, used for sampling
                ) ENGINE = MergeTree()
                PARTITION BY pid
                ORDER BY (passenger_count, pickup_datetime)
                SETTINGS index_granularity = 32
            """
        self.db_client.command(sql)

    def create_dsrc(self, dtable: str):
        sql_create = f"""
            -- create tables for trips in clickhouse
            CREATE TABLE {dtable} (
                trip_id UInt32,
                pickup_datetime DateTime,
                dropoff_datetime DateTime,
                pickup_longitude Nullable(Float64),
                pickup_latitude Nullable(Float64),
                dropoff_longitude Nullable(Float64),
                dropoff_latitude Nullable(Float64),
                passenger_count UInt8,
                trip_distance Float32,
                fare_amount Float32,
                extra Float32,
                tip_amount Float32,
                tolls_amount Float32,
                total_amount Float32,
                payment_type Enum(
                    'CSH' = 1,
                    'CRE' = 2,
                    'NOC' = 3,
                    'DIS' = 4,
                    'UNK' = 5
                ),
                pickup_ntaname LowCardinality(String),
                dropoff_ntaname LowCardinality(String)
            ) ENGINE = MergeTree PARTITION BY passenger_count
            ORDER BY (pickup_datetime, dropoff_datetime)
            """
        dsrc_home = "/var/lib/clickhouse/user_files/taxi-2015/trips_{0..19}.gz"
        sql_insert = f"""
            -- insert data into trips from lcoal files (20m records)
            INSERT INTO {dtable}
            SELECT * FROM file('{dsrc_home}', TSVWithNames)
            -- FROM INFILE '{dsrc_home}' FORMAT TSVWithNames
            """
        sql_alter = f"""
            -- add new column trip_duration as (dropoff_datetime - pickup_datetime)
            ALTER TABLE {dtable}
            ADD COLUMN trip_duration Float32
            """
        sql_update = f"""
            -- update trip_duration
            ALTER TABLE {dtable}
            UPDATE trip_duration = (dropoff_datetime - pickup_datetime)
            WHERE 1
            """
        sql_clean = f"""
            -- clean the data.
            -- remove records with negative trip_duration, trip_distance, fare_amount, total_amount, and passenger_count, 
            ALTER TABLE {dtable} DELETE
            WHERE trip_duration < 0
                OR trip_distance < 0
                OR fare_amount < 0
                OR extra < 0
                OR tip_amount < 0
                OR total_amount < 0
                OR passenger_count < 0
            """
        self.db_client.command(sql_create)
        self.db_client.command(sql_insert)
        self.db_client.command(sql_alter)
        self.db_client.command(sql_update)
        self.db_client.command(sql_clean)

    def ingest_data(self) -> None:
        assert (
            self.dsrc_type == "clickhouse"
        ), f"Unsupported data source type {self.dsrc_type}"

        self.logger.info(
            f"Ingesting data from {self.dsrc} into table {self.database}.{self.table}"
        )

        if not DBHelper.table_exists(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        ):
            self.create_dsrc(self.dsrc)
        elif DBHelper.table_empty(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        ):
            self.create_dsrc(self.dsrc)

        if not DBHelper.table_empty(self.db_client, self.database, self.table):
            self.logger.info(f"Table {self.database}.{self.table} is not empty")
            return
        nrows = DBHelper.get_table_size(
            self.db_client, self.dsrc.split(".")[0], self.dsrc.split(".")[1]
        )
        sql = f"""
            INSERT INTO {self.database}.{self.table}
            SELECT trip_id, pickup_datetime, dropoff_datetime,
              pickup_longitude, pickup_latitude,
              dropoff_longitude, dropoff_latitude,
              passenger_count, trip_distance, fare_amount,
              extra, tip_amount, tolls_amount, total_amount,
              payment_type, pickup_ntaname, dropoff_ntaname,
              trip_duration, pid
            FROM
            (
                SELECT *, rowNumberInAllBlocks() as row_id
                FROM {self.dsrc}
            ) as tmp1
            JOIN
            (
                SELECT value % {self.nparts} as pid, rowNumberInAllBlocks() as row_id
                FROM (
                    SELECT *
                    FROM generateRandom('value UInt32', {self.seed})
                    LIMIT {nrows}
                )
            ) as tmp2
            ON tmp1.row_id = tmp2.row_id
        """
        self.db_client.command(sql)


class TaxiTripLoader(XIPDataLoader):
    def __init__(
        self,
        backend: str,
        database: str,
        table: str,
        seed: int,
        enable_cache: bool,
        nparts: int,
        window_hours: int = 1,
        condition_cols: List[str] = ["pickup_ntaname"],
        finished_only: bool = False,
    ) -> None:
        super().__init__(backend, database, table, seed, enable_cache)
        self.nparts = nparts
        self.window_hours = window_hours  # window_size in hours
        self.condition_cols = condition_cols
        self.finished_only = finished_only

    def load_data(
        self, req: TaxiTripRequest, qcfg: XIPQueryConfig,
        cols: List[str], loading_nthreads: int = 1
    ) -> np.ndarray:
        from_pid = self.nparts * qcfg.get("qoffset", 0)
        to_pid = self.nparts * qcfg["qsample"]

        to_dt = pd.to_datetime(req["req_pickup_datetime"])
        from_dt = to_dt - dt.timedelta(hours=self.window_hours)

        conditon_values = [req[f"req_{col}"] for col in self.condition_cols]
        conditon_values = [
            val.replace("'", r"\'") if isinstance(val, str) else val
            for val in conditon_values
        ]
        condtions = [
            f"{col} = '{val}'" for col, val in zip(self.condition_cols, conditon_values)
        ]
        finished_only = (
            f"dropoff_datetime IS NOT NULL AND dropoff_datetime <= '{to_dt}'"
            if self.finished_only
            else "1 = 1"
        )
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.database}.{self.table}
            WHERE pid >= {from_pid} AND pid < {to_pid}
                AND pickup_datetime >= '{from_dt}' AND pickup_datetime < '{to_dt}'
                AND {' AND '.join(condtions)}
                AND {finished_only}
            SETTINGS max_threads = {loading_nthreads}
        """
        df: pd.DataFrame = self.db_client.query_df(sql)
        return df.values
