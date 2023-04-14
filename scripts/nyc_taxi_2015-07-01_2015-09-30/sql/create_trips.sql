-- create tables for trips in clickhouse
CREATE TABLE trips (
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
ORDER BY (pickup_datetime, dropoff_datetime);
-- insert data into trips from lcoal files (20m records)
INSERT INTO trips
FROM INFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/data_src/trips_{0..19}.gz' FORMAT TSVWithNames;
-- add new column trip_duration as (dropoff_datetime - pickup_datetime)
ALTER TABLE trips
ADD COLUMN trip_duration Float32;
-- update trip_duration
ALTER TABLE trips
UPDATE trip_duration = (dropoff_datetime - pickup_datetime)
WHERE 1;