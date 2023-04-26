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
FROM INFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/db_src/trips_{0..19}.gz' FORMAT TSVWithNames;
-- add new column trip_duration as (dropoff_datetime - pickup_datetime)
ALTER TABLE trips
ADD COLUMN trip_duration Float32;
-- update trip_duration
ALTER TABLE trips
UPDATE trip_duration = (dropoff_datetime - pickup_datetime)
WHERE 1;
-- clean the data. 
-- remove records with negative trip_duration, trip_distance, fare_amount, total_amount, and passenger_count, 
ALTER TABLE trips DELETE
WHERE trip_duration < 0
    OR trip_distance < 0
    OR fare_amount < 0
    OR extra < 0
    OR tip_amount < 0
    OR total_amount < 0
    OR passenger_count < 0;
-- create table with sample
CREATE TABLE trips_w_samples (
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
    trip_duration Float32
) ENGINE = MergeTree PRIMARY KEY (intHash64(pickup_datetime), dropoff_datetime) SAMPLE BY intHash64(pickup_datetime);
-- insert data into trips_w_samples from trips
INSERT INTO trips_w_samples
SELECT *
FROM trips;