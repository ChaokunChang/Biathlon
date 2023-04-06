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