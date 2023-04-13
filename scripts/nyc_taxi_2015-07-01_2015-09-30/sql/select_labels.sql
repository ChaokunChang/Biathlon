select trip_id,
    trip_duration,
    trip_distance,
    fare_amount,
    extra,
    tip_amount,
    tolls_amount,
    total_amount,
    payment_type,
    dropoff_datetime,
    dropoff_longitude,
    dropoff_latitude,
    dropoff_ntaname
from trips
where pickup_datetime >= '2015-08-01 00:00:00'
    and pickup_datetime < '2015-08-15 00:00:00'
order by pickup_datetime INTO OUTFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/labels_08-01_08-15.csv' FORMAT CSVWithNames;