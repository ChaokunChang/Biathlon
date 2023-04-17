select trip_id,
    passenger_count,
    pickup_datetime,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    pickup_ntaname,
    dropoff_ntaname,
    trip_distance
from trips
where pickup_datetime >= '2015-08-01 00:00:00'
    and pickup_datetime < '2015-08-15 00:00:00'
order by pickup_datetime INTO OUTFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/requests_08-01_08-15.csv' FORMAT CSVWithNames;