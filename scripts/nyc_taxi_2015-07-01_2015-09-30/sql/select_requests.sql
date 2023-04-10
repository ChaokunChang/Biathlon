select trip_id,
    pickup_datetime,
    pickup_longitude,
    pickup_latitude,
    passenger_count,
    pickup_ntaname
from trips
where pickup_datetime >= '2015-08-01 00:00:00'
    and pickup_datetime < '2015-08-08 00:00:00'
order by pickup_datetime INTO OUTFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/requests_08-01_08-08.csv' FORMAT CSVWithNames;