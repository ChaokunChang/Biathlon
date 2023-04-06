select trip_id,
    passenger_count,
    avg(trip_distance) over w as avg_dist,
    varPop(trip_distance) over w as var_dist,
    min(trip_distance) over w as min_dist,
    max(fare_amount) over w as max_fare,
    avg(fare_amount) over w as avg_fare,
    varPop(fare_amount) over w as var_fare,
    min(fare_amount) over w as min_fare,
    max(fare_amount) over w as max_fare,
    avg(tip_amount) over w as avg_tip,
    varPop(tip_amount) over w as var_tip,
    min(tip_amount) over w as min_tip,
    max(tip_amount) over w as max_tip,
    avg(extra) over w as avg_extra,
    varPop(extra) over w as var_extra,
    min(extra) over w as min_extra,
    max(extra) over w as max_extra,
    avg(total_amount) over w as avg_total,
    varPop(total_amount) over w as var_total,
    min(total_amount) over w as min_total,
    max(total_amount) over w as max_total,
    count(*) over w as cnt
from trips WINDOW w AS (
        PARTITION BY passenger_count
        ORDER BY pickup_datetime RANGE BETWEEN (60 * 60) PRECEDING AND CURRENT ROW
    ) INTO OUTFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/features/trips_features_qcs1-hour.csv' FORMAT CSVWithNames;