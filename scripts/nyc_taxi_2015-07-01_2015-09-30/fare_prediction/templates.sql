--- 1h  window
SELECT count(*) as count_1h,
    avg(trip_duration) as avg_trip_duration_1h,
    avg(trip_distance) as avg_trip_distance_1h,
    avg(fare_amount) as avg_fare_amount_1h,
    avg(tip_amount) as avg_tip_amount_1h,
    avg(total_amount) as avg_total_amount_1h,
    sum(trip_duration) as sum_trip_duration_1h,
    sum(trip_distance) as sum_trip_distance_1h,
    sum(fare_amount) as sum_fare_amount_1h,
    sum(tip_amount) as sum_tip_amount_1h,
    sum(total_amount) as sum_total_amount_1h,
    stddevPop(trip_duration) as std_trip_duration_1h,
    stddevPop(trip_distance) as std_trip_distance_1h,
    stddevPop(fare_amount) as std_fare_amount_1h,
    stddevPop(tip_amount) as std_tip_amount_1h,
    stddevPop(total_amount) as std_total_amount_1h,
    varPop(trip_duration) as var_trip_duration_1h,
    varPop(trip_distance) as var_trip_distance_1h,
    varPop(fare_amount) as var_fare_amount_1h,
    varPop(tip_amount) as var_tip_amount_1h,
    varPop(total_amount) as var_total_amount_1h,
    min(trip_duration) as min_trip_duration_1h,
    min(trip_distance) as min_trip_distance_1h,
    min(fare_amount) as min_fare_amount_1h,
    min(tip_amount) as min_tip_amount_1h,
    min(total_amount) as min_total_amount_1h,
    max(trip_duration) as max_trip_duration_1h,
    max(trip_distance) as max_trip_distance_1h,
    max(fare_amount) as max_fare_amount_1h,
    max(tip_amount) as max_tip_amount_1h,
    max(total_amount) as max_total_amount_1h,
    median(trip_duration) as median_trip_duration_1h,
    median(trip_distance) as median_trip_distance_1h,
    median(fare_amount) as median_fare_amount_1h,
    median(tip_amount) as median_tip_amount_1h,
    median(total_amount) as median_total_amount_1h
FROM trips
WHERE (
        pickup_datetime >= (
            toDateTime('{pickup_datetime}') - toIntervalHour(1)
        )
    )
    AND (pickup_datetime < '{pickup_datetime}')
    AND (dropoff_datetime <= '{pickup_datetime}')
    AND (passenger_count = {passenger_count})
    AND (pickup_ntaname = '{pickup_ntaname}')
    AND (dropoff_ntaname = '{dropoff_ntaname}');

--- 24h  window
SELECT count(*) as count_24h,
    avg(trip_duration) as avg_trip_duration_24h,
    avg(trip_distance) as avg_trip_distance_24h,
    avg(fare_amount) as avg_fare_amount_24h,
    avg(tip_amount) as avg_tip_amount_24h,
    avg(total_amount) as avg_total_amount_24h,
    sum(trip_duration) as sum_trip_duration_24h,
    sum(trip_distance) as sum_trip_distance_24h,
    sum(fare_amount) as sum_fare_amount_24h,
    sum(tip_amount) as sum_tip_amount_24h,
    sum(total_amount) as sum_total_amount_24h,
    stddevPop(trip_duration) as std_trip_duration_24h,
    stddevPop(trip_distance) as std_trip_distance_24h,
    stddevPop(fare_amount) as std_fare_amount_24h,
    stddevPop(tip_amount) as std_tip_amount_24h,
    stddevPop(total_amount) as std_total_amount_24h,
    varPop(trip_duration) as var_trip_duration_24h,
    varPop(trip_distance) as var_trip_distance_24h,
    varPop(fare_amount) as var_fare_amount_24h,
    varPop(tip_amount) as var_tip_amount_24h,
    varPop(total_amount) as var_total_amount_24h,
    min(trip_duration) as min_trip_duration_24h,
    min(trip_distance) as min_trip_distance_24h,
    min(fare_amount) as min_fare_amount_24h,
    min(tip_amount) as min_tip_amount_24h,
    min(total_amount) as min_total_amount_24h,
    max(trip_duration) as max_trip_duration_24h,
    max(trip_distance) as max_trip_distance_24h,
    max(fare_amount) as max_fare_amount_24h,
    max(tip_amount) as max_tip_amount_24h,
    max(total_amount) as max_total_amount_24h,
    median(trip_duration) as median_trip_duration_24h,
    median(trip_distance) as median_trip_distance_24h,
    median(fare_amount) as median_fare_amount_24h,
    median(tip_amount) as median_tip_amount_24h,
    median(total_amount) as median_total_amount_24h
FROM trips
WHERE (
        pickup_datetime >= (
            toDateTime('{pickup_datetime}') - toIntervalHour(24)
        )
    )
    AND (pickup_datetime < '{pickup_datetime}')
    AND (dropoff_datetime <= '{pickup_datetime}')
    AND (passenger_count = {passenger_count})
    AND (pickup_ntaname = '{pickup_ntaname}')
    AND (dropoff_ntaname = '{dropoff_ntaname}');

--- 168h  window
SELECT count(*) as count_168h,
    avg(trip_duration) as avg_trip_duration_168h,
    avg(trip_distance) as avg_trip_distance_168h,
    avg(fare_amount) as avg_fare_amount_168h,
    avg(tip_amount) as avg_tip_amount_168h,
    avg(total_amount) as avg_total_amount_168h,
    sum(trip_duration) as sum_trip_duration_168h,
    sum(trip_distance) as sum_trip_distance_168h,
    sum(fare_amount) as sum_fare_amount_168h,
    sum(tip_amount) as sum_tip_amount_168h,
    sum(total_amount) as sum_total_amount_168h,
    stddevPop(trip_duration) as std_trip_duration_168h,
    stddevPop(trip_distance) as std_trip_distance_168h,
    stddevPop(fare_amount) as std_fare_amount_168h,
    stddevPop(tip_amount) as std_tip_amount_168h,
    stddevPop(total_amount) as std_total_amount_168h,
    varPop(trip_duration) as var_trip_duration_168h,
    varPop(trip_distance) as var_trip_distance_168h,
    varPop(fare_amount) as var_fare_amount_168h,
    varPop(tip_amount) as var_tip_amount_168h,
    varPop(total_amount) as var_total_amount_168h,
    min(trip_duration) as min_trip_duration_168h,
    min(trip_distance) as min_trip_distance_168h,
    min(fare_amount) as min_fare_amount_168h,
    min(tip_amount) as min_tip_amount_168h,
    min(total_amount) as min_total_amount_168h,
    max(trip_duration) as max_trip_duration_168h,
    max(trip_distance) as max_trip_distance_168h,
    max(fare_amount) as max_fare_amount_168h,
    max(tip_amount) as max_tip_amount_168h,
    max(total_amount) as max_total_amount_168h,
    median(trip_duration) as median_trip_duration_168h,
    median(trip_distance) as median_trip_distance_168h,
    median(fare_amount) as median_fare_amount_168h,
    median(tip_amount) as median_tip_amount_168h,
    median(total_amount) as median_total_amount_168h
FROM trips
WHERE (
        pickup_datetime >= (
            toDateTime('{pickup_datetime}') - toIntervalHour(168)
        )
    )
    AND (pickup_datetime < '{pickup_datetime}')
    AND (dropoff_datetime <= '{pickup_datetime}')
    AND (passenger_count = {passenger_count})
    AND (pickup_ntaname = '{pickup_ntaname}')
    AND (dropoff_ntaname = '{dropoff_ntaname}');