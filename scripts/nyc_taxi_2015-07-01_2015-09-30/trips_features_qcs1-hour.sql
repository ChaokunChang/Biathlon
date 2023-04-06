select trip_id,
    passenger_count,
    avg(trip_distance) over w as avg_dist,
    from trips WINDOW w AS (
        PARTITION BY passenger_count
        ORDER BY pickup_datetime RANGE BETWEEN (60 * 60) PRECEDING AND CURRENT ROW
    ) INTO OUTFILE '/home/ckchang/dataset/nyc_taxi/features/trips_features_qcs1-hour.csv' FORMAT CSVWithNames;