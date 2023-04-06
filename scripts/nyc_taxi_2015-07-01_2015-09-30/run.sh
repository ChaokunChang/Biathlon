clickhouse client --queries-file=create_trips.sql
clickhouse client --queries-file=clean_trips.sql
clickhouse client --queries-file=create_trips_w_samples.sql
clickhouse client --queries-file=trips_features_qcs1-hour.sql
clickhouse client --queries-file=trips_features_qcs1-day.sql
clickhouse client --queries-file=trips_features_qcs1-week.sql
