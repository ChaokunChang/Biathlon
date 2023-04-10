clickhouse client --queries-file=sql/create_trips.sql
clickhouse client --queries-file=sql/clean_trips.sql
clickhouse client --queries-file=sql/create_trips_w_samples.sql
clickhouse client --queries-file=sql/trips_features_qcs1-hour.sql
clickhouse client --queries-file=sql/trips_features_qcs1-day.sql
clickhouse client --queries-file=sql/trips_features_qcs1-week.sql
