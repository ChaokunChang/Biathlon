-- Query on original table with index
select avg(trip_distance) as avg_dist,
    varPop(trip_distance) as var_dist,
    min(trip_distance) as min_dist,
    max(fare_amount) as max_fare,
    avg(fare_amount) as avg_fare,
    varPop(fare_amount) as var_fare,
    min(fare_amount) as min_fare,
    max(fare_amount) as max_fare,
    avg(tip_amount) as avg_tip,
    varPop(tip_amount) as var_tip,
    min(tip_amount) as min_tip,
    max(tip_amount) as max_tip,
    avg(extra) as avg_extra,
    varPop(extra) as var_extra,
    min(extra) as min_extra,
    max(extra) as max_extra,
    avg(total_amount) as avg_total,
    varPop(total_amount) as var_total,
    min(total_amount) as min_total,
    max(total_amount) as max_total,
    count(*) as count
from trips
where passenger_count = 1
    AND pickup_datetime between '2015-08-02 04:47:25' AND '2015-08-03 04:47:25';
-- Query on original table with index and limit
select avg(trip_distance) as avg_dist,
    varPop(trip_distance) as var_dist,
    min(trip_distance) as min_dist,
    max(fare_amount) as max_fare,
    avg(fare_amount) as avg_fare,
    varPop(fare_amount) as var_fare,
    min(fare_amount) as min_fare,
    max(fare_amount) as max_fare,
    avg(tip_amount) as avg_tip,
    varPop(tip_amount) as var_tip,
    min(tip_amount) as min_tip,
    max(tip_amount) as max_tip,
    avg(extra) as avg_extra,
    varPop(extra) as var_extra,
    min(extra) as min_extra,
    max(extra) as max_extra,
    avg(total_amount) as avg_total,
    varPop(total_amount) as var_total,
    min(total_amount) as min_total,
    max(total_amount) as max_total,
    count(*) as count
from (
        select *
        from trips
        where passenger_count = 1
            AND pickup_datetime between '2015-08-02 04:47:25' AND '2015-08-03 04:47:25'
        limit 19502
    );
-- Query on full table with sample installed
select avg(trip_distance) as avg_dist,
    varPop(trip_distance) as var_dist,
    min(trip_distance) as min_dist,
    max(fare_amount) as max_fare,
    avg(fare_amount) as avg_fare,
    varPop(fare_amount) as var_fare,
    min(fare_amount) as min_fare,
    max(fare_amount) as max_fare,
    avg(tip_amount) as avg_tip,
    varPop(tip_amount) as var_tip,
    min(tip_amount) as min_tip,
    max(tip_amount) as max_tip,
    avg(extra) as avg_extra,
    varPop(extra) as var_extra,
    min(extra) as min_extra,
    max(extra) as max_extra,
    avg(total_amount) as avg_total,
    varPop(total_amount) as var_total,
    min(total_amount) as min_total,
    max(total_amount) as max_total,
    count(*) as count
from trips_w_samples
where passenger_count = 1
    AND pickup_datetime between '2015-08-02 04:47:25' AND '2015-08-03 04:47:25';
-- Query on sampled table
select avg(trip_distance) as avg_dist,
    varPop(trip_distance) as var_dist,
    min(trip_distance) as min_dist,
    max(fare_amount) as max_fare,
    avg(fare_amount) as avg_fare,
    varPop(fare_amount) as var_fare,
    min(fare_amount) as min_fare,
    max(fare_amount) as max_fare,
    avg(tip_amount) as avg_tip,
    varPop(tip_amount) as var_tip,
    min(tip_amount) as min_tip,
    max(tip_amount) as max_tip,
    avg(extra) as avg_extra,
    varPop(extra) as var_extra,
    min(extra) as min_extra,
    max(extra) as max_extra,
    avg(total_amount) as avg_total,
    varPop(total_amount) as var_total,
    min(total_amount) as min_total,
    max(total_amount) as max_total,
    count(*) as count
from trips_w_samples sample 1 / 10
where passenger_count = 1
    AND pickup_datetime between '2015-08-02 04:47:25' AND '2015-08-03 04:47:25';