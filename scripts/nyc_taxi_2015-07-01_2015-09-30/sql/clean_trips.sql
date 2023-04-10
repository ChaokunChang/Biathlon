-- clean the data. 
-- remove records with negative trip_duration, trip_distance, fare_amount, total_amount, and passenger_count, 
ALTER TABLE trips DELETE
WHERE trip_duration < 0
    OR trip_distance < 0
    OR fare_amount < 0
    OR extra < 0
    OR tip_amount < 0
    OR total_amount < 0
    OR passenger_count < 0;