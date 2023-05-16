-- There will be 8 queries for 8 sensors

-- Example queries for sensor 0
-- WITH toInt32(250000*0.1) AS offset, toInt32(250000*0.1) AS nsamples
-- SELECT avg(sensor_0) as sensor_0_mean
-- FROM machinery.sensors
-- WHERE bid=1 AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_0) as sensor_0_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_1) as sensor_1_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_2) as sensor_2_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_3) as sensor_3_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_4) as sensor_4_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_5) as sensor_5_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_6) as sensor_6_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);

WITH toInt32(250000*{sample_offset}) AS offset, toInt32(250000*{sample}) AS nsamples
SELECT avg(sensor_7) as sensor_7_mean
FROM machinery.sensors
WHERE bid={bid} AND pid >= offset AND pid < (offset+nsamples);