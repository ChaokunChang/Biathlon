select min(pickup_datetime),
    max(pickup_datetime),
    min(dropoff_datetime),
    max(dropoff_datetime)
from trips FORMAT Vertical;
/*
 SELECT
 min(pickup_datetime),
 max(pickup_datetime),
 min(dropoff_datetime),
 max(dropoff_datetime)
 FROM trips
 FORMAT Vertical
 
 Query id: 5db7eed5-b662-472a-a37c-a100adbadb04
 
 Row 1:
 ──────
 min(pickup_datetime):  2015-07-01 00:00:00
 max(pickup_datetime):  2015-09-30 23:59:59
 min(dropoff_datetime): 2010-01-01 02:57:21
 max(dropoff_datetime): 2015-12-09 07:33:08
 
 1 row in set. Elapsed: 0.070 sec. Processed 20.00 million rows, 160.00 MB (287.74 million rows/s., 2.30 GB/s.)
 */