select *
from trips INTO OUTFILE '/home/ckchang/ApproxInfer/data/nyc_taxi_2015-07-01_2015-09-30/features/trips_features_raw.csv' FORMAT CSVWithNames;