-- Assume all aggregation functions are available,
-- If not, create as user defined function
WITH toInt32(100*{sample}) AS max_pid
SELECT 
    avgAbs(signal) AS B_mean,
    stddevPop(signal) AS B_std,
    skewPop(signal) AS B_skew,
    kurtPop(signal) AS B_kurtosis,
    entropy(signal) AS B_entropy,
    rmsPop(signal) AS B_rms,
    maxAbs(signal) AS B_max,
    p2pPop(signal) AS B_p2p,
    crestFactorPop(signal) AS B_crest,
    clearenceFactorPop(signal) AS B_clearence,
    shapFactorPop(signal) AS B_shape,
    impulsePop(signal) AS B_impulse
FROM bearing_online
WHERE bid = {bid} AND pid < max_pid AND timestamp = toDateTime('{time}');

