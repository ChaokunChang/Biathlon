from extract_features import *

if __name__ == '__main__':
    args = SimpleParser().parse_args()
    num_reqs = args.num_reqs

    sql_template = """
    select {aggs} from trips 
    where (pickup_datetime >= (toDateTime('{pickup_datetime}') - toIntervalHour({hours}))) 
    AND (pickup_datetime < '{pickup_datetime}') AND (dropoff_datetime <= '{pickup_datetime}') 
    AND (passenger_count = {passenger_count}) AND (pickup_ntaname = '{pickup_ntaname}')
    """

    output_name = 'features'
    if args.sampling_rate > 0 and args.sampling_rate < 1.0:
        sql_template = sql_template_example.replace(
            'trips', f'trips_w_samples SAMPLE {args.sampling_rate}')
        output_name = f'features2_apx_{args.sampling_rate}'

    if num_reqs > 0:
        feature_dir = os.path.join(
            data_dir, f'test_{num_reqs}xReqs', output_name)
    else:
        feature_dir = os.path.join(data_dir, output_name)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    df = pd.read_csv(os.path.join(data_dir, 'requests_08-01_08-15.csv'))
    df.head()
    # extract features and save to csv
    # sample 10000 from df
    if num_reqs > 0:
        df = df.sample(n=num_reqs, random_state=0)

    extractor_1 = FeatureExtractor(sql_template, interval_hours=1)
    agg_feas_1 = extractor_1.apply_on(df)
    agg_feas_1.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_1.csv'), index=False)

    extractor_2 = FeatureExtractor(sql_template, interval_hours=24)
    agg_feas_2 = extractor_2.apply_on(df)
    agg_feas_2.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_2.csv'), index=False)

    extractor_3 = FeatureExtractor(sql_template, interval_hours=24*7)
    agg_feas_3 = extractor_3.apply_on(df)
    agg_feas_3.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_3.csv'), index=False)

    # merge three agg features on trip_id
    all_feas = df.merge(agg_feas_1, on='trip_id').merge(
        agg_feas_2, on='trip_id').merge(agg_feas_3, on='trip_id')
    all_feas.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.feas.csv'), index=False)
