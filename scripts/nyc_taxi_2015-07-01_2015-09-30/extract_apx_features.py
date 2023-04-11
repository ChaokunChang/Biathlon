from extract_features import *


class SimpleParser(Tap):
    sampling_rate: float = 0.1


if __name__ == '__main__':
    args = SimpleParser().parse_args()
    sampling_rate = args.sampling_rate
    feature_dir = os.path.join(data_dir, f'apx_features_{sampling_rate}')
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    df = pd.read_csv(os.path.join(data_dir, 'requests_08-01_08-15.csv'))
    df.head()

    sql_template = sql_template_example.replace(
        'trips', f'trips_w_samples SAMPLE {sampling_rate}')

    # extract features and save to csv
    # df = df.iloc[:10000]
    extractor_1 = FeatureExtractor(sql_template=sql_template, interval_hours=1)
    agg_feas_1 = extractor_1.apply_on(df)
    agg_feas_1.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_1.csv'), index=False)

    extractor_2 = FeatureExtractor(
        sql_template=sql_template, interval_hours=24)
    agg_feas_2 = extractor_2.apply_on(df)
    agg_feas_2.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_2.csv'), index=False)

    extractor_3 = FeatureExtractor(
        sql_template=sql_template, interval_hours=24*7)
    agg_feas_3 = extractor_3.apply_on(df)
    agg_feas_3.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.agg_feas_3.csv'), index=False)

    # merge three agg features on trip_id
    all_feas = df.merge(agg_feas_1, on='trip_id').merge(
        agg_feas_2, on='trip_id').merge(agg_feas_3, on='trip_id')
    all_feas.to_csv(os.path.join(
        feature_dir, 'requests_08-01_08-15.feas.csv'), index=False)
