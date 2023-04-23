from shared import *
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
# pandarallel.initialize(progress_bar=True, nb_workers=2)


class DBConnector:
    def __init__(self, host='localhost', port=0, username='default', passwd='') -> None:
        self.host = host
        self.port = port
        self.username = username
        self.passwd = passwd
        # get current process id for identifying the session
        self.thread_id = os.getpid()
        self.session_time = time.time()
        self.session_id = f'session_{self.thread_id}_{self.session_time}'
        self.client = clickhouse_connect.get_client(
            host=self.host, port=self.port,
            username=self.username, password=self.passwd,
            session_id=self.session_id)

    def execute(self, sql):
        return self.client.query_df(sql)


class FeatureExtractor:
    def __init__(self, sql_template: str, key='trip_id'):
        self.sql_template = sql_template
        self.key = key

    def extract_once(self, x: pd.Series or pd.DataFrame):
        sql = self.sql_template.format(**x.to_dict())
        rows_df = DBConnector().execute(sql)
        if self.key not in rows_df.columns:
            rows_df[self.key] = x[self.key]
        aggregations = rows_df.iloc[0]
        return aggregations

    def extract_with_df(self, df: pd.DataFrame, parallel=True):
        st = time.time()
        if parallel:
            features = df.parallel_apply(self.extract_once, axis=1)
        else:
            features = df.apply(self.extract_once, axis=1)
        print(f'Elapsed time: {time.time() - st}')
        return features


if __name__ == "__main__":
    args = SimpleParser().parse_args()

    reqs = pd.read_csv(args.req_src)

    feature_dir = args.feature_dir

    for i, sql_template in enumerate(args.sql_templates):
        print(f'Extracting features with sql template: {sql_template}')
        extractor = FeatureExtractor(sql_template, key=args.keycol)
        features = extractor.extract_with_df(reqs, parallel=True)
        save_features(features, feature_dir,
                      output_name=f'{args.ffile_prefix}_{i}.csv')

    # combine features in to one file, remove duplicate columns
    features = pd.concat([pd.read_csv(os.path.join(
        feature_dir, f'{args.ffile_prefix}_{i}.csv')) for i in range(len(args.sql_templates))], axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    save_features(features, feature_dir,
                  output_name=f'{args.ffile_prefix}.csv')
