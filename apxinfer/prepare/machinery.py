import os
import os.path as osp
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from tap import Tap
import glob
from tqdm import tqdm

from apxinfer.utils import DBConnector, create_model

EXP_HOME = '/home/ckchang/.cache/apxinf'


class PrepareStageArgs(Tap):
    task: str = 'machinery'  # task name
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation
    multi_class: bool = False  # whether the task is multi-class classification

    raw_data_dir: str = '/home/ckchang/ApproxInfer/data/machinery'

    def process_args(self) -> None:
        self.task_dir = os.path.join(EXP_HOME, self.task)
        self.model_dir = os.path.join(self.task_dir, self.model)
        self.exp_dir = os.path.join(self.model_dir, f'seed-{self.seed}')
        os.makedirs(self.exp_dir, exist_ok=True)


def raw_data_preprocessing(raw_data_dir: str, database: str, base_table_name: str, num_sensors: int, seed: int) -> None:
    db_client = DBConnector().client

    def create_tables():
        table_name = base_table_name
        print(f"Create tables {table_name} to store data from all (8x) sensors")
        typed_signal = ", ".join([f"sensor_{i} Float32" for i in range(8)])
        db_client.command(
            """CREATE TABLE IF NOT EXISTS {database}.{table_name}
                        (rid UInt64, label UInt32, tag String,
                            bid UInt32, pid UInt32,
                            {typed_signal})
                        ENGINE = MergeTree()
                        ORDER BY (bid, pid)
                        SETTINGS index_granularity = 32
                        """.format(
                database=database, table_name=table_name, typed_signal=typed_signal
            )
        )

        for i in range(num_sensors):
            sensor_table_name = f"{table_name}_sensor_{i}"
            print(f"Create tables {table_name} to store data from sensor_{i}")
            db_client.command(
                """CREATE TABLE IF NOT EXISTS {database}.{sensor_table_name}
                            (rid UInt64, label UInt32, tag String,
                                bid UInt32, pid UInt32,
                                sensor_{i} Float32)
                            ENGINE = MergeTree()
                            ORDER BY (bid, pid)
                            SETTINGS index_granularity = 32
                            """.format(
                    database=database, sensor_table_name=sensor_table_name, i=i
                )
            )

    def get_raw_data_files_list(data_dir: str) -> list:
        normal_file_names = glob.glob(os.path.join(data_dir, "normal", "normal", "*.csv"))
        imnormal_file_names_6g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "6g", "*.csv")
        )
        imnormal_file_names_10g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "10g", "*.csv")
        )
        imnormal_file_names_15g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "15g", "*.csv")
        )
        imnormal_file_names_20g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "20g", "*.csv")
        )
        imnormal_file_names_25g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "25g", "*.csv")
        )
        imnormal_file_names_30g = glob.glob(
            os.path.join(data_dir, "imbalance", "imbalance", "30g", "*.csv")
        )

        # concat all file names
        file_names = (
            normal_file_names
            + imnormal_file_names_6g
            + imnormal_file_names_10g
            + imnormal_file_names_15g
            + imnormal_file_names_20g
            + imnormal_file_names_25g
            + imnormal_file_names_30g
        )

        return file_names

    def ingest_data(raw_data_dir: str):
        all_file_names = get_raw_data_files_list(raw_data_dir)
        table_name = "machinery_shuffle"
        file_nrows = 250000
        segments_per_file = 5
        segment_nrows = file_nrows // segments_per_file

        print(f"Ingest data to tables {table_name}")
        for bid, src in tqdm(enumerate(all_file_names)):
            filename = os.path.basename(src)
            tag = filename.split(".")[0]
            dirname = os.path.basename(os.path.dirname(src))
            label = ["normal", "6g", "10g", "15g", "20g", "25g", "30g"].index(dirname)
            cnt = db_client.command(f"SELECT count(*) FROM {database}.{table_name}")
            # print(f'dbsize={cnt}')
            command = """
                    clickhouse-client \
                        --query \
                        "INSERT INTO {database}.{table_name} \
                            SELECT ({cnt} + row_number() OVER ()) AS rid, \
                                    {label} AS label, {tag} AS tag, {bid} + floor(((row_number() OVER ()) - 1)/{segment_nrows}) AS bid,
                                    ((row_number() OVER ()) - 1) % {segment_nrows} AS pid,
                                    {values} \
                            FROM input('{values_w_type}') \
                            FORMAT CSV" \
                            < {filepath}
                    """.format(
                database=database,
                table_name=table_name,
                cnt=cnt,
                label=label,
                tag=tag,
                bid=bid * segments_per_file,
                segment_nrows=segment_nrows,
                values=", ".join([f"sensor_{i} AS sensor_{i}" for i in range(8)]),
                values_w_type=", ".join([f"sensor_{i} Float32" for i in range(8)]),
                filepath=src,
            )
            # print(command)
            os.system(command)

        for i in range(num_sensors):
            sensor_table_name = f"{table_name}_sensor_{i}"
            print(f"Ingest data to tables {sensor_table_name}")
            db_client.command(
                f"INSERT INTO {database}.{sensor_table_name} SELECT rid, label, tag, bid, pid, sensor_{i} FROM {database}.{table_name}"
            )

    if not db_client.command(f"SHOW DATABASES LIKE '{database}'"):
        print(f"Create database {database}")
        db_client.command(f"CREATE DATABASE {database}")
        create_tables()
        ingest_data(raw_data_dir)
    else:
        print(f'Database "{database}" already exists')


def train_valid_test_split(data: pd.DataFrame, train_ratio: float, valid_ratio: float, seed: int):
    # shuffle the data
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    # calculate the number of rows for each split
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    # n_test = n_total - n_train - n_valid

    # split the data
    train_set = data[:n_train]
    valid_set = data[n_train:n_train + n_valid]
    test_set = data[n_train + n_valid:]

    return train_set, valid_set, test_set


def run(raw_data_dir: str, save_dir: str, seed: int, binary_classification: bool):
    """ in test.py: run, we create synthetic pipeline to test our system
    1. data pre-processing and save to clickhouse, with sampling supported // this is ingored in test.py
    2. prepare all requests and labels and save
        requests : (request_id, f1: a float from 0 to 1000, f2: a float from 0 to 1, f3: a float from 0 to 10)
        labels   : (f1+f2+f3 // 2).astype(int)
    3. extract all (exact) features and save along with request_id
        feature = request['f1'], request['f2'], request['f3']
    4. split the requests, labels, and features into train_set, valid_set, and test_set, 0.5, 0.3, 0.2
            For classification task, make sure stratified sample
    5. train model with train_set, save the model and save with joblib
    6. evaluation model with valid_set and test_set, save into json
    """

    # data pre-processing and save to clickhouse
    database = 'xip'
    table_name = 'machinery_shuffle'
    num_sensors = 8
    raw_data_preprocessing(raw_data_dir, database, table_name, num_sensors, seed)

    # prepare all requests and labels and features and save
    fnames = [f'feature_{i}' for i in range(num_sensors)]
    db_client = DBConnector().client
    sql = """
        SELECT bid as request_bid, label as request_label,
        avg(sensor_0) as feature_0,
        avg(sensor_1) as feature_1,
        avg(sensor_2) as feature_2,
        avg(sensor_3) as feature_3,
        avg(sensor_4) as feature_4,
        avg(sensor_5) as feature_5,
        avg(sensor_6) as feature_6,
        avg(sensor_7) as feature_7
        FROM {database}.{table_name} GROUP BY bid, label
        ORDER BY bid
    """.format(database=database, table_name=table_name)
    requests: pd.DataFrame = db_client.query_df(sql)
    num_reqs = len(requests)
    requests.insert(0, 'request_id', list(range(num_reqs)))
    if binary_classification:
        requests['request_label'] = (requests['request_label'] > 0).astype(int)

    # split the requests into train_set, valid_set, and test_set
    train_set, valid_set, test_set = train_valid_test_split(requests, 0.5, 0.3, seed)

    # train model with train_set, save the model and save with joblib
    model = create_model('classifier', 'lgbm', random_state=seed)
    ppl = Pipeline(
        [
            ("model", model)
        ]
    )
    ppl.fit(train_set[fnames], train_set['request_label'])
    joblib.dump(ppl, osp.join(save_dir, "pipeline.pkl"))

    # 6. evaluation model with valid_set and test_set, save into json
    valid_pred = ppl.predict(valid_set[fnames])
    test_pred = ppl.predict(test_set[fnames])
    valid_set['ppl_pred'] = valid_pred
    test_set['ppl_pred'] = test_pred
    valid_set.to_csv(osp.join(save_dir, 'valid_set.csv'), index=False)
    test_set.to_csv(osp.join(save_dir, 'test_set.csv'), index=False)

    valid_acc = metrics.accuracy_score(valid_set['request_label'], valid_set['ppl_pred'])
    test_acc = metrics.accuracy_score(test_set['request_label'], test_set['ppl_pred'])
    print(f"valid_acc: {valid_acc}, test_acc: {test_acc}")


if __name__ == '__main__':
    args = PrepareStageArgs().parse_args()
    save_dir = os.path.join(args.exp_dir, 'prepare')
    print(f'Save dir: {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    run(args.raw_data_dir, save_dir=save_dir, seed=args.seed,
        binary_classification=not args.multi_class)
