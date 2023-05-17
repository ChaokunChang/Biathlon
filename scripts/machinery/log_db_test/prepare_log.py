import numpy as np
import pandas as pd
import os
import time
import glob
import pickle
import joblib
import matplotlib.pyplot as plt
from tap import Tap
import seaborn as sns
from skimage.transform import resize
from sklearn import pipeline, metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from xgboost import XGBClassifier, XGBRegressor
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm
import clickhouse_connect

DATA_HOME = "/home/ckchang/ApproxInfer/data"
RESULTS_HOME = "/home/ckchang/ApproxInfer/results"
filecols = [f'sensor_{i}' for i in range(8)]

datadir = os.path.join(DATA_HOME, 'machinery')
database = 'machinery_more_log'
work_dir = os.path.join(RESULTS_HOME, database)
spectograms_dir = os.path.join(work_dir, 'spectograms')
os.makedirs(spectograms_dir, exist_ok=True)


class MachineryFaultArgs(Tap):
    # binary_classification, multi_classification
    task: str = 'binary_classification'
    aggs: list[str] = ['mean']  # aggregation functions
    model_name: str = 'mlp'  # mlp, svm, dt, knn, rf, lgbm, xgb

    def process_args(self) -> None:
        self.task += f'_{self.model_name}'


def get_all_file_names():
    normal_file_names = glob.glob(os.path.join(
        datadir, 'normal', 'normal', '*.csv'))
    imnormal_file_names_6g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '6g', '*.csv'))
    imnormal_file_names_10g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '10g', '*.csv'))
    imnormal_file_names_15g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '15g', '*.csv'))
    imnormal_file_names_20g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '20g', '*.csv'))
    imnormal_file_names_25g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '25g', '*.csv'))
    imnormal_file_names_30g = glob.glob(os.path.join(
        datadir, 'imbalance', 'imbalance', '30g', '*.csv'))

    return normal_file_names, imnormal_file_names_6g, imnormal_file_names_10g, imnormal_file_names_15g, imnormal_file_names_20g, imnormal_file_names_25g, imnormal_file_names_30g


def load_data():
    normal_file_names, imnormal_file_names_6g, imnormal_file_names_10g, imnormal_file_names_15g, imnormal_file_names_20g, imnormal_file_names_25g, imnormal_file_names_30g = get_all_file_names()

    normal_df = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=0) for f in normal_file_names), ignore_index=True)
    imnormal_df_6g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=1) for f in imnormal_file_names_6g), ignore_index=True)
    imnormal_df_10g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=2) for f in imnormal_file_names_10g), ignore_index=True)
    imnormal_df_15g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=3) for f in imnormal_file_names_15g), ignore_index=True)
    imnormal_df_20g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=4) for f in imnormal_file_names_20g), ignore_index=True)
    imnormal_df_25g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=5) for f in imnormal_file_names_25g), ignore_index=True)
    imnormal_df_30g = pd.concat((pd.read_csv(f, names=filecols).assign(
        src=f, label=6) for f in imnormal_file_names_30g), ignore_index=True)

    df = pd.concat([normal_df, imnormal_df_6g, imnormal_df_10g,
                    imnormal_df_15g, imnormal_df_20g, imnormal_df_25g, imnormal_df_30g], ignore_index=True)

    # split each file into segments with 50000 records
    # we need to assign a segment id to each row
    seg_size = 50000
    df['seg_id'] = df.groupby(['src', 'label']).cumcount() // seg_size
    print(df.head())
    return df


def prepare_file2bid():
    normal_file_names, imnormal_file_names_6g, imnormal_file_names_10g, imnormal_file_names_15g, imnormal_file_names_20g, imnormal_file_names_25g, imnormal_file_names_30g = get_all_file_names()
    all_file_names = normal_file_names + imnormal_file_names_6g + imnormal_file_names_10g + \
        imnormal_file_names_15g + imnormal_file_names_20g + \
        imnormal_file_names_25g + imnormal_file_names_30g

    # save the file names to a file as csv, the first column as filename, the second as id
    df = pd.DataFrame({'src': [f for f in all_file_names for _ in range(5)],
                       'seg_id': list(range(5)) * len(all_file_names),
                      'bid': range(5 * len(all_file_names))})
    df.to_csv(os.path.join(work_dir, 'file2bid.csv'), index=False)


def prepare_db(database):
    dbconn = clickhouse_connect.get_client(
        host='localhost', port=0, username='default', password='', session_id=f'session_{database}')

    def prepare_logdb(dbconn, database, table_name):
        normal_file_names, imnormal_file_names_6g, imnormal_file_names_10g, imnormal_file_names_15g, imnormal_file_names_20g, imnormal_file_names_25g, imnormal_file_names_30g = get_all_file_names()
        all_file_names = normal_file_names + imnormal_file_names_6g + imnormal_file_names_10g + \
            imnormal_file_names_15g + imnormal_file_names_20g + \
            imnormal_file_names_25g + imnormal_file_names_30g
        typed_signal = ', '.join([f'sensor_{i} Float32' for i in range(8)])

        for bid, src in tqdm(enumerate(all_file_names)):
            for i in range(5):
                tid = bid * 5 + i
                dbconn.command("""CREATE TABLE IF NOT EXISTS {database}.{table_name} 
                                ({typed_signal}) 
                                ENGINE = Log 
                                SETTINGS index_granularity = 32
                                """.format(database=database,
                                           table_name=f'{table_name}_{tid}',
                                           typed_signal=typed_signal))
                dbconn.command("""
                    INSERT INTO {database}.{table_name} 
                    SELECT {values}
                    FROM machinery_more.sensors
                    WHERE bid={bid}
                    """.format(database=database,
                               table_name=f'{table_name}_{tid}',
                               values=', '.join(
                                   [f'sensor_{i} AS sensor_{i}' for i in range(8)]),
                               bid=tid))

    def prepare_logdb_shuffle(dbconn, database, table_name):
        normal_file_names, imnormal_file_names_6g, imnormal_file_names_10g, imnormal_file_names_15g, imnormal_file_names_20g, imnormal_file_names_25g, imnormal_file_names_30g = get_all_file_names()
        all_file_names = normal_file_names + imnormal_file_names_6g + imnormal_file_names_10g + \
            imnormal_file_names_15g + imnormal_file_names_20g + \
            imnormal_file_names_25g + imnormal_file_names_30g
        typed_signal = ', '.join([f'sensor_{i} Float32' for i in range(8)])

        for bid, src in tqdm(enumerate(all_file_names)):
            for i in range(5):
                tid = bid * 5 + i
                dbconn.command("""CREATE TABLE IF NOT EXISTS {database}.{table_name} 
                                ({typed_signal}) 
                                ENGINE = Log 
                                SETTINGS index_granularity = 32
                                """.format(database=database,
                                           table_name=f'{table_name}_{tid}',
                                           typed_signal=typed_signal))
                dbconn.command("""
                    INSERT INTO {database}.{table_name} 
                    SELECT {values}
                    FROM machinery_more.sensors_shuffle
                    WHERE bid={bid}
                    """.format(database=database,
                               table_name=f'{table_name}_{tid}',
                               values=', '.join(
                                   [f'sensor_{i} AS sensor_{i}' for i in range(8)]),
                               bid=tid))

    # check whether the database exists
    if not dbconn.command(f"SHOW DATABASES LIKE '{database}'"):
        print(f'Create database {database}')
        dbconn.command(f'CREATE DATABASE {database}')
    else:
        return dbconn, database

    print(f'Create table {database}.sensors')
    table_name = 'sensors'
    prepare_logdb(dbconn, database, table_name)

    print(f'Create table {database}.sensors_shuffle')
    table_name = 'sensors_shuffle'
    prepare_logdb_shuffle(dbconn, database, table_name)


def compute_features(df: pd.DataFrame, nsample: int = 0):
    # compute exact features for model building
    # We aggregate the data of each file as a single data point
    # We compute the mean, std, min, max, and median of each sensor
    if nsample > 0:
        df = df.groupby(['src', 'label', 'seg_id'])[filecols].head(
            nsample).agg(['mean', 'std', 'min', 'max', 'median'])
        df.columns = df.columns.map('_'.join)
    else:
        df = df.groupby(['src', 'label', 'seg_id'])[filecols].agg(
            ['mean', 'std', 'min', 'max', 'median'])
        df.columns = df.columns.map('_'.join)
    return df


def compute_spectograms(x):
    def _compute(x, fs=44100, noverlap=128, img_size=(230, 230)):
        Sxx = plt.specgram(x, Fs=fs, noverlap=noverlap, cmap="rainbow")[0]
        return resize(Sxx, img_size)
    gsrc = x['src'].iloc[0]
    seg_id = x['seg_id'].iloc[0]
    srcname = os.path.basename(gsrc)
    pdir = os.path.dirname(gsrc)
    pname = os.path.basename(pdir)
    specdir = os.path.join(spectograms_dir, pname)
    os.makedirs(specdir, exist_ok=True)
    filenames = {}
    for col in filecols:
        spec = _compute(x[col])
        filename = os.path.join(
            specdir, srcname.replace('.csv', f'_seg_{seg_id}_{col}_spec.pkl'))
        with open(filename, 'wb') as f:
            pickle.dump(spec, f)
        # filenames.append(filename)
        filenames[f'{col}_spec'] = filename
    return pd.Series(filenames)


def prepare_features():
    if os.path.exists(os.path.join(work_dir, 'df.csv')):
        return pd.read_csv(os.path.join(work_dir, 'df.csv'))

    # load_data
    df = load_data()

    # compute features
    feas_df = compute_features(df).reset_index()

    # compute spectograms
    spec_df = df.groupby(['src', 'label', 'seg_id']).apply(
        compute_spectograms).reset_index()

    # combine the features and spectograms
    df = pd.merge(feas_df, spec_df, on=['src', 'label', 'seg_id'])

    # save the data
    feas_df.to_csv(os.path.join(work_dir, 'feas.csv'), index=False)
    spec_df.to_csv(os.path.join(work_dir, 'spec.csv'), index=False)
    df.to_csv(os.path.join(work_dir, 'df.csv'), index=False)

    return df


def prepare_feature_dataset(task='binary_classification', fcols=[f'{col}_mean' for col in filecols]):
    jobdir = os.path.join(work_dir, task)

    # load all features
    df = pd.read_csv(os.path.join(work_dir, 'df.csv'))

    # load file2bid
    file2bid = pd.read_csv(os.path.join(work_dir, 'file2bid.csv'))

    # add id to df
    df = pd.merge(df, file2bid, how='left', on=['src', 'seg_id'])

    # select features for this task
    df = df[['src', 'label', 'seg_id', 'bid'] + fcols]

    # if task is binary classification, we only keep normal and imnormal data
    # i.e. we set label bigger than 0 to 1
    if 'binary_classification' in task:
        df.loc[df['label'] > 0, 'label'] = 1

    # split the data into train and test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv(os.path.join(jobdir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(jobdir, 'test.csv'), index=False)

    # prepare test_requests.csv and test_labels.csv
    test_requests = test_df[['bid']]
    test_requests.to_csv(os.path.join(
        jobdir, 'test_requests.csv'), index=False)
    test_labels = test_df[['label']]
    test_labels.to_csv(os.path.join(jobdir, 'test_labels.csv'), index=False)

    return train_df, test_df


def compute_permuation_importance(pipe: Pipeline, X: pd.DataFrame, y, random_state=0, use_pipe_feature=True):
    # We can compute feature importance of model's feature, or pipeline's feature.
    if use_pipe_feature:
        imps = permutation_importance(pipe, X, y, n_repeats=10, max_samples=min(
            1000, len(X)), random_state=random_state, n_jobs=-1).importances_mean
    else:
        X = pipe[:-1].transform(X)
        imps = permutation_importance(pipe[-1], X, y, n_repeats=10, max_samples=min(
            1000, len(X)), random_state=0, n_jobs=-1).importances_mean
    print("permutation importance(original): ", imps)
    imps = np.abs(imps)
    return X.columns, imps


def _get_feature_importance(pipe: Pipeline, X, y):
    model = pipe[-1]
    if isinstance(model, (LGBMRegressor, LGBMClassifier)):
        return model.feature_name_, model.feature_importances_
    elif isinstance(model, (XGBRegressor, XGBClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (DecisionTreeRegressor, DecisionTreeClassifier)):
        return model.feature_names_in_, model.feature_importances_
    elif isinstance(model, (LinearRegression, LogisticRegression)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (SVC, SVR)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (KNeighborsRegressor, KNeighborsClassifier)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    elif isinstance(model, (MLPRegressor, MLPClassifier)):
        return compute_permuation_importance(pipe, X, y, random_state=0, use_pipe_feature=True)
    else:
        raise ValueError("model name not supported")


def prepare_pipeline(task='binary_classification', fcols=[f'{col}_mean' for col in filecols], model_name='mlp') -> Pipeline:
    jobdir = os.path.join(work_dir, task)

    # load train and test data
    train_df = pd.read_csv(os.path.join(jobdir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(jobdir, 'test.csv'))

    # create the pipeline
    if model_name == 'mlp':
        # model = MLPClassifier(solver='lbfgs', random_state=42)
        model = MLPClassifier(solver='adam', random_state=42,
                              max_iter=200, learning_rate_init=1e-1, verbose=True)
    elif model_name == 'svm':
        model = SVC(random_state=42)
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'lgbm':
        model = LGBMClassifier(random_state=42)
    elif model_name == 'xgb':
        model = XGBClassifier(random_state=42)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f'Unknown model type: {model_name}')

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('model', model)
    ])

    # train the model
    pipeline.fit(train_df[fcols], train_df['label'])

    # evaluate the model
    y_pred = pipeline.predict(test_df[fcols])
    print(metrics.classification_report(test_df['label'], y_pred))

    # save the model
    joblib.dump(pipeline, os.path.join(jobdir, 'pipeline.pkl'))

    # save feature importance of pipeline
    feature_names, feature_importances = _get_feature_importance(
        pipeline, train_df[fcols], train_df['label'])
    feature_importance_df = pd.DataFrame(
        {'fname': feature_names, 'importance': feature_importances})
    feature_importance_df.to_csv(os.path.join(
        jobdir, 'feature_importances.csv'), index=False)

    # save train features and test features
    train_df[fcols].to_csv(os.path.join(
        jobdir, 'train_features.csv'), index=False)
    test_df[fcols].to_csv(os.path.join(
        jobdir, 'test_features.csv'), index=False)

    return pipeline


if __name__ == '__main__':
    args = MachineryFaultArgs().parse_args()
    task = args.task
    fcols = [f'{col}_{agg}' for col in filecols for agg in args.aggs]
    model_name = args.model_name

    jobdir = os.path.join(work_dir, task)
    os.makedirs(jobdir, exist_ok=True)

    print(f'Prepare data for {task}')
    first_time = False
    if first_time:
        print(f'prepare file2bid ...')
        prepare_file2bid()
        print('Prepare features ......')
        prepare_features()
        print(f'prepare online database')
        prepare_db(database=database)

    print('Prepare feature dataset ......')
    prepare_feature_dataset(task, fcols)
    print('Prepare pipeline ......')
    prepare_pipeline(task, fcols, model_name)

    print('Done!')
