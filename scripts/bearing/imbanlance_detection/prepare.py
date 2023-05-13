# importing necessary libraries
from sklearn.pipeline import Pipeline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
import xgboost as xgb
import catboost as cb
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import plotly.express as px
import plotly.graph_objects as go
import optuna
import joblib
import clickhouse_connect

HOME_DIR: str = '/home/ckchang/ApproxInfer'  # project home
DATA_HOME: str = os.path.join(HOME_DIR, 'data')  # data home

RESULTS_HOME: str = os.path.join(HOME_DIR, 'results')  # results home
LOG_DIR: str = os.path.join(HOME_DIR, 'logs')  # log home

bearing_data_dir = os.path.join(DATA_HOME, 'bearing')
bearing_1st_dir = os.path.join(
    bearing_data_dir, '1st_test', '1st_test')  # 2156 files
bearing_2nd_dir = os.path.join(
    bearing_data_dir, '2nd_test', '2nd_test')  # 984 files
bearing_3rd_dir = os.path.join(
    bearing_data_dir, '3rd_test', '4th_test', 'txt')  # 6324 files

selected_features = ['B_mean', 'B_std', 'B_skew', 'B_kurtosis', 'B_entropy',
                     'B_rms', 'B_max', 'B_p2p', 'B_crest', 'B_clearence', 'B_shape', 'B_impulse']
job_dir = os.path.join(bearing_data_dir, 'status_classification')

selected_features = ['B_std', 'B_clearence', 'B_mean',
                     'B_rms', 'B_shape', 'B_max', 'B_impulse']
job_dir = os.path.join(bearing_data_dir, 'status_classification_v2')

selected_features = ['B_std', 'B_clearence', 'B_kurtosis', 'B_mean', 'B_rms']
job_dir = os.path.join(bearing_data_dir, 'status_classification_v3')

selected_features = ['B_std', 'B_clearence', 'B_mean', 'B_rms']
job_dir = os.path.join(bearing_data_dir, 'status_classification_v4')

selected_features = ['B_mean', 'B_std', 'B_skew', 'B_kurtosis', 'B_rms',
                     'B_max', 'B_p2p', 'B_crest', 'B_clearence', 'B_shape', 'B_impulse']
job_dir = os.path.join(bearing_data_dir, 'status_classification_noentropy')

# selected_features = ['B_std', 'B_clearence', 'B_mean', 'B_rms', 'B_shape']
# job_dir = os.path.join(bearing_data_dir, 'status_classification_easy')


def prepare_keys_features_labels():
    X_y = pd.read_csv(os.path.join(bearing_data_dir, "set1_with_labels.csv"))

    B1_cols = ['time'] + [col for col in X_y.columns if "B1" in col]
    B2_cols = ['time'] + [col for col in X_y.columns if "B2" in col]
    B3_cols = ['time'] + [col for col in X_y.columns if "B3" in col]
    B4_cols = ['time'] + [col for col in X_y.columns if "B4" in col]

    B1 = X_y[B1_cols]
    B2 = X_y[B2_cols]
    B3 = X_y[B3_cols]
    B4 = X_y[B4_cols]

    # add new column to B1 to indicate the class
    # insert column after time
    B1.insert(1, 'bid', 1)
    B2.insert(1, 'bid', 2)
    B3.insert(1, 'bid', 3)
    B4.insert(1, 'bid', 4)

    cols = ['time', 'bid'] + ['Bx_mean', 'Bx_std', 'Bx_skew', 'Bx_kurtosis', 'Bx_entropy', 'Bx_rms', 'Bx_max', 'Bx_p2p', 'Bx_crest', 'Bx_clearence', 'Bx_shape', 'Bx_impulse',
                              'By_mean', 'By_std', 'By_skew', 'By_kurtosis', 'By_entropy', 'By_rms', 'By_max', 'By_p2p', 'By_crest', 'By_clearence', 'By_shape', 'By_impulse',
                              'class']
    B1.columns = cols
    B2.columns = cols
    B3.columns = cols
    B4.columns = cols
    final_data = pd.concat([B1, B2, B3, B4], axis=0, ignore_index=True)
    final_data.describe()
    final_data.to_csv(os.path.join(
        job_dir, 'keys_features_labels.csv'), index=False)


def get_splitted_data():
    final_data = pd.read_csv(os.path.join(job_dir, "keys_features_labels.csv"))
    X = final_data.copy()
    y = X.pop("class")
    times = X.pop("time")
    bids = X.pop("bid")
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y = pd.DataFrame(y, columns=["class"])

    time_features_list = ["mean", "std", "skew", "kurtosis", "entropy",
                          "rms", "max", "p2p", "crest", "clearence", "shape", "impulse"]
    x_axis_cols = ["Bx_"+tf for tf in time_features_list]
    # print(x_axis_cols)
    X_x = X.copy()
    X_x = X[x_axis_cols]
    cols = ['B_mean', 'B_std', 'B_skew', 'B_kurtosis', 'B_entropy',
            'B_rms', 'B_max', 'B_p2p', 'B_crest', 'B_clearence', 'B_shape', 'B_impulse']

    X_x.columns = cols

    X_x = X_x[selected_features]

    X_x_train, X_x_test, y_train, y_test, times_train, times_test, bids_train, bids_test = train_test_split(
        X_x, y, times, bids, test_size=0.3, random_state=1)
    return X_x_train, X_x_test, y_train, y_test, times_train, times_test, bids_train, bids_test


def prepare_test_data():
    X_x_train, X_x_test, y_train, y_test, times_train, times_test, bids_train, bids_test = get_splitted_data()
    test_reqs = pd.concat([times_test, bids_test], axis=1)
    test_reqs_w_labels = pd.concat([test_reqs, y_test], axis=1)

    # save the test X_x, y, times, and bids
    X_x_test.to_csv(os.path.join(job_dir, 'test_features.csv'), index=False)
    y_test.to_csv(os.path.join(job_dir, 'test_labels.csv'), index=False)
    test_reqs.to_csv(os.path.join(job_dir, 'test_requests.csv'), index=False)
    test_reqs_w_labels.to_csv(os.path.join(
        job_dir, 'test_reqs_w_labels.csv'), index=False)

    times_test.to_csv(os.path.join(job_dir, 'test_times.csv'), index=False)
    bids_test.to_csv(os.path.join(job_dir, 'test_bids.csv'), index=False)


def prepare_model():
    X_x_train, X_x_test, y_train, y_test, times_train, times_test, bids_train, bids_test = get_splitted_data()

    def objective(trial):
        xgb_params = dict(
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-4, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
            subsample=trial.suggest_float("subsample", 0.2, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
        )
        xgb_cl = xgb.XGBClassifier(
            booster='gbtree',
            tree_method='gpu_hist',
            use_label_encoder=False,
            **xgb_params)
        xgb_cl.fit(X_x_train, y_train)
        preds = xgb_cl.predict(X_x_test)
        accuracy_score(y_test, preds)
        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    xgb_params = study.best_params

    final_model = xgb.XGBClassifier(use_label_encoder=False,
                                    **xgb_params)
    final_model.fit(X_x_train, y_train)
    preds = final_model.predict(X_x_test)
    print("best model acc: ", accuracy_score(y_test, preds))
    joblib.dump(final_model, os.path.join(job_dir, 'best_model.pkl'))


def prepare_pipeline():
    # first, let's load the saved model and make a pipeline with the model, then save the pipeline
    # load the model
    model = joblib.load(os.path.join(job_dir, 'best_model.pkl'))
    # make pipeline with the model
    pipeline = Pipeline([('model', model)])
    # save the pipeline
    joblib.dump(pipeline, os.path.join(job_dir, 'pipeline.pkl'))

    X_x_train, X_x_test, y_train, y_test, times_train, times_test, bids_train, bids_test = get_splitted_data()
    # compuate feature importance and save
    fnames, imps = model.feature_names_in_, model.feature_importances_
    fimps = pd.DataFrame({'fname': fnames, 'importance': imps})
    fimps.to_csv(os.path.join(job_dir, 'feature_importances.csv'), index=False)


def prepare_online_table():
    """
    Create a table "bearing_online" for the X channel records in 1st test
    The table schema will be:
    rid UInt64, # record id, auto increment
    bid UInt8, # bearing id
    pid UInt8, # partiton id, used for sampling, i.e. our min sampling granularity is 1%
    timestamp DateTime, # timestamp
    signal Float32 # signal value

    The data records will be inserted into the table from anothe table with different schema: bearing
    The scehma of bearing is:
    timestamp DateTime, # timestamp
    B1X Float32, # signal value of bid=1 X channel
    B1Y Float32, # signal value of bid=1 Y channel
    B2X Float32, # signal value of bid=2 X channel
    B2Y Float32, # signal value of bid=2 Y channel
    B3X Float32, # signal value of bid=3 X channel
    B3Y Float32, # signal value of bid=3 Y channel
    B4X Float32, # signal value of bid=4 X channel
    B4Y Float32, # signal value of bid=4 Y channel
    """
    table_name = 'bearing_online'
    dbconn = clickhouse_connect.get_client(
        host='localhost', port=0, username='default', password='', session_id=f'session_{table_name}')
    dbconn.command(f"DROP TABLE IF EXISTS {table_name}")
    dbconn.command(
        f"CREATE TABLE IF NOT EXISTS {table_name} (rid UInt64, bid UInt8, pid UInt8, timestamp DateTime, signal Float32) ENGINE = MergeTree() ORDER BY (bid, pid, timestamp) SETTINGS index_granularity = 1024")
    # insert data into the table from bearing
    for bid in range(1, 5):
        cnt = dbconn.command(f'SELECT count(*) FROM {table_name}')
        print(f'cnt={cnt}')
        sql = """
                INSERT INTO {table_name} (rid, bid, pid, timestamp, signal)
                SELECT ({size} + row_number() OVER (order by timestamp)) as rid, 
                {bid} as bid, 
                rand() % 100 as pid,
                timestamp as timestamp, 
                B{bid}X as signal
                FROM bearing
                """.format(table_name=table_name, size=cnt, bid=bid)
        dbconn.command(sql)


if __name__ == "__main__":
    prepare_keys_features_labels()
    prepare_test_data()
    prepare_model()
    prepare_pipeline()
    # prepare_online_table()
    pass
