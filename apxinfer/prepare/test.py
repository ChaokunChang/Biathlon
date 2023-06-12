import os
import os.path as osp
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from lightgbm import LGBMClassifier

EXP_HOME = '/home/ckchang/.cache/apxinf'


def run(save_dir: str, seed):
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

    # 1. data pre-processing and save to clickhouse, with sampling supported // this is ingored in test.py

    # 2. prepare all requests and labels and save
    np.random.seed(seed)
    num_reqs = 1000
    requests = pd.DataFrame({
        'request_id': list(range(num_reqs)),
        'request_f1': np.random.normal(100., 100, num_reqs),
        'request_f2': np.random.normal(10., 1, num_reqs),
        'request_f3': np.random.normal(50., 10, num_reqs),
    })
    requests.to_csv(osp.join(save_dir, 'requests.csv'), index=False)

    labels = pd.DataFrame({
        'request_id': list(range(num_reqs)),
        'request_label': ((requests['request_f1'] + requests['request_f2'] + requests['request_f3']) % 2).astype(int)
    })
    labels.to_csv(osp.join(save_dir, 'labels.csv'), index=False)

    # 3. extract all (exact) features and save along with request_id
    features = pd.DataFrame({
        'request_id': list(range(num_reqs)),
        'feature_1': requests['request_f1'],
        'feature_2': requests['request_f2'],
        'feature_3': requests['request_f3'],
    })
    features.to_csv(osp.join(save_dir, 'features.csv'), index=False)

    # merge requests, features, labels
    requests = requests.merge(features, on='request_id')
    requests = requests.merge(labels, on='request_id')

    # 4. split the requests, labels, and features into train_set, valid_set, and test_set, 0.5, 0.3, 0.2
    # For classification task, make sure stratified sample
    train_set = requests.sample(frac=0.5, random_state=seed)
    valid_set = requests.drop(train_set.index).sample(frac=0.6, random_state=seed)
    test_set = requests.drop(train_set.index).drop(valid_set.index)

    # 5. train model with train_set, save the model and save with joblib
    model = LGBMClassifier(random_state=seed)
    ppl = Pipeline(
        [
            # ('scaler', StandardScaler()),
            ("model", model)
        ]
    )
    ppl.fit(train_set[['feature_1', 'feature_2', 'feature_3']], train_set['request_label'])
    joblib.dump(ppl, osp.join(save_dir, "pipeline.pkl"))

    # 6. evaluation model with valid_set and test_set, save into json
    valid_pred = ppl.predict(valid_set[['feature_1', 'feature_2', 'feature_3']])
    test_pred = ppl.predict(test_set[['feature_1', 'feature_2', 'feature_3']])
    valid_set['ppl_pred'] = valid_pred
    test_set['ppl_pred'] = test_pred
    valid_set.to_csv(osp.join(save_dir, 'valid_set.csv'), index=False)
    test_set.to_csv(osp.join(save_dir, 'test_set.csv'), index=False)

    valid_acc = metrics.accuracy_score(valid_set['request_label'], valid_set['ppl_pred'])
    test_acc = metrics.accuracy_score(test_set['request_label'], test_set['ppl_pred'])
    print(f"valid_acc: {valid_acc}, test_acc: {test_acc}")


if __name__ == '__main__':
    seed = 0
    save_dir = osp.join(EXP_HOME, 'test', 'lgbm', f'seed-{seed}', 'prepare')
    os.makedirs(save_dir, exist_ok=True)
    run(save_dir, seed=seed)
