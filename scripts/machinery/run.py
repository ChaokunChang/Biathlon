import os

prepare_script_path = '/home/ckchang/ApproxInfer/scripts/machinery/prepare.py'
plotting_script_path = '/home/ckchang/ApproxInfer/scripts/machinery/binary_classification/plotting.py'
for model_name in ['mlp', 'knn', 'svm', 'xgb', 'dt', 'lgbm']:
    cmd = '/home/ckchang/anaconda3/envs/amd/bin/python ' \
        + prepare_script_path + ' ' \
        + f'--model_name {model_name}' + ' '
    print("executing ", cmd)
    os.system(cmd)

    exps = []
    exps += ['equal', 'fimp', 'finf', 'finf_fimp']
    for sample_strategy in exps:
        for thresh in [0.0, 1.0, 0.99]:
            cmd = '/home/ckchang/anaconda3/envs/amd/bin/python ' \
                + plotting_script_path + ' ' \
                + f'--model_name {model_name}' + ' ' \
                + f'--sample_strategy {sample_strategy}' + ' ' \
                + f'--npoints_for_conf 1000' + ' ' \
                + f'--low_conf_threshold {thresh}' + ' '
            print("executing ", cmd)
            os.system(cmd)

"""
for model_name in ['mlp', 'knn', 'svm', 'xgb', 'dt', 'lgbm']:
    for init_sample_strategy in ['equal', 'fimp']:
        for prediction_estimator in ['joint_distribution', 'feature_distribution']:
            for prediction_estimator_thresh in [0.0, 1.0, 0.9, 0.99]:
                for sample_increase_policy in ['one_step', 'ten_steps_equal', 'exponential_increase', 'auto']:
                    for sample_allocation_policy in ['equal', 'fimp', 'finf', 'auto']:
                        cmd = '/home/ckchang/anaconda3/envs/amd/bin/python ' \
                            + prepare_script_path + ' ' \
                            + f'--model_name {model_name}' + ' ' \
                            + f'--init_sample_strategy {init_sample_strategy}' + ' ' \
                            + f'--prediction_estimator {prediction_estimator}' + ' ' \
                            + f'--prediction_estimator_thresh {prediction_estimator_thresh}' + ' ' \
                            + f'--sample_increase_policy {sample_increase_policy}' + ' ' \
                            + f'--sample_allocation_policy {sample_allocation_policy}' + ' ' \
                            + f'--npoints_for_conf 1000' + ' ' 
                        print("executing ", cmd)
                        os.system(cmd)
"""
