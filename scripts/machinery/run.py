import os

prepare_script_path = '/home/ckchang/ApproxInfer/scripts/machinery/prepare.py'
plotting_script_path = '/home/ckchang/ApproxInfer/scripts/machinery/binary_classification/plotting.py'
for model_name in ['mlp', 'knn', 'svm', 'xgb', 'dt', 'lgbm']:
    cmd = '/home/ckchang/anaconda3/envs/amd/bin/python ' \
        + prepare_script_path + ' ' \
        + f'--model_name {model_name}' + ' '
    print("executing ", cmd)
    os.system(cmd)

    for sample_strategy in ['equal', 'online', 'fimp', 'online_fimp'] + ['online_0.9', 'online_0.9_fimp']:
        cmd = '/home/ckchang/anaconda3/envs/amd/bin/python ' \
            + plotting_script_path + ' ' \
            + f'--model_name {model_name}' + ' ' \
            + f'--sample_strategy {sample_strategy}' + ' ' \
            + f'--npoints_for_conf 1000' + ' '
        print("executing ", cmd)
        os.system(cmd)
