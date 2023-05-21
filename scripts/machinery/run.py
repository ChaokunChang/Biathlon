import os

class OnlineParser(Tap):
    database = "machinery_more"
    segment_size = 50000

    # path to the task directory
    task = "binary_classification"

    model_name: str = "xgb"  # model name
    model_type: Literal["regressor", "classifier"] = "classifier"  # model type
    multi_class: bool = False  # multi class classification

    num_agg_queries: int = 8  # number of aggregation queries

    max_sample_budget: float = 1.0  # max sample budget each in avg
    init_sample_budget: float = 0.01  # initial sample budget each in avg

    init_sample_policy: Literal[
        "uniform", "fimp", "finf", "auto"
    ] = "uniform"  # initial sample policy

    prediction_estimator: Literal[
        "joint_distribution", "feature_distribution", "auto"
    ] = "auto"  # prediction estimator
    prediction_estimator_thresh: float = 0.0  # prediction estimator threshold
    prediction_estimation_nsamples: int = (
        100  # number of points for prediction estimation
    )

    feature_influence_estimator: Literal["shap", "lime", "auto"] = "auto"
    feature_influence_estimator_thresh: float = 0.0
    feature_influence_estimation_nsamples: int = 1000

    # policy to increase sample to budget
    sample_budget: float = 0.1  # sample budget each in avg
    sample_refine_max_niters: int = 0  # nax number of iters to refine the sample budget
    sample_refine_step_policy: Literal[
        "uniform", "exponential"
    ] = "uniform"  # sample refine step policy
    sample_allocation_policy: Literal[
        "uniform", "fimp", "finf", "auto"
    ] = "uniform"  # sample allocation policy

    seed = 7077  # random seed
    clear_cache: bool = False  # clear cache

    def process_args(self) -> None:
        self.job_dir: str = os.path.join(
            RESULTS_HOME, self.database, f"{self.task}_{self.model_name}"
        )

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
