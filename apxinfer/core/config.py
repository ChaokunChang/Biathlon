import os
from typing import Literal
from tap import Tap

EXP_HOME = '/home/ckchang/.cache/apxinf/xip'


class BaseXIPArgs(Tap):
    task: str = 'test'  # task name
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation
    max_nchunks = 100  # maximum number of chunks to sample


class PrepareArgs(BaseXIPArgs):
    skip_dataset: bool = False  # whether to skip dataset preparation, by loading cached one
    max_requests: int = 2000  # maximum number of requests
    train_ratio: float = 0.5  # ratio of training data
    valid_ratio: float = 0.3  # ratio of validation data


class TrainerArgs(BaseXIPArgs):
    pass


class OnlineArgs(BaseXIPArgs):
    num_requests: int = 0  # number of test requests
    n_cfgs: int = 5  # number of query configurations

    disable_sample_cache: bool = False  # whether to disable cache the sample in loader
    disable_query_cache: bool = False  # whether to disable cache the query in loader

    pest_constraint: Literal['conf', 'error', 'relative_error'] = 'relative_error'  # prediction estimation constraint
    pest: Literal['MC'] = 'MC'  # prediction estimation method
    pest_nsamples: int = 1000  # number of samples for prediction estimation
    pest_seed: int = 0

    qinf: Literal['direct', 'by_finf'] = 'direct'  # query inference method

    scheduler: Literal['greedy', 'random'] = 'greedy'  # scheduler
    scheduler_batch: int = 1

    # pipeline settings
    termination_condition: Literal['conf', 'error', 'relative_error', 'min_max'] = 'conf'  # termination condition
    max_relative_error: float = 0.05  # maximum relative error
    max_error: float = 0.1  # maximum error
    min_conf: float = 0.99  # minimum confidence
    max_time: float = 60.0  # maximum time
    max_memory: float = 2048 * 1.0  # maximum memory
    max_rounds: int = 10  # maximum rounds

    exact: bool = False  # run exact version
    verbose_execution: bool = False  # whether to print execution details

    def process_args(self):
        assert self.termination_condition != self.pest_constraint


class DIRHelper:
    def get_working_dir(args: BaseXIPArgs) -> str:
        working_dir = os.path.join(EXP_HOME, args.task, f'seed-{args.seed}')
        os.makedirs(working_dir, exist_ok=True)
        return working_dir

    def get_prepare_dir(args: BaseXIPArgs) -> str:
        working_dir = DIRHelper.get_working_dir(args)
        prepare_dir = os.path.join(working_dir, 'prepare')
        os.makedirs(prepare_dir, exist_ok=True)
        return prepare_dir

    def get_dataset_dir(args: BaseXIPArgs) -> str:
        prepare_dir = DIRHelper.get_prepare_dir(args)
        dataset_dir = os.path.join(prepare_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def get_model_dir(args: BaseXIPArgs) -> str:
        prepare_dir = DIRHelper.get_prepare_dir(args)
        model_dir = os.path.join(prepare_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def get_model_path(args: BaseXIPArgs) -> str:
        model_dir = DIRHelper.get_model_dir(args)
        return os.path.join(model_dir, f'{args.model}.pkl')

    def get_online_dir(args: OnlineArgs) -> str:
        working_dir = DIRHelper.get_working_dir(args)
        online_dir = os.path.join(working_dir, 'online', args.model)
        if args.exact:
            online_dir = os.path.join(online_dir, 'exact')
        else:
            online_dir = os.path.join(online_dir, f'ncfgs-{args.n_cfgs}')
            online_dir = os.path.join(online_dir, f'pest-{args.pest_constraint}-{args.pest}-{args.pest_nsamples}-{args.pest_seed}')
            online_dir = os.path.join(online_dir, f'qinf-{args.qinf}')
            online_dir = os.path.join(online_dir, f'scheduler-{args.scheduler}-{args.scheduler_batch}')
        os.makedirs(online_dir, exist_ok=True)
        return online_dir
