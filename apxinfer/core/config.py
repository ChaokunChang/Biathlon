import os
from tap import Tap

EXP_HOME = '/home/ckchang/.cache/apxinf/xip'


class PrepareArgs(Tap):
    task: str = 'test'  # task name
    model: str = 'lgbm'  # model name
    seed: int = 0  # seed for prediction estimation

    max_nchunks = 100

    skip_dataset: bool = False  # whether to skip dataset preparation, by loading cached one
    max_requests: int = 2000  # maximum number of requests
    train_ratio: float = 0.5  # ratio of training data
    valid_ratio: float = 0.3  # ratio of validation data


def get_prepare_dir(args: PrepareArgs) -> str:
    working_dir = os.path.join(EXP_HOME, args.task, args.model, f'seed-{args.seed}', 'prepare')
    os.makedirs(working_dir, exist_ok=True)
    return working_dir
