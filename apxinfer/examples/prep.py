import os
from tap import Tap


class EvalArgs(Tap):
    task_name: str = None
    prepare_again: bool = False

    def process_args(self):
        assert self.task_name is not None


TASK_HOME = "final"
args = EvalArgs().parse_args()

TASK_NAME = args.task_name

for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example {TASK_NAME} --stage ingest --task {TASK_HOME}/{TASK_NAME} --nparts {nparts}"
    os.system(command=command)

nparts = 2
command = f"python run.py --example {TASK_NAME} --stage prepare --task {TASK_HOME}/{TASK_NAME} --nparts {nparts}"
if args.prepare_again:
    os.system(command=command)

models = ["xgb", "lgbm", "dt", "rf", "lr", "knn", "svm", "mlp"]
for model in models:
    command = f"python run.py --example {TASK_NAME} --stage train --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts}"
    os.system(command=command)
