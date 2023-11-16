import os
from tap import Tap


class EvalArgs(Tap):
    interpreter: str = "python"
    task_home: str = "final"
    task_name: str = None
    prepare_again: bool = False
    all_nparts: list[int] = [2, 5, 10, 20, 100]
    all_models: list[str] = ["xgb", "lgbm", "dt", "rf", "lr", "knn", "svm", "mlp"]

    def process_args(self):
        assert self.task_name is not None


args = EvalArgs().parse_args()

interpreter = args.interpreter
if interpreter != "python":
    interpreter = f"sudo {interpreter}"
TASK_HOME = args.task_home
TASK_NAME = args.task_name

for nparts in args.all_nparts:
    command = f"{interpreter} run.py --example {TASK_NAME} --stage ingest --task {TASK_HOME}/{TASK_NAME} --nparts {nparts}"
    os.system(command=command)

nparts = min(args.all_nparts)
command = f"{interpreter} run.py --example {TASK_NAME} --stage prepare --task {TASK_HOME}/{TASK_NAME} --nparts {nparts}"
if args.prepare_again:
    os.system(command=command)

for model in args.all_models:
    command = f"{interpreter} run.py --example {TASK_NAME} --stage train --task {TASK_HOME}/{TASK_NAME} --model {model} --nparts {nparts}"
    os.system(command=command)
