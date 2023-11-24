import os

TASK_HOME = "test"

for nparts in [2, 5, 10, 20, 50, 100]:
    command = f"python run.py --example trips --stage ingest --task {TASK_HOME}/trips --nparts {nparts}"
    os.system(command=command)

nparts = 2
command = f"python run.py --example trips --stage prepare --task {TASK_HOME}/trips --nparts {nparts}"
os.system(command=command)

for model in ["xgb", "lgbm", "dt", "rf", "lr"]:
    command = f"python run.py --example trips --stage train --task {TASK_HOME}/trips --model {model} --nparts {nparts}"
    os.system(command=command)
