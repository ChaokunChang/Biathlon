import os

model = "lr"

for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example tick --stage ingest --task test/tick --model {model} --nparts {nparts}"
    # os.system(command=command)

nparts = 2
command = f"python run.py --example tick --stage prepare --task test/tick --model {model} --nparts {nparts}"
# os.system(command=command)

for model in ["lr", "xgb", "rf", "lgbm", "mlp", "lstm"]:
    command = f"python run.py --example tick --stage train --task test/tick --model {model} --nparts {nparts}"
    os.system(command=command)
