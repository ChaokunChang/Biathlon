import os

for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example tickv2 --stage ingest --task test/tickv2 --nparts {nparts}"
    # os.system(command=command)

nparts = 2
command = f"python run.py --example tickv2 --stage prepare --task test/tickv2 --nparts {nparts}"
os.system(command=command)

for model in ["lr", "xgb", "rf", "lgbm", "mlp", "lstm"]:
    command = f"python run.py --example tickv2 --stage train --task test/tickv2 --model {model} --nparts {nparts}"
    os.system(command=command)
