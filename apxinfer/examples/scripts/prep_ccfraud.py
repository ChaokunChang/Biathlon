import os


for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example ccfraud --stage ingest --task test/ccfraud --nparts {nparts}"
    os.system(command=command)

nparts = 2
command = f"python run.py --example ccfraud --stage prepare --task test/ccfraud --nparts {nparts}"
os.system(command=command)

for model in ["lr", "xgb", "rf", "lgbm", "mlp"]:
    command = f"python run.py --example ccfraud --stage train --task test/ccfraud --model {model} --nparts {nparts}"
    os.system(command=command)
