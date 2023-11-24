import os

for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example tickv3 --stage ingest --task test/tickv3 --nparts {nparts}"
    # os.system(command=command)

nparts = 2
command = f"python run.py --example tickv3 --stage prepare --task test/tickv3 --nparts {nparts}"
os.system(command=command)

for model in ["lr", "xgb", "rf", "lgbm", "mlp", "knn", "dt", "svm", "ridge"]:
    command = f"python run.py --example tickv3 --stage train --task test/tickv3 --model {model} --nparts {nparts}"
    os.system(command=command)
