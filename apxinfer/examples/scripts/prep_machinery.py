import os

for nparts in [2, 5, 10, 20, 100]:
    command = f"python run.py --example machinery --stage ingest --task test/machinery"
    os.system(command=command)

nparts = 2
command = f"python run.py --example machinery --stage prepare --task test/machinery --nparts {nparts}"
os.system(command=command)

for model in ['mlp', 'dt', 'knn', 'xgb', 'lgbm']:
    command = f"python run.py --example machinery --stage train --task test/machinery --model mlp --nparts {nparts}"
    os.system(command=command)
