# ApproxInfer
Approximate Machine Learning Inference with Approximate Query Processing


0. install python>3.8 and pip install required packages
1. git clone this repo in home directory
2. setup clickhouse, install it
3. download three dataset using scp from numa:/public/ckchang/db/clickhouse/user_files/
    - machinery
    - taxi-2015
    - tick-data
4. put the dataset in the right path (required by ingestor) on your server. 
5. run the script run.sh in ApproxInfer/apxinfer/example/
    - make sure you are using the right python or python env


run experiments

1. run "evaluate_all.py --exp prepare" (around 1h)
2. run "evaluate_all.py --exp prepare" again. (will be fast, keep eye on it to see whether there is a bug)
3. run evaluate_all.py with different arguments, see below

Before Evluation
```
python evaluate_all.py --exp prepare --seed xxx
```

These are must
```
python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 0 --seed xxx
```


These are Optional 1
```
python evaluate_all.py --exp trips --model xgb --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp trips --model dt --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model dt --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model rf --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model dt --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model rf --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model lgbm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model dt --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp trips --model xgb --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp trips --model dt --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model dt --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v1 --model rf --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model dt --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model rf --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model lgbm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp cheaptrips --model dt --ncores 0 --loading_mode 0 --seed xxx

```


These are Optional 2
```
python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 1 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 1 --seed xxx
python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 2 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 2 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 2 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 5 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 5 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 5 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 10 --seed xxx

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 10 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 10 --seed xxx

```

These are Optional 3
```
python evaluate_all.py --exp trips --model lgbm --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp tick-v1 --model lr --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp cheaptrips --model xgb --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model dt --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10

python evaluate_all.py --exp trips --model lgbm --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp tick-v1 --model lr --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp cheaptrips --model xgb --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model dt --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 0 --seed xxx --nparts 10 --ncfgs 10
```