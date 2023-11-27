Note run ncores=1 first. 


old ones
```
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --seed xxx --skip_shared
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 0 --seed xxx --skip_shared
python evaluate_all.py --exp machinery --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 0 --seed xxx --skip_shared

python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 0 --seed xxx --skip_shared
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 0 --seed xxx --skip_shared
python evaluate_all.py --exp machinery --model svm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 0 --seed xxx --skip_shared
```

New ones:

```
python evaluate_all.py --exp prepare --prep_single tripsfeast --seed xxx
python evaluate_all.py --exp prepare --prep_single machinerymulti --seed xxx

python evaluate_all.py --exp tripsfeast --model lgbm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model knn --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp tripsfeast --model lgbm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model mlp --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model svm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model knn --ncores 0 --loading_mode 0 --seed xxx
```


For Varying number of aggregation features without changing total number of features (These will be fast):
Remember to add seed to evaluate_all.py
``` bash
python evaluate_all.py --exp prepare --prep_single machineryxf8
python evaluate_all.py --exp machineryxf8 --model mlp --ncores 1 --loading_mode 0
for nf in {1..7}; do cp -r /home/ckchang/.cache/apxinf/xip/final/machineryxf8 /home/ckchang/.cache/apxinf/xip/final/machineryxf$nf; done
for nf in {1..7}; do rm -rf /home/ckchang/.cache/apxinf/xip/final/machineryxf$nf/seed-0/online/mlp/ncores-1/ldnthreads-0/nparts-100/ncfgs-100; done
for nf in {1..7}; do python evaluate_all.py --exp machineryxf$nf --model mlp --ncores 1 --loading_mode 0 --skip_shared; done
```


For Varing the size of data (These will be fast):
Remember to add seed to evaluate_all.py
``` bash
python evaluate_all.py --exp prepare --prep_single tickvaryNM1
python evaluate_all.py --exp tickvaryNM1 --model lr --ncores 1 --loading_mode 0
python evaluate_all.py --exp tickprice --model lr --ncores 1 --loading_mode 0 --skip_shared
```

``` bash 
# scp -r numa:/public/ckchang/db/clickhouse/user_files/tick-data server:/xxx/user_files/
```

``` bash
for nm in 2 8 16 29
do    
    cp -r /home/ckchang/.cache/apxinf/xip/final/tickvaryNM1 /home/ckchang/.cache/apxinf/xip/final/tickvaryNM$nm
    rm -rf /home/ckchang/.cache/apxinf/xip/final/tickvaryNM$nm/seed-0/online/lr/ncores-1/ldnthreads-0/nparts-100/ncfgs-100
done
for nm in 2 8 16 29; do python run.py --example tickvaryNM$nm --stage ingest --task final/tickvaryNM$nm --nparts 100; done
for nm in 2 8 16 29; do python evaluate_all.py --exp tickvaryNM$nm --model lr --ncores 1 --loading_mode 0 --skip_shared; done

# vary size of window for tripsfeast
for nw in 2 4 8 24 48; do python evaluate_all.py --exp tripsfeastw$nw --model lgbm --ncores 1 --loading_mode 0; done
```

# new must piplines
```bash
scp -r numa:/public/ckchang/db/clickhouse/user_files/talkingdata /var/lib/clickhouse/user_files/
# use three seeds
# sudo /home/ckchang/anaconda3/envs/apx/bin/python run.py --example tdfraud --stage ingest --task final/tdfraud --nparts 100 --seed xxx
# scp -r numa:/home/ckchang/.cache/apxinf/xip/final/tdfraud /home/ckchang/.cache/apxinf/xip/final/tdfraud
# mv /home/ckchang/.cache/apxinf/xip/final/tdfraud/seed-0 /home/ckchang/.cache/apxinf/xip/final/tdfraud/seed-xxx
python evaluate_all.py --exp prepare --prep_single tdfraud --seed xxx # too slow
python evaluate_all.py --exp tdfraud --model xgb --ncores 1 --loading_mode 0 --seed xxx

# use two seeds
# sudo /home/ckchang/anaconda3/envs/apx/bin/python run.py --example tdfraudrandom --stage ingest --task final/tdfraudrandom --nparts 100 --seed xxx
# scp -r numa:/home/ckchang/.cache/apxinf/xip/final/tdfraudrandom /home/ckchang/.cache/apxinf/xip/final/tdfraudrandom
# mv /home/ckchang/.cache/apxinf/xip/final/tdfraudrandom/seed-0 /home/ckchang/.cache/apxinf/xip/final/tdfraudrandom/seed-xxx
python evaluate_all.py --exp prepare --prep_single tdfraudrandom --seed xxx # too slow
python evaluate_all.py --exp tdfraudrandom --model xgb --ncores 1 --loading_mode 0 --seed xxx

```

# complementary experiments
```bash
# make sure that the experiment has been run on the server before, otherwise will be slow
# and I recommend to use 3 or 2 seeds to reduce the time for waiting.
python evaluate_all.py --exp tripsfeast --model lgbm --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
python evaluate_all.py --exp tdfraud --model xgb --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
python evaluate_all.py --exp tickvaryNM8 --model lr --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
python evaluate_all.py --exp machinerymulti --model svm --ncores 1 --loading_mode 0 --skip_shared --seed xxx 
```