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

New ones (After above are finished) (These will be fast):

```
python evaluate_all.py --exp prepare --prep_single machineryf1 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf2 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf3 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf4 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf5 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf6 --seed xxx
python evaluate_all.py --exp prepare --prep_single machineryf7 --seed xxx

python evaluate_all.py --exp machineryf1 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf2 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf3 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf4 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf5 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf6 --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf7 --model mlp --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp machineryf1 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf2 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf3 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf4 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf5 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf6 --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf7 --model knn --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp machineryf1 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf2 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf3 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf4 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf5 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf6 --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machineryf7 --model svm --ncores 1 --loading_mode 0 --seed xxx


```