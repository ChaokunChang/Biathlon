Note run ncores=1 first. 


old ones
```
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp machinery --model mlp --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model svm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinery --model knn --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tick-v2 --model lr --ncores 0 --loading_mode 0 --seed xxx
```

New ones:

```
python evaluate_all.py --exp machinerymulti --model mlp --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model svm --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model knn --ncores 1 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tripsfeast --model lgbm --ncores 1 --loading_mode 0 --seed xxx

python evaluate_all.py --exp machinerymulti --model mlp --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model svm --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp machinerymulti --model knn --ncores 0 --loading_mode 0 --seed xxx
python evaluate_all.py --exp tripsfeast --model lgbm --ncores 0 --loading_mode 0 --seed xxx
```