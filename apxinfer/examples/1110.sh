for model in "lr" "rf" "xgb" "lgbm"
do
    nparts=100
    qinf="sobolT"
    for ncfgs in 2 5 10 20 50
    do
        python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0.1 --nocache --qinf $qinf
        python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0.1 --nocache --qinf $qinf
        for max_error in 0.05 0.02 0.01 0.001
        do
            python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf
            python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf
        done
    done
done
