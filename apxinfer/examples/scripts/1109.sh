for model in "lr" "rf" "xgb" "lgbm"
do
    nparts=100
    for ncfgs in 2 5 10 20 50
    do
        python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0.1 --nocache
        python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0.1 --nocache
        for max_error in 0.05 0.02 0.01 0.001
        do
            python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error
            python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error
        done
    done
done
