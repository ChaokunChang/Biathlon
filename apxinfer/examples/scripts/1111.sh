for qinf in "sobol" "sobolT"
do
    for model in "dt" "knn"
    do
        for nparts in 2 5 10 20 100
        do
            ncfgs=$nparts
            python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0 --nocache --qinf $qinf
            python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0 --nocache --qinf $qinf
        done

        nparts=100
        for ncfgs in 2 5 10 20 50
        do
            python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0 --nocache --qinf $qinf
            python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0 --nocache --qinf $qinf
        done
    done
done