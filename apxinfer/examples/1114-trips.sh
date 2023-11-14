task_name="trips"
agg_qids="1 2 3"
# model ncores max_error scheduler_init scheduler_batch

for model in "lgbm" "xgb"
do
    ncores=1 # only one core by default
    nparts=100 # ncfgs=nparts by default
    shared_opts="--task_name $task_name --agg_qids $agg_qids --model $model --nparts $nparts --ncores $ncores"
    python eval_reg.py $shared_opts --run_shared
    for scheduler_init in 50 20 10 5 1
    do
        for scheduler_batch in 1 5 10 20 50
        do
            for max_error in 0.5 1.0 2.0 3.0
            do
                python eval_reg.py $shared_opts --scheduler_init $scheduler_init --scheduler_batch $scheduler_batch --max_error $max_error
            done
        done
    done
done
