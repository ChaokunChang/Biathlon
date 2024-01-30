task_name="machinery"
agg_qids="0 1 2 3 4 5 6 7"
# model ncores max_error scheduler_init scheduler_batch
python prep.py --task_name $task_name --prepare_again

for model in "mlp" "dt" "knn"
do
    ncores=1 # only one core by default
    nparts=100 # ncfgs=nparts by default
    shared_opts="--task_name $task_name --agg_qids $agg_qids --model $model --nparts $nparts --ncores $ncores"
    python eval_reg.py $shared_opts --run_shared
    for scheduler_init in 1 5 10 20
    do
        for scheduler_batch in 1 5 10 20
        do
            python eval_reg.py $shared_opts --scheduler_init $scheduler_init --scheduler_batch $scheduler_batch --max_error 0.0
        done
    done
done