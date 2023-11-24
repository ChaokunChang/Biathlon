model="rf"
for nparts in 2 5 10 20 100
do
    python eval_tick.py --model $model --nparts $nparts --ncores 0 --max_error 0.1 --nocache
    python eval_tick.py --model $model --nparts $nparts --ncores 1 --max_error 0.1 --nocache
    for max_error in 0.05 0.02 0.01 0.001
    do
        python eval_tick.py --model $model --nparts $nparts --ncores 0 --max_error $max_error
        python eval_tick.py --model $model --nparts $nparts --ncores 1 --max_error $max_error
    done
done
