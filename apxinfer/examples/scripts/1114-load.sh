qinf="sobol"
model="lr"
# for nparts in 2 5 10 20 50 100
# do
#     ncfgs=$nparts
#     for max_error in 0.001 0.01 0.02 0.05 0.1
#     do
#         python eval_tick.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
#         python eval_tick.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
#         python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
#         python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
#     done
# done

# model="lgbm"
# for nparts in 2 5 10 20 50 100
# do
#     ncfgs=$nparts
#     for max_error in 0.5 1.0 2.0
#     do
#         python eval_trips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
#         python eval_trips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
#     done
# done

model="xgb"
max_error=0.0
for nparts in 2 5 10 20 50 100
do
    ncfgs=$nparts
    python eval_cheaptrips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
    python eval_cheaptrips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
done

for model in "mlp" "dt" "knn"
do
    for nparts in 2 5 10 20 50 100
    do
        ncfgs=$nparts
        python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
        python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
    done

    nparts=100
    for ncfgs in 2 5 10 20 50
    do
        python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --load_only
        python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error $max_error --qinf $qinf --load_only
    done
done
