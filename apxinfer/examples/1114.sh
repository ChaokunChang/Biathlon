qinf="sobol"
nparts=20
ncfgs=20
model="lgbm"
for max_error in 0.1 2.0
do
    python eval_trips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf
done

model="xgb"
for max_error in 0.1 0.5 2.0
do
    python eval_trips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf
done