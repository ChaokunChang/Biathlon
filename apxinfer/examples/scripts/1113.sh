# Missing Experiments for Trips

qinf="sobol"
nparts=50
ncfgs=50
max_error=1.0
python run.py --example trips --stage ingest --task test/trips --nparts $nparts
for model in "lgbm" "xgb"
do
    python eval_trips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --nocache
done

model="lr"
max_error=0.01
python run.py --example tick --stage ingest --task test/tick --nparts $nparts
python eval_tick.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --nocache
python eval_tickv2.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --nocache

max_error=0.0
python run.py --example cheaptrips --stage ingest --task test/trips --nparts $nparts
for model in "lgbm" "xgb"
do
    python eval_cheaptrips.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error $max_error --qinf $qinf --nocache
done
