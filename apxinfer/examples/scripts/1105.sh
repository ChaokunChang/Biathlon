model="lgbm"
for nparts in 2 5 10 20 100
do
    python eval_ccfraud.py --model $model --nparts $nparts --ncores 0 --nocache
    python eval_ccfraud.py --model $model --nparts $nparts --ncores 1 --nocache
done