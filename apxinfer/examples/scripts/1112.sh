# Missing Experiment for Machinery

# rerun (previously failed) experiments
# the failure is that ncores=1 is better than ncores=0
qinf="sobol"
model="dt"
for nparts in 2 5 10 20
do
    ncfgs=$nparts
    python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0 --nocache --qinf $qinf
    python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0 --nocache --qinf $qinf
done

# complement experiements for varying ncfgs
model="mlp"
nparts=100
for ncfgs in 2 5 10 20 50
do
    python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 0 --max_error 0 --nocache --qinf $qinf
    python eval_machinery.py --model $model --nparts $nparts --ncfgs $ncfgs --ncores 1 --max_error 0 --nocache --qinf $qinf
done
