#!/bin/bash

interpreter='python'
# interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

task=$1
if [ $# -gt 1 ]; then
    nthreads=$2
    echo nthreads=$nthreads
else
    nthreads=1
fi

task_home='main'
max_requests=2000
min_conf=0.99
nparts=100
ncfgs=10

if [ "$task" = "trips" ] || [ "$task" == "taxi" ]; then
    model="lgbm"
    error_cons="--pest_constraint error --max_error 1.0"
elif [ "$task" = "machinery" ]; then
    model="mlp"
    error_cons="--pest_constraint error --max_error 0.0"
elif [ "$task" == "ccfraud" ]; then
    model="xgb"
    error_cons="--pest_constraint error --max_error 0.0"
elif [ "$task" == "traffic" ] || [ "$task" == "tick" ]; then
    model="rf"
    error_cons="--pest_constraint relative_error --max_relative_error 0.1"
else
    model=""
    error_cons=""
fi


echo running $task with model $model and $error_cons, $min_conf, $ncfgs

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads $nthreads

echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads $nthreads

echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 10 --clear_cache
$interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 10 --clear_cache --loading_nthreads $nthreads

echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --exact
$interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --exact --loading_nthreads $nthreads

echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --min_conf $min_conf --ncfgs $ncfgs $error_cons
$interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --min_conf $min_conf --ncfgs $ncfgs $error_cons --loading_nthreads $nthreads
