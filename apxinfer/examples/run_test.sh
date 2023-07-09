#!/bin/bash

interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

task=$1
max_requests=100
if [ "$task" = "trips" ] || [ "$task" == "taxi" ]; then
    model="lgbm"
    error_cons="--pest_constraint error --max_error 1.0"
elif [ "$task" = "machinery" ]; then
    model="mlp"
    error_cons="--pest_constraint error --max_error 0.0"
elif [ "$task" == "ccfraud" ]; then
    max_request=1000
    model="xgb"
    error_cons="--pest_constraint error --max_error 0.0"
elif [ "$task" == "traffic" ] || [ "$task" == "tick" ]; then
    model="rf"
    error_cons="--pest_constraint relative_error --max_relative_error 0.1"
else
    model=""
    error_cons=""
fi

min_conf=0.99
ncfgs=10

echo running $task with model $model and $error_cons, $min_conf, $ncfgs

echo running prepare with $interpreter $task/prepare.py --task test/$task --max_requests $max_requests --skip_dataset
$interpreter $task/prepare.py --task test/$task --max_requests $max_requests --skip_dataset

echo running trainer with $interpreter $task/trainer.py --task test/$task --model $model
$interpreter $task/trainer.py --task test/$task --model $model

echo running offline with $interpreter $task/offline.py --task test/$task --model $model --nreqs 10 --clear_cache
$interpreter $task/offline.py --task test/$task --model $model --nreqs 10 --clear_cache

echo running online with $interpreter $task/online.py --task test/$task --model $model --exact
$interpreter $task/online.py --task test/$task --model $model --exact

echo running online with $interpreter $task/online.py --task test/$task --model $model --min_conf $min_conf --ncfgs $ncfgs $error_cons
$interpreter $task/online.py --task test/$task --model $model --min_conf $min_conf --ncfgs $ncfgs $error_cons
