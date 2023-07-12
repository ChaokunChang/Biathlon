#!/bin/bash
interpreter="python"
# interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

task_home='ablation'

# run for machinery
task="machinery"
nparts=100
max_requests=2000
model="mlp"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    error_cons="--pest_constraint error --max_error 0"
    for ncfgs in 2 5 10 20 50 100; do
        for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
            $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
        done
    done
done

# run for ccfraud
task="ccfraud"
nparts=100
max_requests=2000
model="xgb"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    error_cons="--pest_constraint error --max_error 0"
    for ncfgs in 2 5 10 20 50 100; do
        for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
            $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
        done
    done
done


# run for taxi
task="taxi"
nparts=100
max_requests=2000
model="lgbm"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    for ncfgs in 2 5 10 20 50 100; do
        for max_error in 5 2 1 0.5 0.1 0.0; do
            error_cons="--pest_constraint error --max_error $max_error"
            for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
                $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
            done
        done
    done
done

# run for tick
task="tick"
nparts=100
max_requests=2000
model="rf"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    for ncfgs in 2 5 10 20 50 100; do
        for max_error in 0.5 0.2 0.1 0.05 0.01 0.0; do
            error_cons="--pest_constraint relative_error --max_relative_error $max_error"
            for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
                $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
            done
        done
    done
done

# run for tick1000
task="tick1000"
nparts=1000
max_requests=2000
model="rf"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    for ncfgs in 2 5 10 20 50 100; do
        for max_error in 0.5 0.2 0.1 0.05 0.01 0.0; do
            error_cons="--pest_constraint relative_error --max_relative_error $max_error"
            for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
                $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
            done
        done
    done
done

# run for traffic
task="traffic"
nparts=100
max_requests=2000
model="rf"

echo running prepare with $interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
$interpreter $task/prepare.py --task $task_home/$task --nparts $nparts --max_requests $max_requests --skip_dataset --loading_nthreads 8
echo running trainer with $interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8
$interpreter $task/trainer.py --task $task_home/$task --nparts $nparts --model $model --loading_nthreads 8

for ncfgs in 10 50 100; do
    echo running offline with $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8
    $interpreter $task/offline.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model --nreqs 20 --clear_cache --loading_nthreads 8

for ldnthreads in 8 4 2 1; do
    echo running online with $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads
    $interpreter $task/online.py --task $task_home/$task --nparts $nparts --model $model --exact --loading_nthreads $ldnthreads

    for ncfgs in 2 5 10 20 50 100; do
        for max_error in 0.5 0.2 0.1 0.05 0.01 0.0; do
            error_cons="--pest_constraint relative_error --max_relative_error $max_error"
            for min_conf in 0.0 0.1 0.3 0.5 0.7 0.9 0.95 0.99 0.999 1.0; do
                $interpreter $task/online.py --task $task_home/$task --nparts $nparts --ncfgs $ncfgs --model $model $error_cons --ncfgs $ncfgs --min_conf $min_conf --loading_nthreads $ldnthreads
            done
        done
    done
done