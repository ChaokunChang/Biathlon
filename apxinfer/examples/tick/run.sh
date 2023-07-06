#!/bin/bash
# python prepare.py --max_request 10000 --task tick --skip_dataset
# python trainer.py --task tick --model rf
# python online.py --task tick --model rf --exact
sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python online.py --task tick --model rf
for error in 0.2 0.1 0.05 0.01 0.0; do
    for min_conf in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99 0.999 1.0; do
        for ncfgs in 2 3 5 10 50 100; do
            sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python online.py --task tick --model rf --max_relative_error $error --min_conf $min_conf --ncfgs $ncfgs
        done
    done
done