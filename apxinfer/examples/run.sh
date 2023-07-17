# !/bin/bash
interpreter="python"
interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

$interpreter run.py --stage prepare --task test/trips2 --model lgbm --max_requests 200
$interpreter run.py --stage train --task test/trips2 --model lgbm --max_requests 200
$interpreter run.py --stage offline --task test/trips2 --model lgbm --max_requests 200 --nreqs 20 --clear_cache
$interpreter run.py --stage online --task test/trips2 --model lgbm --max_requests 200 --offline_nreqs 20 --pest_constraint error --max_error 0.5 --min_conf 0.9
