# !/bin/bash
interpreter="python"
# interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

$interpreter run.py --stage prepare --task test/trips --model lgbm
$interpreter run.py --stage train --task test/trips --model lgbm
$interpreter run.py --stage offline --task test/trips --model lgbm --nreqs 20 --clear_cache
$interpreter run.py --stage online --task test/trips --model lgbm --nreqs 20 --offline_nreqs 20 --pest_constraint error --max_error 0.5 --min_conf 0.95
