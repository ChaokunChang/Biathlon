# !/bin/bash
interpreter="python"
# interpreter="sudo PYTHONPATH=/home/ckchang/ApproxInfer /home/ckchang/anaconda3/envs/apxinf/bin/python"

task_home="test/trips"
nparts=2
oflnreqs=20
nreqs=200
ncfgs=2

# python run.py --stage prepare --task test/trips --model lgbm --nparts 2
# python run.py --stage train --task test/trips --model lgbm --nparts 2
# python run.py --stage offline --task test/trips --model lgbm --nparts 2 --nreqs 20 --ncfgs 2 --clear_cache --ncores 0
# python run.py --stage online --task test/trips --model lgbm --nparts 2 --offline_nreqs 20 --nreqs 200 --ncfgs 2 --exact --ncores 0
# python run.py --stage online --task test/trips --model lgbm --nparts 2 --offline_nreqs 20 --nreqs 200 --ncfgs 2 --ncores 0 --pest_constraint error --max_error 0.5 --min_conf 0.95

$interpreter run.py --stage prepare --task $task_home --model lgbm --nparts $nparts
$interpreter run.py --stage train --task $task_home --model lgbm --nparts $nparts
$interpreter run.py --stage offline --task $task_home --model lgbm --nparts $nparts --nreqs $oflnreqs --ncfgs $ncfg --clear_cache --ncores 0
$interpreter run.py --stage online --task $task_home --model lgbm --nparts $nparts --offline_nreqs $oflnreqs --nreqs $nreqs --ncfgs $ncfgs --exact --ncores 0
$interpreter run.py --stage online --task $task_home --model lgbm --nparts $nparts --offline_nreqs $oflnreqs --nreqs $nreqs --ncfgs $ncfgs --ncores 0 --pest_constraint error --max_error 0.5 --min_conf 0.95
