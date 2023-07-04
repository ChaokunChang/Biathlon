# Usage: bash run.sh

python prepare.py --max_request 0 --task machinery --skip_dataset
python trainer.py --task machinery --model mlp
python online.py --task machinery --model mlp --exact
python online.py --task machinery --model mlp  # default setting
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 0.99
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 1.0
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --ncfgs 3
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --ncfgs 3
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --ncfgs 10
python online.py --task machinery --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --ncfgs 10