# Usage: bash run_plus.sh

python prepare.py --max_request 0 --task machinery_plus --skip_dataset --plus
python trainer.py --task machinery_plus --model mlp --plus
python online.py --task machinery_plus --model mlp --exact --plus
python online.py --task machinery_plus --model mlp --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 3 --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 3 --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 10 --plus
python online.py --task machinery_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 10 --plus