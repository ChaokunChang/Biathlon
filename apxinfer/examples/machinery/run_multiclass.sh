# Usage: bash run_multiclass.sh

python prepare.py --max_request 0 --task machinery_multiclass --skip_dataset --multiclass
python trainer.py --task machinery_multiclass --model mlp --multiclass
python online.py --task machinery_multiclass --model mlp --exact --multiclass
python online.py --task machinery_multiclass --model mlp --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 3 --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 3 --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 10 --multiclass
python online.py --task machinery_multiclass --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 10 --multiclass