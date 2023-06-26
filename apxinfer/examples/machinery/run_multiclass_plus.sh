# Usage: bash run_multiclass_plus.sh

python prepare.py --max_request 0 --task machinery_multiclass_plus --skip_dataset --multiclass --plus
python trainer.py --task machinery_multiclass_plus --model mlp --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --exact --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 3 --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 3 --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 10 --multiclass --plus
python online.py --task machinery_multiclass_plus --model mlp --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 10 --multiclass --plus