# Usage: bash run_plus.sh

python prepare.py --max_request 2000 --task trips_plus --skip_dataset --plus
python trainer.py --task trips_plus --model lgbm --plus
python online.py --task trips_plus --model lgbm --exact --plus
python online.py --task trips_plus --model lgbm --plus
python online.py --task trips_plus --model lgbm --max_relative_error 0.1 --min_conf 0.99 --plus
python online.py --task trips_plus --model lgbm --max_relative_error 0.1 --min_conf 1.0 --plus
python online.py --task trips_plus --model lgbm --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 3 --plus