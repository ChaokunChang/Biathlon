# Usage: bash run_test.sh

python prepare.py --max_request 100 --task trips_test --skip_dataset
python trainer.py --task trips_test --model lgbm
python online.py --task trips_test --model lgbm --exact --num_requests 1 --verbose_execution
python online.py --task trips_test --model lgbm  # default setting
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 5
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 1.0 --n_cfgs 5
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 3
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 10