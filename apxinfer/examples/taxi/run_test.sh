# Usage: bash run_test.sh

python prepare.py --max_request 100 --task trips_test --skip_dataset
python trainer.py --task trips_test --model lgbm
python online.py --task trips_test --model lgbm --exact --nreqs 1 --verbose_execution
python online.py --task trips_test --model lgbm  # default setting
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 5
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 1.0 --ncfgs 5
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 3
python online.py --task trips_test --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 10