# Usage: bash run.sh

python prepare.py --max_request 2000 --task trips --skip_dataset
python trainer.py --task trips --model lgbm
python online.py --task trips --model lgbm --exact
python online.py --task trips --model lgbm  # default setting
python online.py --task trips --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 5
python online.py --task trips --model lgbm --max_relative_error 0.1 --min_conf 1.0 --ncfgs 5
python online.py --task trips --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 3
python online.py --task trips --model lgbm --max_relative_error 0.1 --min_conf 0.99 --ncfgs 10