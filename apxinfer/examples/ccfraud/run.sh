# Usage: bash run.sh

python prepare.py --max_request 0 --task cc_fraud --skip_dataset
python trainer.py --task cc_fraud --model xgb
python online.py --task cc_fraud --model xgb --exact
python online.py --task cc_fraud --model xgb  # default setting
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 0.99
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 1.0
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 3
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 3
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 10
python online.py --task cc_fraud --model xgb --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 10