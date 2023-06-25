# Usage: bash run_plus.sh

python prepare.py --max_request 2000 --task cc_fraud_plus --skip_dataset --plus
python trainer.py --task cc_fraud_plus --model xgb --plus
python online.py --task cc_fraud_plus --model xgb --exact --plus
python online.py --task cc_fraud_plus --model xgb --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 0.99 --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 1.0 --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 3 --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 3 --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 0.99 --n_cfgs 10 --plus
python online.py --task cc_fraud_plus --model xgb --pest_constraint error --max_error 0 --min_conf 1.0 --n_cfgs 10 --plus