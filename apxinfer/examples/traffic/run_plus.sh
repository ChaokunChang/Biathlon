# Usage: bash run_plus.sh

python prepare.py --max_request 10000 --task traffic_plus --skip_dataset --plus
python trainer.py --task traffic_plus --model lr --plus
python online.py --task traffic_plus --model lr --exact --plus
python online.py --task traffic_plus --model lr --plus
python online.py --task traffic_plus --model lr --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 5 --plus
python online.py --task traffic_plus --model lr --max_relative_error 0.1 --min_conf 1.0 --n_cfgs 5 --plus
python online.py --task traffic_plus --model lr --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 3 --plus
python online.py --task traffic_plus --model lr --max_relative_error 0.1 --min_conf 0.99 --n_cfgs 10 --plus