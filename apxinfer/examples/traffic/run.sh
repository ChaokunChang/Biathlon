# Usage: bash run.sh

python prepare.py --max_request 10000 --task traffic --skip_dataset
python trainer.py --task traffic --model lr
python online.py --task traffic --model lr --exact
python online.py --task traffic --model lr
python online.py --task traffic --model lr --max_relative_error 0.1 --min_conf 0.99 --ncfgs 5
python online.py --task traffic --model lr --max_relative_error 0.1 --min_conf 1.0 --ncfgs 5
python online.py --task traffic --model lr --max_relative_error 0.1 --min_conf 0.99 --ncfgs 3
python online.py --task traffic --model lr --max_relative_error 0.1 --min_conf 0.99 --ncfgs 10

