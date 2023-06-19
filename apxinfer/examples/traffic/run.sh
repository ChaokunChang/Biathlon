python prepare.py --task traffic
python prepare.py --task traffic --model mlp
python prepare.py --task traffic --model xgb
python prepare.py --task traffic --model dt
python prepare.py --task traffic --model lr

python prepare.py --task traffic_plus --max_request 10000
python prepare.py --task traffic_plus --model mlp --max_request 10000
python prepare.py --task traffic_plus --model xgb --max_request 10000
python prepare.py --task traffic_plus --model dt --max_request 10000
python prepare.py --task traffic_plus --model lr --max_request 10000
