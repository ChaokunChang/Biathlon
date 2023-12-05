# ApproxInfer
Approximate Machine Learning Inference with Approximate Query Processing

## Setup dataset
0. install python>3.8 and pip install required packages
1. git clone this repo in home directory
2. setup clickhouse, install it
3. download three dataset using scp from numa:/public/ckchang/db/clickhouse/user_files/
    - machinery
    - taxi-2015
    - tick-data
    - talkingdata/adtracking-fraud
4. put the dataset in the right path (required by ingestor) on your server. 
5. run the script run.sh in ApproxInfer/apxinfer/example/
    - make sure you are using the right python or python env

## Run experiments
``` bash
# --interpreter should be set as your python env with ApproxInfer in PYTHONPATH
interpreter=python
# run prepare to setup database, prepare requests for training and test, as well as train the model
python evaluate_all.py --exp prepare --prep_single machinery --interpreter $interpreter
python evaluate_all.py --exp prepare --prep_single tdfraud --interpreter $interpreter
python evaluate_all.py --exp prepare --prep_single tripsfeast --interpreter $interpreter
python evaluate_all.py --exp prepare --prep_single tickvaryNM8 --interpreter $interpreter

# evaluate the pipeline with different settings.
seed=0
# warm up first with useless config
python run.py --example machinery --stage online --task final/machinery --nparts 100 --ncfgs 100 --model mlp --offline_nreqs 50  --ncores 1 --loading_mode 0 --scheduler_init 0 --scheduler_batch 100000 --max_error 0.0 --min_conf 1.0
python evaluate_all.py --exp machinery --model mlp --ncores 1 --loading_mode 0 --interpreter $interpreter --seed $seed

python run.py --example tdfraud --stage online --task final/tdfraud --nparts 100 --ncfgs 100 --model xgb --offline_nreqs 50  --ncores 1 --loading_mode 0 --scheduler_init 0 --scheduler_batch 100000 --max_error 0.0 --min_conf 1.0
python evaluate_all.py --exp tdfraud --model xgb --ncores 1 --loading_mode 0 --interpreter $interpreter --seed $seed

python run.py --example tripsfeast --stage online --task final/tripsfeast --nparts 100 --ncfgs 100 --model lgbm --offline_nreqs 50  --ncores 1 --loading_mode 0 --scheduler_init 0 --scheduler_batch 100000 --max_error 0.0 --min_conf 1.0
python evaluate_all.py --exp tripsfeast --model lgbm --ncores 1 --loading_mode 0 --interpreter $interpreter --seed $seed

python run.py --example tickvaryNM8 --stage online --task final/tickvaryNM8 --nparts 100 --ncfgs 100 --model lr --offline_nreqs 50  --ncores 1 --loading_mode 0 --scheduler_init 0 --scheduler_batch 100000 --max_error 0.0 --min_conf 1.0
python evaluate_all.py --exp tickpricemiddle --model lr --ncores 1 --loading_mode 0 --interpreter $interpreter --seed $seed

```