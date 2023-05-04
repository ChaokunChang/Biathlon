model="lgbm"
bash run.sh 7 $model None prep
bash run.sh 7 $model None feature
bash run.sh 7 $model None build
bash run.sh 7 $model None test
bash run.sh 7 $model 0.1 test
bash run.sh 7 $model auto:0.1 test
bash run.sh 7 $model auto:0.2 test
bash run.sh 7 $model None selection

bash run.sh 7 $model None feature 10
bash run.sh 7 $model None build 10
bash run.sh 7 $model None test 10
bash run.sh 7 $model 0.1 test 10
bash run.sh 7 $model auto:0.1 test 10
bash run.sh 7 $model auto:0.2 test 10
