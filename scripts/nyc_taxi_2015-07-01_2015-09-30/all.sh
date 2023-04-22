
# iterate i from 1 to 6
for i in $(seq 1 6); do
    # iterate model from lbgm, xgb, mlp, dt, rf, lr
    for model in lgbm xgb mlp dt rf lr; do
        # run all.sh with i and model as args
        bash tmp.sh $i $model
    done
done