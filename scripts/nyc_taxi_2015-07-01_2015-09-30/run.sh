# make sure we have at least three arguments cfgid, model, and sample
if [ $# -lt 4 ]; then
    echo "Usage: $0 cfgid model sample action"
    echo "cfgid: 1-9"
    echo "model: lgbm, mlp, xgb, rf, etc"
    echo "sample: 0.001, 0.01, 0.1, 0.5, etc"
    echo "action: prep, feature, build, test"
    exit 1
fi

cfgid=$1
model=$2
sample=$3
action=$4

# if we have the forth argument, it will be fcols option
if [ $# -eq 5 ]; then
    fcols="--fcols $5"
else
    fcols=""
fi

cur_dir=$(pwd)
scripts_dir=$(dirname $cur_dir)
home_dir=$(dirname $scripts_dir)
apxinfer_dir=$home_dir/apxinfer

data_name="nyc_taxi_2015-07-01_2015-09-30"
data_dir=$home_dir/data/$data_name

# classification tasks
cfg1="--data $data_name --task day_of_week --keycol day_of_year --sort_by day_of_year --target day_of_week --sql_templates_file day_of_week/templates.sql --model_type classifier --model_name $model --multi_class"
cfg2="--data $data_name --task is_weekend --keycol day_of_year --sort_by day_of_year --target is_weekend --sql_templates_file is_weekend/templates.sql --model_type classifier --model_name $model"
cfg3="--data $data_name --task hour_of_day --keycol hourstamp --sort_by hourstamp --target hour_of_day --sql_templates_file hour_of_day/templates.sql --model_type classifier --model_name $model --multi_class"
cfg4="--data $data_name --task is_night --keycol hourstamp --sort_by hourstamp --target is_night --sql_templates_file is_night/templates.sql --model_type classifier --model_name $model"

# regression tasks
cfg5="--data $data_name --task fare_prediction_2015-08-01_2015-08-15_10000 --keycol trip_id --sort_by pickup_datetime --target fare_amount --sql_templates_file fare_prediction_2015-08-01_2015-08-15_10000/templates.sql --model_type regressor --model_name $model"
cfg6="--data $data_name --task duration_prediction_2015-08-01_2015-08-15_10000 --keycol trip_id --sort_by pickup_datetime --target trip_duration --sql_templates_file duration_prediction_2015-08-01_2015-08-15_10000/templates.sql --model_type regressor --model_name $model"
cfg7="--data $data_name --task fare_prediction_2015-08-01_2015-08-15_100 --keycol trip_id --sort_by pickup_datetime --target fare_amount --sql_templates_file fare_prediction_2015-08-01_2015-08-15_100/templates.sql --model_type regressor --model_name $model"
cfg8="--data $data_name --task trips_num_forecasting_1h --keycol hourstamp --sort_by hourstamp --target trips_num --sql_templates_file trips_num_forecasting_1h/templates.sql --model_type regressor --model_name $model"
cfg9="--data $data_name --task income_forecasting_1h --keycol hourstamp --sort_by hourstamp --target income --sql_templates_file income_forecasting_1h/templates.sql --model_type regressor --model_name $model"

cfgs="$cfg1|$cfg2|$cfg3|$cfg4|$cfg5|$cfg6|$cfg7|$cfg8|$cfg9"

cfg=$(echo $cfgs | cut -d'|' -f$cfgid)

if [ $action == "prep" ]; then
    # prepare stage
    if [ $cfgid == 7 ]; then
        cfg="$cfg --prediction_sample 100"
    fi
    python $apxinfer_dir/prepares.py $cfg
    exit 0
elif [ $action == "feature" ]; then
    # feature extraction stage
    python $apxinfer_dir/fextractor.py $cfg $fcols --sample $sample
    exit 0
elif [ $action == "build" ]; then
    # build pipeline stage
    python $apxinfer_dir/pipeline.py $cfg $fcols --sample $sample --apx_training
    exit 0
elif [ $action == "test" ]; then
    # test pipline stage
    python $apxinfer_dir/test_pipeline.py $cfg $fcols --sample $sample
    exit 0
else
    echo "action should be one of prep, feature, build, test"
    exit 1
fi
