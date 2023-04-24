# make sure we have at least three arguments cfgid, model, and sample
if [ $# -lt 3 ]; then
    echo "Usage: $0 cfgid model sample"
    echo "cfgid: 1-6"
    echo "model: lgbm, mlp, xgb, rf, etc"
    echo "sample: 0.001, 0.01, 0.1, 0.5, etc"
    exit 1
fi

cfgid=$1
model=$2
sample=$3

# if we have the forth argument, it will be fcols option
if [ $# -eq 4 ]; then
    fcols="--fcols $4"
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

cfgs="$cfg1|$cfg2|$cfg3|$cfg4|$cfg5|$cfg6"

cfg=$(echo $cfgs | cut -d'|' -f$cfgid)
echo $cfg

if [ $sample == -1 ]; then
    # prepare stage
    python $apxinfer_dir/prepares.py $cfg
    exit 0
fi

cfg="$cfg $fcols"
if [ $sample == 0 ]; then
    # feature extraction stage
    python $apxinfer_dir/fextractor.py $cfg
    python $apxinfer_dir/pipeline.py $cfg
    python $apxinfer_dir/test_pipeline.py $cfg
    exit 0
else
    # if $sample starts with auto
    if [[ $sample == auto* ]]; then
        python $apxinfer_dir/test_auto_sampling.py $cfg --sample $sample
    else
        python $apxinfer_dir/fextractor.py $cfg --sample $sample
        python $apxinfer_dir/pipeline.py $cfg
        python $apxinfer_dir/pipeline.py $cfg --apx_training --sample $sample
        python $apxinfer_dir/test_pipeline.py $cfg --sample $sample
    fi
fi
