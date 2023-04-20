cfgid=$1
# if we have two args, then set model as the second arg, otherwise use default
if [ $# -eq 2 ]; then
    model=$2
else
    model="lgbm"
fi

cfg1="--task day_of_week --keycol day_of_year --sort_by day_of_year --target day_of_week --sql_templates_file day_of_week/templates.sql --model_type classifier --model_name $model --multi_class"
cfg2="--task is_weekend --keycol day_of_year --sort_by day_of_year --target is_weekend --sql_templates_file is_weekend/templates.sql --model_type classifier --model_name $model"
cfg3="--task hour_of_day --keycol hourstamp --sort_by hourstamp --target hour_of_day --sql_templates_file hour_of_day/templates.sql --model_type classifier --model_name $model --multi_class"
cfg4="--task is_night --keycol hourstamp --sort_by hourstamp --target is_night --sql_templates_file is_night/templates.sql --model_type classifier --model_name $model"
cfgs="$cfg1|$cfg2|$cfg3|$cfg4"

cfg=$(echo $cfgs | cut -d'|' -f$cfgid)
echo $cfg

python prepares.py $cfg

python fextractor.py $cfg --sample 0
python fextractor.py $cfg --sample 0.01
python fextractor.py $cfg --sample 0.1
python fextractor.py $cfg --sample 0.5

python pipeline.py $cfg

python test_pipeline.py $cfg --sample 0.01
python test_pipeline.py $cfg --sample 0.1
python test_pipeline.py $cfg --sample 0.5
