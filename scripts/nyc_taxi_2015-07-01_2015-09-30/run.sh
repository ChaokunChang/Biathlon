cfg="--task is_weekend --keycol day_of_year --sort_by day_of_year --target is_weekend --sql_templates_file is_weekend/templates.sql --model_type classifier"
# cfg="--task day_of_week --keycol day_of_year --sort_by day_of_year --target day_of_week --sql_templates_file day_of_week/templates.sql --model_type classifier"
model="lgbm"

python fextractor.py $cfg --sample 0
python fextractor.py $cfg --sample 0.01
python fextractor.py $cfg --sample 0.1
python fextractor.py $cfg --sample 0.5

python pipeline.py $cfg --model_name $model
python test_pipeline.py $cfg --model_name $model --sample 0.01
python test_pipeline.py $cfg --model_name $model --sample 0.1
python test_pipeline.py $cfg --model_name $model --sample 0.5
