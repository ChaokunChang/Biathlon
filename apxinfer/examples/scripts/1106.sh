nparts=2
model="lgbm"
python run.py --example cheaptrips --stage prepare --task test/cheaptrips --model $model --nparts $nparts
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --nparts $nparts --ncores 0 
python eval_cheaptrips.py --nparts $nparts --ncores 1

model="xgb"
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

nparts=5
model="lgbm"
python run.py --example cheaptrips --stage prepare --task test/cheaptrips --model $model --nparts $nparts
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

model="xgb"
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

nparts=10
model="lgbm"
python run.py --example cheaptrips --stage prepare --task test/cheaptrips --model $model --nparts $nparts
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

model="xgb"
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

nparts=20
model="lgbm"
python run.py --example cheaptrips --stage prepare --task test/cheaptrips --model $model --nparts $nparts
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

model="xgb"
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

nparts=100
model="lgbm"
python run.py --example cheaptrips --stage prepare --task test/cheaptrips --model $model --nparts $nparts
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1

model="xgb"
python run.py --example cheaptrips --stage train --task test/cheaptrips --model $model --nparts $nparts
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 0 
python eval_cheaptrips.py --model $model --nparts $nparts --ncores 1
