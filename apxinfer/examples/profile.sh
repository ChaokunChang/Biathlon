python -m cProfile -s cumtime -o ~/.cache/apxinf/xip/online.cprof run.py --stage online --task test/trips --model lgbm --nreqs 20 --offline_nreqs 20 --pest_constraint error --max_error 0.5 --min_conf 0.95
pip install gprof2dot
gprof2dot -f pstats ~/.cache/apxinf/xip/online.pstats | dot -Tsvg -o ~/.cache/apxinf/xip/online.svg
