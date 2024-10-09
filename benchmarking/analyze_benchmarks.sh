#!/bin/bash
DATEANDTIME=$(date +%Y-%m-%dT%T%z)
# we assume the user is running on Linux because I don't believe the benchmark setup works on MacOS or Windows anyway
# user will need to have `schedutils` installed
taskset -c 1 pytest --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=10000 --benchmark-json=./benchmark-new.json --benchmark-histogram=../images/benchmark-$DATEANDTIME ../jsonpickle_benchmarks.py
# assuming the user is doing work in a different branch than main
git checkout main
taskset -c 1 pytest --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=10000 --benchmark-json=./benchmark-main.json --benchmark-histogram=../images/benchmark-$DATEANDTIME ../jsonpickle_benchmarks.py
# return to previous branch
git checkout -
pytest-benchmark compare --columns=median --name=long --sort=fullname --csv=perf --histogram=perf benchmark-main.json benchmark-new.json
python3 analyze_benchmarks.py
