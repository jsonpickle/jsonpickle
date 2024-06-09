#!/bin/bash
DATEANDTIME=$(date +%Y-%m-%dT%T%z)
pytest --benchmark-only --benchmark-disable-gc --benchmark-json=./benchmark-new.json --benchmark-histogram=../images/benchmark-$DATEANDTIME ../jsonpickle_benchmarks.py
# assuming the user is doing work in a different branch than main
git checkout main
pytest --benchmark-only --benchmark-disable-gc --benchmark-json=./benchmark-main.json --benchmark-histogram=../images/benchmark-$DATEANDTIME ../jsonpickle_benchmarks.py
# return to previous branch
git checkout -
pytest-benchmark compare --columns=median --name=long --sort=fullname --csv=perf --histogram=perf benchmark-main.json benchmark-new.json
python3 analyze_benchmarks.py
