#!/bin/bash
DATEANDTIME=$(date +%Y-%m-%dT%T%z)

# assume user is running this cd'ed into this dir
cd ..

# we assume the user is running on Linux because I don't believe the benchmark setup works on MacOS or Windows anyway
# user will need to be willing to install `util-linux` on Ubuntu
make benchmark EXTRA_BENCH_ARGS="--benchmark-json=./benchmarking/benchmark-new.json"
# assuming the user is doing work in a different branch than main
git checkout main

make benchmark EXTRA_BENCH_ARGS="--benchmark-json=./benchmarking/benchmark-main.json"
# return to previous branch
git checkout -

cd benchmarking

pytest-benchmark compare --columns=median --name=long --sort=fullname --csv=perf --histogram=perf benchmark-main.json benchmark-new.json
python3 analyze_benchmarks.py
