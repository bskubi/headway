#!/bin/sh
# Runs each version of cellscore on the benchmark and builds its headway with the results (bench/run.py), adding
# each to an atlas (atlas/); then the comparison page (v4 with v1-v3 in lanes above it) and the atlas's page. Each
# vN/ directory stands in for a checkout of the method at that point.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONPATH="$PWD/bench${PYTHONPATH:+:$PYTHONPATH}"     # the benchmark suite, cellscore_bench
rm -rf atlas                           # made again from the versions as they are now
$PY bench/run.py                       # results/vN.json, headways/vN.{html,json} and atlas/, through headway's Python API
cd v4
$PY -m headway.cli build cellscore.graphspec --name v4 --key centered=true --key weighted=true \
  --results ../results/v4.json -o ../compare.html \
  --compare ../headways/v1.json --compare ../headways/v2.json --compare ../headways/v3.json
cd ..
$PY -m headway.cli build cellscore_bench.suite -o suite.html                 # the benchmark's own headway
$PY -m headway.cli relate atlas update "v3->v4" --outcome "Better on the noisy module's split agreement only; elsewhere within the uncertainty."
$PY -m headway.cli atlas view atlas "the line" v1 v2 v3 v4 --baseline v1
$PY -m headway.cli page atlas -o atlas.html                                  # every version, explored in one page
$PY -m headway.cli diff headways/v1.json headways/v2.json headways/v3.json headways/v4.json > changes.txt
$PY -m headway.cli results headways/v1.json headways/v2.json headways/v3.json headways/v4.json > results.txt
find . -name __pycache__ -exec rm -rf {} +
