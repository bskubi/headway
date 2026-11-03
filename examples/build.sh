#!/usr/bin/env bash
# Rebuild every example's pages with the headway installed now (run after changing headway, the viewer above all):
# cellscore (its benchmark, each version's headway, the comparison and the suite), the pipeline, and the models'
# drafts. PYTHON=... chooses the interpreter (default: the repository's .venv, else python).
set -e
cd "$(dirname "$0")"
here=$PWD
PY=${PYTHON:-$( [ -x ../.venv/bin/python ] && echo "$here/../.venv/bin/python" || echo python )}
export PYTHON=$PY
./cellscore/build.sh
./pipeline/build.sh
cd "$here/models"
for spec in de_fit_spec em_spec; do
  "$PY" -m headway.cli build "$spec" -o "$spec.html"
done
find "$here" -name __pycache__ -prune -exec rm -rf {} +
