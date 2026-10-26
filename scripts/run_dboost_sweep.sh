#!/usr/bin/env bash
# uso: bash scripts/run_dboost_sweep.sh <nome> <dirty.csv> <clean.csv>
set -eo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?}; DIRTY=${2:?}; CLEAN=${3:?}
mkdir -p logs

run() {
  python runners/run_dboost.py --name "$NAME" --dirty "$DIRTY" --config "$@" 2>&1 | tee -a "logs/dboost_${NAME}.log"
  python evaluate.py --name "$NAME" --method "dboost_$(IFS=_; echo "$*")" --dirty "$DIRTY" --clean "$CLEAN"
}

for g in 1.0 1.3 1.5 1.7 2.0 2.3 2.5 2.7 3.0; do run gaussian $g; done
for a in 0.1 0.3 0.5 0.7 0.9; do for b in 0.1 0.3 0.5 0.7 0.9; do run histogram $a $b; done; done