#!/usr/bin/env bash
# uso: ZEROED_PY=/caminho/do/python bash scripts/run_zeroed.sh <nome> <dirty.csv> <clean.csv> [execuções]
set -eo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }

PY=${ZEROED_PY:-python}
NAME=${1:?uso: run_zeroed.sh <nome> <dirty.csv> <clean.csv> [execuções]}
DIRTY=${2:?}; CLEAN=${3:?}; RUNS=${4:-1}
mkdir -p logs

"$PY" runners/run_zeroed.py all --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
      --runs "$RUNS" 2>&1 | tee -a "logs/zeroed_${NAME}.log"

for s in $(seq 1 "$RUNS"); do
  "$PY" evaluate.py --name "$NAME" --method "zeroed_s${s}" --dirty "$DIRTY" --clean "$CLEAN"
done