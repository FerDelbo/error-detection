#!/usr/bin/env bash
# uso: bash scripts/run_raha.sh <nome> <dirty.csv> <clean.csv> ["sementes"] [budget]
set -eo pipefail
cd "$(dirname "$0")/.."

NAME=${1:?uso: run_raha.sh <nome> <dirty.csv> <clean.csv> [sementes] [budget]}
DIRTY=${2:?falta o caminho do dirty.csv}
CLEAN=${3:?falta o caminho do clean.csv}
SEEDS=${4:-"0 1 2"}
BUDGET=${5:-2}

mkdir -p logs
for s in $SEEDS; do
  python runners/run_raha.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
         --budget "$BUDGET" --seed "$s" 2>&1 | tee -a "logs/raha_${NAME}.log"
  python evaluate.py --name "$NAME" --method "raha_s${s}" --dirty "$DIRTY" --clean "$CLEAN"
done