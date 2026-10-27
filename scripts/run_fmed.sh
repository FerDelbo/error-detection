#!/usr/bin/env bash
# uso: bash scripts/run_fmed.sh <nome> <dirty_COMPLETO.csv> <clean_COMPLETO.csv> ["0 1 2"] [k]
set -eo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
NAME=${1:?}; DIRTY=${2:?}; CLEAN=${3:?}; SEEDS=${4:-"0"}; K=${5:-10}
mkdir -p logs

for s in $SEEDS; do
  python runners/run_fmed.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" --k "$K" \
    2>&1 | tee -a "logs/fmed_${NAME}.log"
  python evaluate.py --name "$NAME" --method "fmed_s${s}" \
    --dirty "outputs/${NAME}/fmed_s${s}/eval_dirty.csv" \
    --clean "outputs/${NAME}/fmed_s${s}/eval_clean.csv"
done