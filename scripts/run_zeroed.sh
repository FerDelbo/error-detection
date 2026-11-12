#!/usr/bin/env bash
# uso: [RESULTS=results.csv] [CLUSTER=KMeans] ZEROED_PY=/caminho/do/python bash scripts/run_zeroed.sh <nome> <dirty.csv> <clean.csv> [execuções]
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
[ -f .env ] && { set -a; source .env; set +a; }

PY=${ZEROED_PY:-$HOME/miniconda3/envs/zeroed/bin/python}
[ -x "$PY" ] || PY=python
NAME=${1:?uso: run_zeroed.sh <nome> <dirty.csv> <clean.csv> [execuções]}
DIRTY=${2:?}; CLEAN=${3:?}; RUNS=${4:-1}
CLUSTER=${CLUSTER:-RANDOM}   # RANDOM (código publicado) | KMeans (artigo) | DBSCAN | Agglomerative
SUF=""; [ "$CLUSTER" = RANDOM ] || SUF="_$(echo "$CLUSTER" | tr '[:upper:]' '[:lower:]')"
mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")

"$PY" runners/run_zeroed.py all --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
      --runs "$RUNS" --cluster "$CLUSTER" 2>&1 | tee -a "logs/zeroed${SUF}_${NAME}.log"

for s in $(seq 1 "$RUNS"); do
  evaluate "$PY" "outputs/$NAME/zeroed${SUF}_s${s}" "$GT"
done
