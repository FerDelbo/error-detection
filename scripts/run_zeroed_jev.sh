#!/usr/bin/env bash
# ZeroED com o Jev como rotulador (methods/zeroed_jev). Não toca no ZeroED original.
# uso: [VARIANTS="base jevlab jevlab_feat jevlab_aug"] [CLUSTER=RANDOM] [RESULTS=results.csv] \
#      ZEROED_PY=/caminho/do/python bash scripts/run_zeroed_jev.sh <nome> <dirty.csv> <clean.csv> [execuções]
# LLM: Qwen (QWEN_MODEL/QWEN_BASE_URL/QWEN_API_KEY; padrão Qwen2.5-14B-Instruct em localhost:8000).
# Jev: JEV_URL, JEV_API_KEY, JEV_MODEL no .env; JEV_BACKEND=mock testa o encanamento sem a API.
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
[ -f .env ] && { set -a; source .env; set +a; }

PY=${ZEROED_PY:-$HOME/miniconda3/envs/zeroed/bin/python}
[ -x "$PY" ] || PY=python
NAME=${1:?uso: run_zeroed_jev.sh <nome> <dirty.csv> <clean.csv> [execuções]}
DIRTY=${2:?}; CLEAN=${3:?}; RUNS=${4:-1}
VARIANTS=${VARIANTS:-"base jevlab jevlab_feat jevlab_aug"}
CLUSTER=${CLUSTER:-RANDOM}
SUF=""; [ "$CLUSTER" = RANDOM ] || SUF="_$(echo "$CLUSTER" | tr '[:upper:]' '[:lower:]')"
MOCK=""; [ "${JEV_BACKEND:-api}" = mock ] && MOCK="_mock"
mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")

for V in $VARIANTS; do
  "$PY" runners/run_zeroed_jev.py all --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
        --variant "$V" --runs "$RUNS" --cluster "$CLUSTER" 2>&1 | tee -a "logs/zeroed_jev_${V}${SUF}_${NAME}.log"
  M="$MOCK"; [ "$V" = base ] && M=""   # a base não usa o Jev
  for s in $(seq 1 "$RUNS"); do
    evaluate "$PY" "outputs/$NAME/zeroed_${V}_qwen${SUF}${M}_s${s}" "$GT"
  done
done
