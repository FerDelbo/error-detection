#!/usr/bin/env bash
# Modo arquivo pronto:  GIDCL_FILE=.../detector.npy [GIDCL_SKIP="0 1"] [GIDCL_ORDER=auto] \
#                       bash scripts/run_gidcl.sh <nome> <dirty.csv> <clean.csv> [sementes]
# Modo comando:         GIDCL_CMD="..." GIDCL_OUT="..." GIDCL_PY=... bash scripts/run_gidcl.sh ...
set -eo pipefail
cd "$(dirname "$0")/.."

PY=${GIDCL_PY:-python}
source scripts/load_env.sh && load_env "$PY"

NAME=${1:?uso: run_gidcl.sh <nome> <dirty.csv> <clean.csv> [sementes]}
DIRTY=${2:?}; CLEAN=${3:?}; SEEDS=${4:-0}
mkdir -p logs

for s in $SEEDS; do
  if [ -n "${GIDCL_FILE:-}" ]; then
    "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" \
          --from-file "$GIDCL_FILE" --order "${GIDCL_ORDER:-auto}" ${GIDCL_SKIP:+--skip-cols $GIDCL_SKIP}
  else
    : "${GIDCL_CMD:?defina GIDCL_FILE (arquivo pronto) ou GIDCL_CMD e GIDCL_OUT}"
    : "${GIDCL_OUT:?defina GIDCL_OUT}"
    "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" \
          --python "$PY" --cmd "$GIDCL_CMD" --output "$GIDCL_OUT" ${GIDCL_SKIP:+--skip-cols $GIDCL_SKIP}
  fi
  "$PY" evaluate.py --name "$NAME" --method "gidcl_s${s}" --dirty "$DIRTY" --clean "$CLEAN"
done