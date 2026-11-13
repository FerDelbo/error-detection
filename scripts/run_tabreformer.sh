#!/usr/bin/env bash
# TabReformer completo: runners/run_tabreformer.py -> evaluate.py (tabela toda e, com HELDOUT=1, sem as tuplas rotuladas)
# uso: bash scripts/run_tabreformer.sh --name <nome> --dirty <dirty.csv> --clean <clean.csv>
#        [--seed "0 1 2"] [--budget 20|5%] [--augment learned|generic|none] [--gamma 1]
#        [--protocol paper|released] [--model bert-base-uncased] [--outdir outputs] [--overwrite] [--results results.csv]
# (os parâmetros são os mesmos do runners/run_tabreformer.py; --seed aceita várias sementes entre aspas)
# ambiente: TABREFORMER_PY=/caminho/do/python (padrão: ~/miniconda3/envs/tabreformer/bin/python)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

usage() { sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

PY=${TABREFORMER_PY:-$HOME/miniconda3/envs/tabreformer/bin/python}
[ -x "$PY" ] || PY=python
NAME= DIRTY= CLEAN= SEEDS=0 BUDGET=20 AUGMENT=learned GAMMA=1 PROTOCOL=paper MODEL=bert-base-uncased
OUTDIR=outputs OVERWRITE=
while [ $# -gt 0 ]; do
  case $1 in
    --name)        NAME=${2:?}; shift 2 ;;
    --dirty)       DIRTY=${2:?}; shift 2 ;;
    --clean)       CLEAN=${2:?}; shift 2 ;;
    --seed|--seeds) SEEDS=${2:?}; shift 2 ;;
    --budget)      BUDGET=${2:?}; shift 2 ;;
    --augment)     AUGMENT=${2:?}; shift 2 ;;
    --gamma)       GAMMA=${2:?}; shift 2 ;;
    --protocol)    PROTOCOL=${2:?}; shift 2 ;;
    --model)       MODEL=${2:?}; shift 2 ;;
    --outdir)      OUTDIR=${2:?}; shift 2 ;;
    --overwrite)   OVERWRITE=--overwrite; shift ;;
    --results)     RESULTS=${2:?}; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "parâmetro desconhecido: $1" >&2; usage ;;
  esac
done
[ -n "$NAME" ] && [ -n "$DIRTY" ] && [ -n "$CLEAN" ] || usage

mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")
LOG="logs/tabreformer_${NAME}.log"
for s in $SEEDS; do
  "$PY" runners/run_tabreformer.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
        --protocol "$PROTOCOL" --budget "$BUDGET" --augment "$AUGMENT" --gamma "$GAMMA" \
        --model "$MODEL" --seed "$s" --outdir "$OUTDIR" $OVERWRITE 2>&1 | tee -a "$LOG"
  # o nome da pasta é montado pelo runner: lê da última linha "... -> <pasta>/pred.csv"
  D=$(grep -o ' -> .*/pred\.csv$' "$LOG" | tail -1 | sed 's/^ -> //; s/\/pred\.csv$//')
  [ -f "$D/pred.csv" ] || { echo "saída do runner não encontrada" >&2; exit 1; }
  evaluate "$PY" "$D" "$GT"
  # opcional (HELDOUT=1): avalia também sem as tuplas rotuladas (só existe no protocolo paper)
  if [ -n "${HELDOUT:-}" ] && [ "$PROTOCOL" = paper ]; then
    evaluate "$PY" "$D" "$GT" --method "$(basename "$D")_heldout" --exclude-rows "$D/labeled_rows.csv"
  fi
done
