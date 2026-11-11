#!/usr/bin/env bash
# Raha completo: runners/run_raha.py -> evaluate.py (tabela toda; HELDOUT=1 avalia também sem as tuplas rotuladas)
# uso: bash scripts/run_raha.sh --name <nome> --dirty <dirty.csv> --clean <clean.csv>
#        [--seed "0 1 2"] [--budget 20] [--algorithms OD,PVD,RVD,KBVD] [--classifier GBC]
#        [--outdir outputs] [--overwrite] [--results results.csv]
# (os parâmetros são os mesmos do runners/run_raha.py; --seed aceita várias sementes entre aspas)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

usage() { sed -n '3,6p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

NAME= DIRTY= CLEAN= SEEDS=0 BUDGET=20 ALGORITHMS=OD,PVD,RVD,KBVD CLASSIFIER=GBC OUTDIR=outputs OVERWRITE=
while [ $# -gt 0 ]; do
  case $1 in
    --name)        NAME=${2:?}; shift 2 ;;
    --dirty)       DIRTY=${2:?}; shift 2 ;;
    --clean)       CLEAN=${2:?}; shift 2 ;;
    --seed|--seeds) SEEDS=${2:?}; shift 2 ;;
    --budget)      BUDGET=${2:?}; shift 2 ;;
    --algorithms)  ALGORITHMS=${2:?}; shift 2 ;;
    --classifier)  CLASSIFIER=${2:?}; shift 2 ;;
    --outdir)      OUTDIR=${2:?}; shift 2 ;;
    --overwrite)   OVERWRITE=--overwrite; shift ;;
    --results)     RESULTS=${2:?}; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "parâmetro desconhecido: $1" >&2; usage ;;
  esac
done
[ -n "$NAME" ] && [ -n "$DIRTY" ] && [ -n "$CLEAN" ] || usage

# mesmo nome de pasta que o runners/run_raha.py monta
ALGS=
IFS=, read -ra _algs <<< "${ALGORITHMS^^}"
for a in "${_algs[@]}"; do
  a=${a// /}
  [ -n "$a" ] && [[ "+$ALGS+" != *"+$a+"* ]] && ALGS+="${ALGS:++}$a"
done
SUFFIX=
[ "$ALGS" != "OD+PVD+RVD+KBVD" ] && SUFFIX+="_$ALGS"
[ "$CLASSIFIER" != "GBC" ] && SUFFIX+="_$CLASSIFIER"

mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN")
for s in $SEEDS; do
  python runners/run_raha.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" \
         --budget "$BUDGET" --seed "$s" --algorithms "$ALGORITHMS" --classifier "$CLASSIFIER" \
         --outdir "$OUTDIR" $OVERWRITE 2>&1 | tee -a "logs/raha_${NAME}_b${BUDGET}.log"
  M="raha_b${BUDGET}_s${s}${SUFFIX}"
  D="$OUTDIR/$NAME/$M"
  evaluate python "$D" "$GT"
  # opcional (HELDOUT=1): avalia também sem as tuplas rotuladas
  if [ -n "${HELDOUT:-}" ]; then
    evaluate python "$D" "$GT" --method "${M}_heldout" --exclude-rows "$D/labeled_rows.csv"
  fi
done
