#!/usr/bin/env bash
# dBoost completo: runners/run_dboost.py -> evaluate.py
# uso: bash scripts/run_dboost_sweep.sh --name <nome> --dirty <dirty.csv> --clean <clean.csv>
#        [--config gaussian 1.5 [--config histogram 0.9 0.1 ...]] [--outdir outputs] [--results results.csv]
# (parâmetros iguais aos do runners/run_dboost.py; sem --config roda a varredura completa)
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

usage() { sed -n '3,5p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

NAME= DIRTY= CLEAN= OUTDIR=outputs
CONFIGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --name)    NAME=${2:?}; shift 2 ;;
    --dirty)   DIRTY=${2:?}; shift 2 ;;
    --clean)   CLEAN=${2:?}; shift 2 ;;
    --outdir)  OUTDIR=${2:?}; shift 2 ;;
    --results) RESULTS=${2:?}; shift 2 ;;
    --config)  # pega os valores até o próximo --flag, como o nargs="+" do runner
      shift; c=
      while [ $# -gt 0 ] && [[ $1 != --* ]]; do c+="${c:+ }$1"; shift; done
      [ -n "$c" ] || { echo "--config precisa de valores (ex.: --config gaussian 1.5)" >&2; exit 1; }
      CONFIGS+=("$c") ;;
    -h|--help) usage ;;
    *) echo "parâmetro desconhecido: $1" >&2; usage ;;
  esac
done
[ -n "$NAME" ] && [ -n "$DIRTY" ] && [ -n "$CLEAN" ] || usage

if [ ${#CONFIGS[@]} -eq 0 ]; then
  for g in 1.0 1.3 1.5 1.7 2.0 2.3 2.5 2.7 3.0; do CONFIGS+=("gaussian $g"); done
  for a in 0.1 0.3 0.5 0.7 0.9; do for b in 0.1 0.3 0.5 0.7 0.9; do CONFIGS+=("histogram $a $b"); done; done
fi

mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN")
for c in "${CONFIGS[@]}"; do
  read -ra cfg <<< "$c"
  python runners/run_dboost.py --name "$NAME" --dirty "$DIRTY" --config "${cfg[@]}" \
         --outdir "$OUTDIR" 2>&1 | tee -a "logs/dboost_${NAME}.log"
  evaluate python "$OUTDIR/$NAME/dboost_$(IFS=_; echo "${cfg[*]}")" "$GT"
done
