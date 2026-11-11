#!/usr/bin/env bash
# Roda todas as combinações de parâmetros de um algoritmo (raha ou dboost) numa base:
# para cada combinação chama o scripts/run_<algoritmo>.sh (runner + evaluate.py).
#
# uso: bash scripts/run_grid.sh <raha|dboost> <nome> <dirty.csv> <clean.csv> <results.csv>
#   ex.: bash scripts/run_grid.sh dboost hospital dataset/hospital/dirty.csv \
#             dataset/hospital/clean.csv outputs/results.csv
#
# opções no ambiente:
#   OUTDIR=outputs          pasta onde ficam os pred.csv/metadata.json
#   DRY_RUN=1               só mostra os comandos, não executa
#   CONTINUE_ON_ERROR=1     se uma combinação falhar, segue para a próxima
#   PY_ENV=~/miniconda3/envs/raha/bin   python usado pelos runners (se existir, vai para o PATH)
set -eo pipefail
cd "$(dirname "$0")/.."

# ============================ GRADES DE PARÂMETROS ============================
# Edite aqui. Valores segundo o artigo do Raha (SIGMOD 2019, seção 4.1 e seção 6).

# --- Raha: produto cartesiano budget x seed x algorithms x classifier
RAHA_BUDGETS=(5 10 15 20)            # artigo: 20 tuplas rotuladas como padrão
RAHA_SEEDS=(0 1 2)
RAHA_ALGORITHMS=("OD,PVD,RVD,KBVD")  # padrão do artigo; ex.: adicionar "OD,PVD,RVD" para ablação
RAHA_CLASSIFIERS=(GBC)               # padrão do artigo; opções: GBC ABC DTC GNB KNC SGDC SVC

# --- dBoost: gaussian (θ_dist) e histogram (peak x θ_tf)
DBOOST_GAUSSIAN=(1.0 1.3 1.5 1.7 2.0 2.3 2.5 2.7 3.0)            # artigo: 9 estratégias
DBOOST_HIST_PEAK=(0.8 0.9)                                        # artigo não define o peak
DBOOST_HIST_OUTLIER=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)         # artigo: θ_tf, 9 estratégias
# ==============================================================================

usage() { sed -n '5,7p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }
[ $# -eq 5 ] || usage
ALGO=$1 NAME=$2 DIRTY=$3 CLEAN=$4 RESULTS_CSV=$5
OUTDIR=${OUTDIR:-outputs}
for f in "$DIRTY" "$CLEAN"; do [ -f "$f" ] || { echo "arquivo não encontrado: $f" >&2; exit 1; }; done

PY_ENV=${PY_ENV:-$HOME/miniconda3/envs/raha/bin}
[ -x "$PY_ENV/python" ] && export PATH="$PY_ENV:$PATH"
command -v python >/dev/null || { echo "python não encontrado (ative o env ou defina PY_ENV)" >&2; exit 1; }

COMMON=(--name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --outdir "$OUTDIR" --results "$RESULTS_CSV")
TOTAL=0 OK=0 FAILED=()

run() {  # run <descrição> <comando...>
  local desc=$1; shift
  TOTAL=$((TOTAL + 1))
  echo "==> [$TOTAL] $ALGO $NAME: $desc"
  if [ -n "${DRY_RUN:-}" ]; then echo "    $*"; OK=$((OK + 1)); return; fi
  if "$@"; then OK=$((OK + 1))
  else
    FAILED+=("$desc")
    [ -n "${CONTINUE_ON_ERROR:-}" ] || { echo "falhou: $desc (use CONTINUE_ON_ERROR=1 para seguir)" >&2; exit 1; }
  fi
}

case $ALGO in
  raha)
    for b in "${RAHA_BUDGETS[@]}"; do
      for a in "${RAHA_ALGORITHMS[@]}"; do
        for c in "${RAHA_CLASSIFIERS[@]}"; do
          for s in "${RAHA_SEEDS[@]}"; do
            run "budget=$b algorithms=$a classifier=$c seed=$s" \
                bash scripts/run_raha.sh "${COMMON[@]}" --budget "$b" --seed "$s" \
                     --algorithms "$a" --classifier "$c"
          done
        done
      done
    done ;;
  dboost)
    for g in "${DBOOST_GAUSSIAN[@]}"; do
      run "gaussian $g" bash scripts/run_dboost_sweep.sh "${COMMON[@]}" --config gaussian "$g"
    done
    for p in "${DBOOST_HIST_PEAK[@]}"; do
      for o in "${DBOOST_HIST_OUTLIER[@]}"; do
        run "histogram $p $o" bash scripts/run_dboost_sweep.sh "${COMMON[@]}" --config histogram "$p" "$o"
      done
    done ;;
  *) echo "algoritmo desconhecido: $ALGO (use raha ou dboost)" >&2; usage ;;
esac

echo "==> fim: $OK/$TOTAL combinações ok -> $RESULTS_CSV"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf '    falhou: %s\n' "${FAILED[@]}" >&2
  exit 1
fi
