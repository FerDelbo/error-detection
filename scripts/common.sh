# Funções comuns aos scripts/run_*.sh.  uso: source scripts/common.sh

# CSV onde o evaluate.py acumula os resultados (sobrescreva com RESULTS=... no ambiente)
RESULTS=${RESULTS:-results.csv}

# ensure_gt <dirty.csv> <clean.csv> [python]
# Gera <pasta do dirty>/GROUND_TRUTH.csv se não existir ou se dirty/clean forem mais novos; imprime o caminho.
ensure_gt() {
  local dirty=${1:?} clean=${2:?} py=${3:-python}
  local gt
  gt="$(dirname "$dirty")/GROUND_TRUTH.csv"
  if [ ! -f "$gt" ] || [ "$dirty" -nt "$gt" ] || [ "$clean" -nt "$gt" ]; then
    "$py" scripts/make_ground_truth.py --dirty "$dirty" --clean "$clean" --out "$gt" >&2 \
      || { echo "falha ao gerar $gt" >&2; return 1; }
  fi
  echo "$gt"
}

# evaluate <python> <pasta da saída do método> <GROUND_TRUTH.csv> [args extras do evaluate.py]
# Exige pred.csv e metadata.json na pasta.
evaluate() {
  local py=${1:?} dir=${2:?} gt=${3:?}
  shift 3
  "$py" evaluate.py --pred "$dir/pred.csv" --metadata "$dir/metadata.json" \
        --ground-truth "$gt" --results "$RESULTS" "$@"
}
