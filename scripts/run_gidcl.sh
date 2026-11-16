#!/usr/bin/env bash
# GIDCL (detecção) completo: runners/run_gidcl.py -> evaluate.py (tabela toda e, com HELDOUT=1, sem as tuplas rotuladas)
# uso: bash scripts/run_gidcl.sh --name <nome> --dirty <dirty.csv> --clean <clean.csv>
#        [--seed "0 1"] [--budget 20] [--clusters 20] [--select cluster|random] [--embed sbert|tfidf]
#        [--tau 0.85] [--iter-max 4] [--rounds 3] [--epochs 8] [--lm roberta-base] [--llm gpt|qwen]
#        [--outdir outputs] [--overwrite] [--results results.csv]
# (os parâmetros são os mesmos do runners/run_gidcl.py; --seed aceita várias sementes entre aspas;
#  o artigo usa budget 20, tau 0.85, 2 execuções e clusters: Hospital 20, Flights 20, Beers 10, Rayyan 10, Tax 30, IMDB 50)
# ambiente: GIDCL_PY=/caminho/do/python (padrão: ~/miniconda3/envs/gidcl/bin/python)   GIDCL_GPU=0
#           GIDCL_EXTRA="--batch 64 --lr 3e-5 --max-len 128"   (demais opções do runner, repassadas como estão)
#           LLM: --llm gpt (padrão) usa OPENAI_API_KEY, OPENAI_BASE_URL e OPENAI_MODEL do .env (artigo: gpt-4-turbo);
#                --llm qwen (ou GIDCL_LLM=qwen) usa Qwen2.5-72B-Instruct por vLLM em localhost:8000 (o do ZeroED);
#                QWEN_MODEL, QWEN_BASE_URL e QWEN_API_KEY trocam modelo e servidor
#           GIDCL_FILE=.../detector.npy -> só converte a detecção pronta dos autores (gidcl_released); exige
#             GIDCL_RUNTIME, GIDCL_PROMPT_TOKENS e GIDCL_COMPLETION_TOKENS; opcionais GIDCL_ORDER e GIDCL_SKIP
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh

usage() { sed -n '3,7p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }

PY=${GIDCL_PY:-$HOME/miniconda3/envs/gidcl/bin/python}
[ -x "$PY" ] || PY=python
NAME= DIRTY= CLEAN= SEEDS=0 OUTDIR=outputs OVERWRITE=
OPTS=()   # opções repassadas ao runner só quando informadas (os padrões ficam no runner)
while [ $# -gt 0 ]; do
  case $1 in
    --name)        NAME=${2:?}; shift 2 ;;
    --dirty)       DIRTY=${2:?}; shift 2 ;;
    --clean)       CLEAN=${2:?}; shift 2 ;;
    --seed|--seeds) SEEDS=${2:?}; shift 2 ;;
    --budget|--clusters|--select|--embed|--tau|--iter-max|--rounds|--epochs|--lm|--llm)
                   OPTS+=("$1" "${2:?}"); shift 2 ;;
    --outdir)      OUTDIR=${2:?}; shift 2 ;;
    --overwrite)   OVERWRITE=--overwrite; shift ;;
    --results)     RESULTS=${2:?}; shift 2 ;;
    -h|--help)     usage ;;
    *) echo "parâmetro desconhecido: $1" >&2; usage ;;
  esac
done
[ -n "$NAME" ] && [ -n "$DIRTY" ] && [ -n "$CLEAN" ] || usage
source scripts/load_env.sh && load_env "$PY"

mkdir -p logs
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")
LOG="logs/gidcl_${NAME}.log"

if [ -n "${GIDCL_FILE:-}" ]; then
  "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --outdir "$OUTDIR" $OVERWRITE \
        --from-file "$GIDCL_FILE" --order "${GIDCL_ORDER:-auto}" ${GIDCL_SKIP:+--skip-cols $GIDCL_SKIP} \
        --run-time "${GIDCL_RUNTIME:?defina GIDCL_RUNTIME}" \
        --prompt-tokens "${GIDCL_PROMPT_TOKENS:?defina GIDCL_PROMPT_TOKENS}" \
        --completion-tokens "${GIDCL_COMPLETION_TOKENS:?defina GIDCL_COMPLETION_TOKENS}" 2>&1 | tee -a "$LOG"
  evaluate "$PY" "$OUTDIR/$NAME/gidcl_released" "$GT"
  exit 0
fi

for s in $SEEDS; do
  "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" \
        "${OPTS[@]}" ${GIDCL_GPU:+--gpu "$GIDCL_GPU"} ${GIDCL_EXTRA:-} \
        --outdir "$OUTDIR" $OVERWRITE 2>&1 | tee -a "$LOG"
  # o nome da pasta é montado pelo runner: lê da última linha "... -> <pasta>/pred.csv"
  D=$(grep -o ' -> .*/pred\.csv$' "$LOG" | tail -1 | sed 's/^ -> //; s/\/pred\.csv$//')
  [ -f "$D/pred.csv" ] || { echo "saída do runner não encontrada" >&2; exit 1; }
  evaluate "$PY" "$D" "$GT"
  # opcional (HELDOUT=1): avalia também sem as tuplas rotuladas
  if [ -n "${HELDOUT:-}" ]; then
    evaluate "$PY" "$D" "$GT" --method "$(basename "$D")_heldout" --exclude-rows "$D/labeled_rows.csv"
  fi
done
