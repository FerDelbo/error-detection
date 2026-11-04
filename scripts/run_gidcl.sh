#!/usr/bin/env bash
# GIDCL (detecção) completo: seleção das tuplas -> criador-crítico (GPU) -> avaliação
# uso: GIDCL_PY=/caminho/python bash scripts/run_gidcl.sh <nome> <dirty.csv> <clean.csv> ["0 1 2"]
# opções: GIDCL_K=20  GIDCL_CLUSTERS=20  GIDCL_GPU=0  GIDCL_EMBED=sbert|tfidf
#         GIDCL_EXTRA="--rounds 3 --epochs 8"   (repassado ao gidcl_detect.py)
#         GIDCL_FILE=.../detector.npy  -> modo antigo: converte o detector pronto do artigo
set -eo pipefail
cd "$(dirname "$0")/.."

PY=${GIDCL_PY:-python}
source scripts/load_env.sh && load_env "$PY"

NAME=${1:?uso: run_gidcl.sh <nome> <dirty.csv> <clean.csv> [sementes]}
DIRTY=${2:?}; CLEAN=${3:?}; SEEDS=${4:-0}
K=${GIDCL_K:-20}; CL=${GIDCL_CLUSTERS:-$K}; GPU=${GIDCL_GPU:-0}; EMBED=${GIDCL_EMBED:-sbert}
EXTRA=${GIDCL_EXTRA:-}
mkdir -p logs

if [ -n "${GIDCL_FILE:-}" ]; then
  for s in $SEEDS; do
    "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" \
          --from-file "$GIDCL_FILE" --order "${GIDCL_ORDER:-auto}" ${GIDCL_SKIP:+--skip-cols $GIDCL_SKIP}
    "$PY" evaluate.py --name "$NAME" --method "gidcl_s${s}" --dirty "$DIRTY" --clean "$CLEAN"
  done
  exit 0
fi

for s in $SEEDS; do
  SPLIT=outputs/$NAME/gidcl_split_s$s

  # 1) o método só enxerga a tabela suja + as K tuplas rotuladas (o clean completo fica de fora)
  "$PY" scripts/make_labeled_split.py --dirty "$DIRTY" --clean "$CLEAN" --k "$K" --clusters "$CL" \
        --seed "$s" --select cluster --embed "$EMBED" --outdir "$SPLIT"

  # 2) criador-crítico na GPU
  "$PY" runners/run_gidcl.py --name "$NAME" --dirty "$DIRTY" --clean "$CLEAN" --seed "$s" --gpu "$GPU" \
        --split-dir "$SPLIT" --python "$PY" --gidcl-dir methods/gidcl_det \
        --cmd "gidcl_detect.py --dirty {dirty} --labeled {labeled_csv} --clusters {split}/clusters.csv --outdir {outdir} --seed {seed} $EXTRA" \
        --output "{outdir}/pred_matrix.npy" 2>&1 | tee -a "logs/gidcl_${NAME}.log"

  # 3) avaliação (clean verdadeiro, fora do processo de treino)
  "$PY" evaluate.py --name "$NAME" --method "gidcl_s${s}" --dirty "$DIRTY" --clean "$CLEAN"
  if grep -q "exclude-rows" evaluate.py; then
    "$PY" evaluate.py --name "$NAME" --method "gidcl_s${s}_heldout" --pred "outputs/$NAME/gidcl_s${s}/pred.csv" \
          --exclude-rows "$SPLIT/labeled_rows.csv" --dirty "$DIRTY" --clean "$CLEAN"
  fi
done