#!/usr/bin/env bash
# Avalia as variantes de saída do GIDCL (modelo, só funções, união) e o resultado sem as tuplas rotuladas.
# uso: bash scripts/eval_gidcl_variants.sh <nome> <dirty.csv> <clean.csv> [semente=0]
set -eo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?}; DIRTY=${2:?}; CLEAN=${3:?}; S=${4:-0}
PY=${GIDCL_PY:-python}
D=outputs/$NAME/gidcl_s$S
SPLIT=outputs/$NAME/gidcl_split_s$S

for v in matrix:model functions:funcs union:union; do
  f=${v%%:*}; tag=${v##*:}; M="gidcl_${tag}_s${S}"
  [ -f "$D/pred_$f.npy" ] || continue
  if ! "$PY" - "$DIRTY" "$D/pred_$f.npy" "outputs/$NAME/$M" <<'EOF'
import os, sys
import numpy as np, pandas as pd
dirty, npy, out = sys.argv[1:4]
m = np.load(npy)
print(f"{os.path.basename(npy)}: células marcadas = {int(m.sum())} de {m.size}")
if m.sum() == 0:
    sys.exit(3)
os.makedirs(out, exist_ok=True)
pd.DataFrame(m, columns=pd.read_csv(dirty, nrows=0).columns).to_csv(out + "/pred.csv", index=False)
EOF
  then echo "  (sem células marcadas: variante ignorada)"; continue; fi
  "$PY" evaluate.py --name "$NAME" --method "$M" --dirty "$DIRTY" --clean "$CLEAN"
  if grep -q "exclude-rows" evaluate.py; then
    "$PY" evaluate.py --name "$NAME" --method "${M}_heldout" --pred "outputs/$NAME/$M/pred.csv" \
          --exclude-rows "$SPLIT/labeled_rows.csv" --dirty "$DIRTY" --clean "$CLEAN"
  fi
done
