#!/usr/bin/env bash
# Avalia as variantes de saída do GIDCL (modelo, só funções, união) e o resultado sem as tuplas rotuladas.
# uso: [RESULTS=results.csv] bash scripts/eval_gidcl_variants.sh <nome> <dirty.csv> <clean.csv> [semente=0]
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
NAME=${1:?}; DIRTY=${2:?}; CLEAN=${3:?}; S=${4:-0}
PY=${GIDCL_PY:-python}
D=outputs/$NAME/gidcl_s$S
SPLIT=outputs/$NAME/gidcl_split_s$S
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")
[ -f "$D/metadata.json" ] || { echo "falta $D/metadata.json (rode o run_gidcl.sh antes)" >&2; exit 1; }

for v in matrix:model functions:funcs union:union; do
  f=${v%%:*}; tag=${v##*:}; M="gidcl_${tag}_s${S}"
  [ -f "$D/pred_$f.npy" ] || continue
  # a variante herda o metadata da execução (mesmo tempo e tokens), com a variante na config
  if ! "$PY" - "$DIRTY" "$D/pred_$f.npy" "outputs/$NAME/$M" "$D/metadata.json" "$M" "$tag" <<'PYEOF'
import json, os, sys
import numpy as np, pandas as pd
dirty, npy, out, meta_src, method, tag = sys.argv[1:7]
m = np.load(npy)
print(f"{os.path.basename(npy)}: células marcadas = {int(m.sum())} de {m.size}")
if m.sum() == 0:
    sys.exit(3)
os.makedirs(out, exist_ok=True)
pd.DataFrame(m, columns=pd.read_csv(dirty, nrows=0).columns).to_csv(out + "/pred.csv", index=False)
meta = json.load(open(meta_src))
meta["method"] = method
meta["config"] = {**meta["config"], "variant": tag}
json.dump(meta, open(out + "/metadata.json", "w"), indent=1, ensure_ascii=False)
PYEOF
  then echo "  (sem células marcadas: variante ignorada)"; continue; fi
  evaluate "$PY" "outputs/$NAME/$M" "$GT"
  evaluate "$PY" "outputs/$NAME/$M" "$GT" --method "${M}_heldout" --exclude-rows "$SPLIT/labeled_rows.csv"
done
