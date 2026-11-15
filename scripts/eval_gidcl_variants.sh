#!/usr/bin/env bash
# Avalia as variantes de saída de uma execução do GIDCL sem rodar de novo:
#   model = M_det (a oficial, igual ao pred.csv), funcs = só as funções F_det, union = M_det OU F_det
# uso: [RESULTS=results.csv] [HELDOUT=1] bash scripts/eval_gidcl_variants.sh <pasta da execução> <dirty.csv> <clean.csv>
#   ex.: bash scripts/eval_gidcl_variants.sh outputs/hospital/gidcl_b20_k20_s0 dataset/hospital/dirty.csv dataset/hospital/clean.csv
set -eo pipefail
cd "$(dirname "$0")/.."
source scripts/common.sh
D=${1:?uso: eval_gidcl_variants.sh <pasta da execução> <dirty.csv> <clean.csv>}; DIRTY=${2:?}; CLEAN=${3:?}
D=${D%/}
PY=${GIDCL_PY:-$HOME/miniconda3/envs/gidcl/bin/python}
[ -x "$PY" ] || PY=python
GT=$(ensure_gt "$DIRTY" "$CLEAN" "$PY")
[ -f "$D/metadata.json" ] || { echo "falta $D/metadata.json (rode o run_gidcl.sh antes)" >&2; exit 1; }

for v in matrix:model functions:funcs union:union; do
  f=${v%%:*}; tag=${v##*:}; M="$(basename "$D")_${tag}"; O="$(dirname "$D")/$M"
  [ -f "$D/pred_$f.npy" ] || continue
  # a variante herda o metadata da execução (mesmo tempo e tokens), com a variante na config
  if ! "$PY" - "$DIRTY" "$D/pred_$f.npy" "$O" "$D/metadata.json" "$M" "$tag" <<'PYEOF'
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
  evaluate "$PY" "$O" "$GT"
  if [ -n "${HELDOUT:-}" ]; then
    evaluate "$PY" "$O" "$GT" --method "${M}_heldout" --exclude-rows "$D/labeled_rows.csv"
  fi
done
