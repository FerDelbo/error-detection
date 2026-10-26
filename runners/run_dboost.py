import argparse, os, sys, tempfile, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "methods", "raha-master"))
from raha.tools.dBoost.dboost import imported_dboost

ap = argparse.ArgumentParser(description="dBoost (detecção) -> pred.csv")
ap.add_argument("--name", required=True, help="rótulo do dataset")
ap.add_argument("--dirty", required=True)
ap.add_argument("--config", nargs="+", required=True,
                help="ex.: gaussian 1.5   |   histogram 0.9 0.1")
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
args = ap.parse_args()

tag = "dboost_" + "_".join(args.config)
dirty = pd.read_csv(args.dirty, dtype=str, keep_default_na=False, low_memory=False)

tmp = tempfile.mkdtemp()
ds_path = os.path.join(tmp, f"{args.name}.csv")
dirty.to_csv(ds_path, sep=",", header=True, index=False, encoding="utf-8")

params = ["-F", ",", "--statistical", "0.5", "--" + args.config[0]] + args.config[1:] + [ds_path]
t0 = time.time()
try:
    imported_dboost.run(params)
except SystemExit:
    pass
elapsed = time.time() - t0

out_path = ds_path + "-dboost_output.csv"
mat = np.zeros(dirty.shape, dtype=int)
if os.path.exists(out_path):
    ocdf = pd.read_csv(out_path, sep=",", header=None, dtype=str, na_filter=False, skiprows=1)
    for row in ocdf.values.tolist():
        i, j = int(row[0]), int(row[1])
        if i > 0 and i - 1 < mat.shape[0] and j < mat.shape[1]:
            mat[i - 1, j] = 1
else:
    print(f"AVISO: dBoost não gerou saída. Arquivos em {tmp}: {os.listdir(tmp)}")

out = os.path.join(args.outdir, args.name, tag)
os.makedirs(out, exist_ok=True)
pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
print(f"[dboost] {args.name} config={' '.join(args.config)} detectadas={int(mat.sum())} "
      f"tempo={elapsed:.1f}s -> {out}/pred.csv")