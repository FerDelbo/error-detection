import argparse, os, random, sys, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "methods", "raha-master"))
from detection import Detection

ap = argparse.ArgumentParser(description="Raha (detecção) -> pred.csv")
ap.add_argument("--name", required=True, help="rótulo do dataset (usado só nos nomes de saída)")
ap.add_argument("--dirty", required=True, help="caminho do CSV sujo")
ap.add_argument("--clean", required=True, help="caminho do CSV limpo (usado para simular o rotulador)")
ap.add_argument("--budget", type=int, default=2, help="tuplas rotuladas (artigo: 2)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
args = ap.parse_args()

dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
for p in (dirty_path, clean_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")

random.seed(args.seed)
np.random.seed(args.seed)

app = Detection()
app.LABELING_BUDGET = args.budget
app.USER_LABELING_ACCURACY = 1.0
app.SAVE_RESULTS = False
app.VERBOSE = False

dd = {"name": args.name, "path": dirty_path, "clean_path": clean_path}
t0 = time.time()
detected = app.run(dd)
elapsed = time.time() - t0

print("exemplo de chaves:", list(detected)[:3])
dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
mat = np.zeros(dirty.shape, dtype=int)
for (i, j) in detected:
    mat[i, j] = 1

out = os.path.join(args.outdir, args.name, f"raha_s{args.seed}")
os.makedirs(out, exist_ok=True)
pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
print(f"[raha] {args.name} seed={args.seed} budget={args.budget} "
      f"detectadas={int(mat.sum())} tempo={elapsed:.1f}s -> {out}/pred.csv")