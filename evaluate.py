import argparse, json, os, sys
import pandas as pd

ap = argparse.ArgumentParser(description="Precisão/recall/F1 de detecção de erros")
ap.add_argument("--name", required=True, help="rótulo do dataset")
ap.add_argument("--method", required=True, help="rótulo do método (ex.: raha_s0)")
ap.add_argument("--dirty", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--pred", help="matriz 0/1; padrão: outputs/<name>/<method>/pred.csv")
ap.add_argument("--results", default="results.jsonl")
args = ap.parse_args()

pred_path = args.pred or os.path.join("outputs", args.name, args.method, "pred.csv")
for p in (args.dirty, args.clean, pred_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")

def load(p): return pd.read_csv(p, dtype=str, keep_default_na=False)
clean, dirty = load(args.clean), load(args.dirty)
pred = pd.read_csv(pred_path, dtype=int)
assert clean.shape == dirty.shape == pred.shape, \
    f"tamanhos diferentes: clean{clean.shape} dirty{dirty.shape} pred{pred.shape}"

y, p = (dirty.values != clean.values), pred.values.astype(bool)
tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

row = dict(dataset=args.name, method=args.method, precision=round(prec, 3),
           recall=round(rec, 3), f1=round(f1, 3), tp=int(tp), fp=int(fp), fn=int(fn),
           erros_reais=int(y.sum()))
print(row)
with open(args.results, "a") as f:
    f.write(json.dumps(row) + "\n")