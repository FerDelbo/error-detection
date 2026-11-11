"""Precisão/recall/F1 de detecção de erros a partir de pred.csv, GROUND_TRUTH.csv e metadata.json.

Acrescenta uma linha ao CSV de resultados (cria com cabeçalho se não existir).
O metadata.json é obrigatório (formato em runners/pipeline_meta.py): traz o tempo de execução,
a configuração do algoritmo e, se o método usa LLM, os tokens de entrada e saída.

Uso:
  python evaluate.py --pred outputs/beers/raha_b20_s0/pred.csv \
      --ground-truth data/dataset/beers/GROUND_TRUTH.csv \
      --results results.csv
  (--metadata padrão: metadata.json na pasta do pred)
"""
import argparse, json, os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "runners"))
from pipeline_meta import validate

ap = argparse.ArgumentParser(description="Precisão/recall/F1 de detecção de erros")
ap.add_argument("--pred", required=True, help="matriz 0/1 gerada pelo algoritmo")
ap.add_argument("--ground-truth", required=True, help="GROUND_TRUTH.csv (matriz 0/1)")
ap.add_argument("--results", required=True, help="CSV onde a linha de resultado é acrescentada")
ap.add_argument("--metadata", help="metadata.json do algoritmo (padrão: ao lado do pred)")
ap.add_argument("--method", help="sobrescreve o nome do método do metadata (ex.: sufixo _heldout)")
ap.add_argument("--exclude-rows", help="CSV com coluna 'row' (índices 0-based) a ignorar, ex.: tuplas rotuladas")
args = ap.parse_args()

meta_path = args.metadata or os.path.join(os.path.dirname(args.pred), "metadata.json")
for p in (args.pred, args.ground_truth, meta_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")
meta = json.load(open(meta_path, encoding="utf-8"))
try:
    validate(meta, meta_path)
except ValueError as e:
    sys.exit(str(e))

gt = pd.read_csv(args.ground_truth, dtype=int)
pred = pd.read_csv(args.pred, dtype=int)
if gt.shape != pred.shape:
    sys.exit(f"tamanhos diferentes: ground_truth{gt.shape} pred{pred.shape}")

y, p = gt.values.astype(bool), pred.values.astype(bool)
n_excl = 0
if args.exclude_rows:
    ex = pd.read_csv(args.exclude_rows)["row"].astype(int).values
    keep = np.ones(len(y), dtype=bool)
    keep[ex] = False
    y, p, n_excl = y[keep], p[keep], len(ex)

tp, fp, fn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum()
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

row = dict(dataset=meta["dataset"], method=args.method or meta["method"],
           config=json.dumps(meta["config"], ensure_ascii=False, sort_keys=True),
           precision=round(float(prec), 3), recall=round(float(rec), 3), f1=round(float(f1), 3),
           run_time_s=meta["run_time_s"], uses_llm=meta["uses_llm"],
           prompt_tokens=meta.get("prompt_tokens"), completion_tokens=meta.get("completion_tokens"),
        #    tp=int(tp), fp=int(fp), fn=int(fn), erros_reais=int(y.sum()), detectadas=int(p.sum()),
           total_tokens = int(meta.get("prompt_tokens")) + int(meta.get("completion_tokens")) if meta.get("prompt_tokens") and meta.get("completion_tokens") else None,
           excluidos=n_excl, pred=args.pred)
print(row) 
os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
pd.DataFrame([row]).to_csv(args.results, mode="a", index=False,
                           header=not os.path.isfile(args.results))
