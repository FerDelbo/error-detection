"""Gera GROUND_TRUTH.csv: matriz 0/1 do tamanho do dirty, onde 1 = célula com erro.

Premissa: o clean.csv é a verdade. Toda célula cujo valor no dirty.csv difere do valor
na mesma posição (linha, coluna) do clean.csv é um erro localizado no dirty.

A comparação é literal, sobre o texto do CSV, e vale para qualquer tipo de valor
(números, textos, datas, vazios): os arquivos são lidos como string, sem inferir tipos
e sem converter vazios/"N/A" em NaN. Assim "64" vs "64.0", "abc" vs "abc " ou "" vs "N/A"
contam como erro. Nenhuma normalização é aplicada.

Uso:
  python scripts/make_ground_truth.py dataset/beers                  # um dataset
  python scripts/make_ground_truth.py                                # todos em dataset/
  python scripts/make_ground_truth.py --dirty d.csv --clean c.csv --out GT.csv
"""
import argparse, glob, os, sys
import pandas as pd


def load(p):
    # mesma leitura do evaluate.py: tudo como string, sem converter vazios em NaN
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def make_gt(dirty_path, clean_path, out_path):
    try:
        dirty, clean = load(dirty_path), load(clean_path)
    except Exception as e:
        print(f"[pulado] erro lendo CSV ({dirty_path} / {clean_path}): {e}")
        return False
    if dirty.shape != clean.shape:
        print(f"[pulado] tamanhos diferentes: dirty{dirty.shape} clean{clean.shape} ({dirty_path})")
        return False
    # comparação por posição (os nomes das colunas podem diferir entre dirty e clean)
    gt = pd.DataFrame((dirty.to_numpy() != clean.to_numpy()).astype(int), columns=dirty.columns)
    gt.to_csv(out_path, index=False)
    print(f"[ok] {out_path}: {gt.values.sum()} erros em {gt.size} células ({gt.values.mean():.2%})")
    return True


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dataset_dirs", nargs="*", help="pastas com clean.csv e dirty.csv")
ap.add_argument("--dirty"); ap.add_argument("--clean"); ap.add_argument("--out")
args = ap.parse_args()

if args.dirty and args.clean:
    ok = make_gt(args.dirty, args.clean,
                 args.out or os.path.join(os.path.dirname(args.dirty), "GROUND_TRUTH.csv"))
    sys.exit(0 if ok else 1)

dirs = args.dataset_dirs or sorted(glob.glob("dataset/*/"))
for d in dirs:
    dirty, clean = os.path.join(d, "dirty.csv"), os.path.join(d, "clean.csv")
    if os.path.isfile(dirty) and os.path.isfile(clean):
        make_gt(dirty, clean, os.path.join(d, "GROUND_TRUTH.csv"))
    else:
        print(f"[pulado] {d}: sem dirty.csv/clean.csv")
