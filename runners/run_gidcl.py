#!/usr/bin/env python3
"""
GIDCL (detecção) no formato comum do pipeline -> outputs/<name>/gidcl_s<seed>/pred.csv

Modos:
  --from-file F  : converte um arquivo de detecção pronto (ex.: GEIL_Data/<ds>/detector/detector.npy)
  --cmd + --output: roda um comando (ex.: notebook convertido em script) e converte a saída
Formatos aceitos: .npy/.npz (matriz n x m, matriz n x colunas_mantidas ou vetor 1-D),
                  .csv/.json/.pkl (matriz 0/1 ou lista de células).
"""
import argparse
import glob
import json
import os
import pickle
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", required=True)
ap.add_argument("--dirty", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--from-file", help="arquivo de detecção já pronto")
ap.add_argument("--cmd", action="append",
                help="comando de uma etapa (sem o python); repita para várias etapas, em ordem")
ap.add_argument("--split-dir", help="pasta criada por scripts/make_labeled_split.py (modo sem vazamento)")
ap.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES das etapas (ex.: 0)")
ap.add_argument("--output", help="arquivo gerado pelo comando (glob; relativo a --gidcl-dir)")
ap.add_argument("--gidcl-dir", default=os.path.join(ROOT, "methods", "GIDCL"))
ap.add_argument("--python", default=sys.executable, help="python do ambiente do GIDCL (modo --cmd)")
ap.add_argument("--labeled", type=int, default=20, help="tuplas rotuladas (artigo: 20)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-cols", type=int, nargs="*", default=[],
                help="colunas (0-based) ausentes do vetor, ex.: 0 1 no Beers (index e id)")
ap.add_argument("--order", choices=["C", "F", "auto"], default="C",
                help="leitura do vetor 1-D: C = linha a linha, F = coluna a coluna, "
                     "auto = escolhe a de maior F1 (usa o clean só para decidir o layout)")
ap.add_argument("--row-offset", type=int, default=0, help="para listas de células com linhas 1-based use 1")
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
args = ap.parse_args()

if not args.from_file and not (args.cmd and args.output):
    sys.exit("use --from-file ARQUIVO, ou --cmd e --output")

load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
dirty, clean = load(dirty_path), load(clean_path)
assert dirty.shape == clean.shape, f"dirty {dirty.shape} e clean {clean.shape} com tamanhos diferentes"
clean.columns = dirty.columns
cols = list(dirty.columns)
n, m = dirty.shape
truth = dirty.values != clean.values
keep = [j for j in range(m) if j not in set(args.skip_cols)]

out = os.path.join(args.outdir, args.name, f"gidcl_s{args.seed}")
os.makedirs(out, exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
fmt = dict(name=args.name, dirty=dirty_path, clean=clean_path, labeled=args.labeled, seed=args.seed, outdir=out)

if args.split_dir:
    if args.order == "auto":
        sys.exit("--order auto usa o clean; em modo sem vazamento use --order C")
    sp = os.path.abspath(args.split_dir)
    fmt.update(dirty=os.path.join(sp, "dirty.csv"),
               clean=os.path.join(sp, "clean_masked.csv"),     # o método nunca vê o clean verdadeiro
               labeled_csv=os.path.join(sp, "labeled_cells.csv"),
               labeled_rows=os.path.join(sp, "labeled_rows.csv"), split=sp)

# ------------------------------------------------------------ obter o arquivo de detecção
elapsed = 0.0
if args.from_file:
    det_file = os.path.abspath(args.from_file)
    if not os.path.isfile(det_file):
        sys.exit(f"arquivo não encontrado: {det_file}")
    cmd = None
# else:
#     gdir = os.path.abspath(args.gidcl_dir)
#     if not os.path.isdir(gdir):
#         sys.exit(f"pasta não encontrada: {gdir}")
#     before = time.time()
#     cmd = [args.python] + shlex.split(args.cmd.format(**fmt))
#     log_path = os.path.join(ROOT, "logs", f"gidcl_{args.name}.log")
#     print("$", " ".join(cmd), flush=True)
#     with open(log_path, "a", encoding="utf-8") as lf:
#         p = subprocess.Popen(cmd, cwd=gdir, env=dict(os.environ, PYTHONUNBUFFERED="1"),
#                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
#                              text=True, encoding="utf-8", errors="replace")
#         for line in p.stdout:
#             sys.stdout.write(line)
#             lf.write(line)
#         rc = p.wait()
#     elapsed = time.time() - before
#     if rc != 0:
#         sys.exit(f"o comando terminou com código {rc}; veja {log_path}")
#     pattern = args.output.format(**fmt)
#     pattern = pattern if os.path.isabs(pattern) else os.path.join(gdir, pattern)
#     hits = [h for h in glob.glob(pattern, recursive=True) if os.path.getmtime(h) >= before - 1]
#     if not hits:
#         sys.exit(f"nenhum arquivo novo para o padrão: {pattern}")
#     det_file = max(hits, key=os.path.getmtime)
# print("arquivo de detecção:", det_file)
else:
    gdir = os.path.abspath(args.gidcl_dir)
    if not os.path.isdir(gdir):
        sys.exit(f"pasta não encontrada: {gdir}")
    before = time.time()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_path = os.path.join(ROOT, "logs", f"gidcl_{args.name}.log")
    cmds = [[args.python] + shlex.split(c.format(**fmt)) for c in args.cmd]
    for c in cmds:
        print("$", " ".join(c), flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("$ " + " ".join(c) + "\n")
            p = subprocess.Popen(c, cwd=gdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                sys.stdout.write(line)
                lf.write(line)
            rc = p.wait()
        if rc != 0:
            sys.exit(f"etapa falhou (código {rc}): {' '.join(c)}\nlog: {log_path}")
    elapsed = time.time() - before
    cmd = [" ; ".join(" ".join(c) for c in cmds)]
    pattern = args.output.format(**fmt)
    pattern = pattern if os.path.isabs(pattern) else os.path.join(gdir, pattern)
    hits = [h for h in glob.glob(pattern, recursive=True) if os.path.getmtime(h) >= before - 1]
    if not hits:
        sys.exit(f"nenhum arquivo novo para o padrão: {pattern}")
    det_file = max(hits, key=os.path.getmtime)


# ------------------------------------------------------------ conversão para matriz 0/1
def f1_of(mat):
    p = mat.astype(bool)
    tp, fp, fn = (truth & p).sum(), (~truth & p).sum(), (truth & ~p).sum()
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def place(sub):
    mat = np.zeros((n, m), dtype=int)
    mat[:, keep] = (sub.astype(float) >= 0.5).astype(int)
    return mat


def array_to_matrix(arr):
    arr = np.asarray(arr)
    print("formato do array:", arr.shape, "| valores distintos:", np.unique(arr)[:8])
    if arr.shape == (n, m):
        return (arr.astype(float) >= 0.5).astype(int)
    if arr.ndim == 2 and arr.shape == (n, len(keep)):
        return place(arr)
    if arr.ndim == 1 and arr.size == n * len(keep):
        orders = ["C", "F"] if args.order == "auto" else [args.order]
        cands = {o: place(arr.reshape((n, len(keep)), order=o)) for o in orders}
        scores = {o: f1_of(mat) for o, mat in cands.items()}
        print("F1 por ordem de leitura:", {o: round(s, 3) for o, s in scores.items()})
        return cands[max(scores, key=scores.get)]
    sys.exit(f"o array tem forma {arr.shape} (tamanho {arr.size}); esperado {n} linhas x {len(keep)} colunas "
             f"= {n * len(keep)} (skip-cols={args.skip_cols}). Ajuste --skip-cols.")


def cells_to_matrix(obj):
    mat = np.zeros((n, m), dtype=int)
    stats = {"fora": 0}

    def mark(i, j):
        try:
            j = cols.index(j) if isinstance(j, str) and j in cols else int(j)
            i = int(i) - args.row_offset
        except (TypeError, ValueError):
            stats["fora"] += 1
            return
        if 0 <= i < n and 0 <= j < m:
            mat[i, j] = 1
        else:
            stats["fora"] += 1

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, (tuple, list)) and len(k) == 2:
                mark(*k)
            elif isinstance(v, (list, tuple, set)):
                for i in v:
                    mark(i, k)
            elif isinstance(k, str) and "," in k:
                mark(*[x.strip() for x in k.split(",", 1)])
    elif isinstance(obj, (list, tuple, set)):
        for c in obj:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                mark(c[0], c[1])
            elif isinstance(c, dict):
                i = next((c[k] for k in ("row", "row_idx", "i", "index", "tuple") if k in c), None)
                j = next((c[k] for k in ("col", "column", "attr", "attribute", "j") if k in c), None)
                if i is not None and j is not None:
                    mark(i, j)
    return mat, stats["fora"]


def convert(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return array_to_matrix(np.load(path)), 0
    if ext == ".npz":
        z = np.load(path)
        return array_to_matrix(z["predictions"] if "predictions" in z.files else z[z.files[0]]), 0
    if ext in (".pkl", ".pickle"):
        obj = pickle.load(open(path, "rb"))              # arquivo gerado por você mesmo
        if isinstance(obj, (np.ndarray, pd.DataFrame)):
            return array_to_matrix(obj.values if isinstance(obj, pd.DataFrame) else obj), 0
        return cells_to_matrix(obj)
    if ext == ".json":
        return cells_to_matrix(json.load(open(path, encoding="utf-8")))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    vals = set(df.values.ravel().tolist())
    if vals <= {"0", "1", "True", "False", "0.0", "1.0", ""} and df.shape in ((n, m), (n, len(keep))):
        return array_to_matrix(df.apply(lambda s: s.isin(["1", "True", "1.0"])).values.astype(int)), 0
    low = {c.lower(): c for c in df.columns}
    rc = next((low[k] for k in ("row", "row_idx", "i", "index", "tuple", "row_id") if k in low), None)
    cc = next((low[k] for k in ("col", "column", "attr", "attribute", "j", "col_idx") if k in low), None)
    pairs = list(zip(df[rc], df[cc])) if rc and cc else (list(zip(df.iloc[:, 0], df.iloc[:, 1])) if df.shape[1] == 2 else [])
    return cells_to_matrix(pairs)


mat, fora = convert(det_file)
marked = int(mat.sum())
if marked == 0:
    sys.exit(f"nenhuma célula marcada em {det_file} (fora_dos_limites={fora}). "
             f"Confira --skip-cols/--order/--row-offset ou me envie o formato do arquivo.")

pd.DataFrame(mat, columns=cols).to_csv(os.path.join(out, "pred.csv"), index=False)
json.dump({"total_s": round(elapsed, 1), "modo": "arquivo" if args.from_file else "comando",
           "origem": det_file, "cmd": " ".join(cmd) if cmd else None, "labeled": args.labeled,
           "skip_cols": args.skip_cols, "order": args.order},
          open(os.path.join(out, "runtime.json"), "w"), indent=1)
print(f"\n[gidcl] {args.name} marcadas={marked} de {mat.size} fora_dos_limites={fora} -> {out}/pred.csv")