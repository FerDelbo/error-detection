#!/usr/bin/env python3
"""
ForestED (detecção de erros) no formato comum do pipeline.

Etapas: data_catalog.py -> gp_sampling.py -> decision_forest.py -> conversão para pred.csv
Saída:  outputs/<name>/forested_s<run-id>/pred.csv  (matriz 0/1 do tamanho do dirty)
        outputs/<name>/forested_s<run-id>/trees/tree_<k>.csv  (uma matriz por árvore, se existirem)
        outputs/<name>/forested_s<run-id>/raw/  (saída original do decision_forest.py)

Execute com o Python do ambiente conda do ForestED (forested).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", required=True, help="rótulo do dataset (nomes de saída)")
ap.add_argument("--dirty", required=True, help="CSV sujo (tabela completa)")
ap.add_argument("--clean", required=True, help="CSV limpo (tabela completa)")
ap.add_argument("--run-id", default="1", help="identifica a repetição (o LLM é estocástico)")
ap.add_argument("--forested-dir", default=os.path.join(ROOT, "methods", "ForestED"))
ap.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL"))
ap.add_argument("--frac", default="0.05", help="fração de linhas amostradas (artigo: 5%%; o código limita a 100)")
ap.add_argument("--batch-size", default="10", help="tuplas por partição/árvore (artigo: 10)")
ap.add_argument("--max-workers", default="2", help="threads de chamadas ao LLM")
ap.add_argument("--voting", default="dawid_skene", choices=["majority", "unanimous", "dawid_skene"])
ap.add_argument("--pred-model", default="gnn", choices=["gnn", "mlp"])
ap.add_argument("--key-env", default="OPENAI_API_KEY",
                help="variável de ambiente com a chave que o SEU decision_forest.py lê")
ap.add_argument("--refresh-catalog", action="store_true", help="recria o catálogo mesmo se já existir")
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
args = ap.parse_args()

FDIR = os.path.abspath(args.forested_dir)
dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
for p in (dirty_path, clean_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")
for s in ("data_catalog.py", "gp_sampling.py", "decision_forest.py"):
    if not os.path.isfile(os.path.join(FDIR, s)):
        sys.exit(f"não achei {s} em {FDIR} (use --forested-dir)")
if not os.getenv(args.key_env):
    sys.exit(f"variável {args.key_env} vazia. Defina-a no .env ou ajuste --key-env para o nome "
             f"que o seu decision_forest.py lê.")

dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False, low_memory=False)
clean_cols = pd.read_csv(clean_path, nrows=0).columns
assert dirty.shape[1] == len(clean_cols), "dirty e clean com números de colunas diferentes"

work = os.path.join(args.outdir, args.name, "forested_work")
run_dir = os.path.join(args.outdir, args.name, f"forested_s{args.run_id}")
raw = os.path.join(run_dir, "raw")
os.makedirs(work, exist_ok=True)
shutil.rmtree(raw, ignore_errors=True)   # nunca misturar com uma execução anterior
os.makedirs(raw, exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
log_path = os.path.join(ROOT, "logs", f"forested_{args.name}.log")

env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd=FDIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()
    if rc != 0:
        sys.exit(f"falhou (código {rc}): {' '.join(cmd)}\nlog: {log_path}")


t0 = time.time()

# 1) catálogo (uma vez por dataset)
catalog = os.path.join(work, "catalog.json")
if args.refresh_catalog or not os.path.exists(catalog):
    run([sys.executable, "data_catalog.py", dirty_path, "--out", catalog])
else:
    print(f"catálogo já existe: {catalog}")

# 2) amostragem com GP (a cada execução)
sample = os.path.join(work, f"sample_{args.run_id}.csv")
run([sys.executable, "gp_sampling.py", dirty_path, "--out", sample, "--frac", str(args.frac)])

# 3) floresta de árvores
run([sys.executable, "decision_forest.py",
     "--clean", clean_path, "--dirty", dirty_path, "--sample", sample, "--catalog", catalog,
     "--outdir", raw, "--llm_model", args.llm_model, "--batch_size", str(args.batch_size),
     "--max_workers", str(args.max_workers), "--voting", args.voting, "--pred_model", args.pred_model])
elapsed = time.time() - t0


# 4) exportar para o formato comum
def to_matrix(path):
    z = np.load(path)
    key = "predictions" if "predictions" in z.files else z.files[0]
    m = np.asarray(z[key])
    if m.shape != dirty.shape:
        sys.exit(f"{os.path.basename(path)} (chave '{key}') tem forma {m.shape}, esperado {dirty.shape}")
    return (m.astype(float) >= 0.5).astype(int)


npzs = sorted(glob.glob(os.path.join(raw, "**", "*.npz"), recursive=True))
cons = [p for p in npzs if "consensus" in os.path.basename(p).lower()]
if not cons:
    print("\nNenhum arquivo *consensus*.npz encontrado. Arquivos .npz na saída:")
    for p in npzs:
        print("  ", os.path.relpath(p, raw))
    sys.exit("Ajuste a busca em 'cons' (ou me envie esta lista).")
cons_path = max(cons, key=os.path.getmtime)

mat = to_matrix(cons_path)
os.makedirs(run_dir, exist_ok=True)
pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(run_dir, "pred.csv"), index=False)

# matrizes por árvore (para TreeED): arquivos .npz dentro de partition_*/ que não são o consenso
trees = [p for p in npzs if p not in cons and "partition" in os.path.relpath(p, raw).lower()]
if trees:
    tdir = os.path.join(run_dir, "trees")
    shutil.rmtree(tdir, ignore_errors=True)
    os.makedirs(tdir)
    for k, p in enumerate(trees):
        pd.DataFrame(to_matrix(p), columns=dirty.columns).to_csv(os.path.join(tdir, f"tree_{k}.csv"), index=False)

partitions = []
for mp in sorted(glob.glob(os.path.join(raw, "**", "metrics.json"), recursive=True)):
    try:
        partitions.append({"file": os.path.relpath(mp, raw), **json.load(open(mp))})
    except Exception:
        pass
json.dump({"total_s": round(elapsed, 1), "llm_model": args.llm_model, "voting": args.voting,
           "pred_model": args.pred_model, "frac": args.frac, "batch_size": args.batch_size,
           "consenso": os.path.relpath(cons_path, raw), "n_arquivos_de_arvore": len(trees),
           "partitions": partitions}, open(os.path.join(run_dir, "runtime.json"), "w"), indent=1)

print(f"\n[forested] {args.name} run={args.run_id} marcadas={int(mat.sum())} de {mat.size} "
      f"arvores_exportadas={len(trees)} tempo={elapsed:.0f}s\n   consenso: {os.path.relpath(cons_path, raw)}"
      f"\n   saída: {run_dir}/pred.csv")