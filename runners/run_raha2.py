#!/usr/bin/env python3
"""
Runner genérico para o método Raha (Mahdavi et al., SIGMOD 2019).
Recebe --dirty e --clean de qualquer caminho no sistema de arquivos,
isola artefatos temporários em diretório temporário para não poluir a pasta da base
e salva a matriz binária 0/1 em outputs/<name>/raha_s<seed>/pred.csv.
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAHA_DIR = os.path.join(ROOT, "methods", "raha-master")
sys.path.insert(0, RAHA_DIR)
sys.path.insert(0, os.path.join(RAHA_DIR, "raha"))

try:
    from detection import Detection
except ImportError:
    from raha.detection import Detection


def main():
    ap = argparse.ArgumentParser(description="Raha (detecção na tabela completa) -> pred.csv")
    ap.add_argument("--name", required=True, help="Rótulo do dataset (ex.: test, hospital)")
    ap.add_argument("--dirty", required=True, help="Caminho para o arquivo dirty.csv (qualquer local)")
    ap.add_argument("--clean", required=True, help="Caminho para o arquivo clean.csv (qualquer local)")
    ap.add_argument("--budget", type=int, default=2, help="Orçamento de tuplas rotuladas (padrão: 2)")
    ap.add_argument("--seed", type=int, default=0, help="Semente aleatória (padrão: 0)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"), help="Diretório raiz de saída")
    args = ap.parse_args()

    dirty_src = os.path.abspath(os.path.expanduser(args.dirty))
    clean_src = os.path.abspath(os.path.expanduser(args.clean))
    for p in (dirty_src, clean_src):
        if not os.path.isfile(p):
            sys.exit(f"[raha] ERRO: arquivo não encontrado: {p}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    dirty = pd.read_csv(dirty_src, dtype=str, keep_default_na=False, low_memory=False)
    clean = pd.read_csv(clean_src, dtype=str, keep_default_na=False, low_memory=False)
    assert dirty.shape == clean.shape, (
        f"[raha] Tamanhos incompatíveis: dirty={dirty.shape} vs clean={clean.shape}"
    )

    # Isola os CSVs em um diretório temporário para evitar que o Raha crie pastas
    # de cache/resultados dentro do diretório original da base de dados.
    tmp_dir = tempfile.mkdtemp(prefix=f"raha_{args.name}_")
    try:
        dirty_tmp = os.path.join(tmp_dir, "dirty.csv")
        clean_tmp = os.path.join(tmp_dir, "clean.csv")
        dirty.to_csv(dirty_tmp, index=False, encoding="utf-8")
        clean.to_csv(clean_tmp, index=False, encoding="utf-8")

        app = Detection()
        app.LABELING_BUDGET = args.budget
        app.USER_LABELING_ACCURACY = 1.0
        app.SAVE_RESULTS = False
        app.VERBOSE = False

        t0 = time.time()
        detection_dictionary = app.run({
            "name": args.name,
            "path": dirty_tmp,
            "clean_path": clean_tmp,
        })
        elapsed = time.time() - t0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mat = np.zeros(dirty.shape, dtype=int)
    for (i, j) in detection_dictionary.keys():
        if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
            mat[i, j] = 1

    tag = f"raha_s{args.seed}"
    out = os.path.join(args.outdir, args.name, tag)
    os.makedirs(out, exist_ok=True)
    out_csv = os.path.join(out, "pred.csv")
    pd.DataFrame(mat, columns=dirty.columns).to_csv(out_csv, index=False)
    print(
        f"[raha] {args.name} budget={args.budget} seed={args.seed} "
        f"detectadas={int(mat.sum())} tempo={elapsed:.1f}s -> {out_csv}"
    )


if __name__ == "__main__":
    main()