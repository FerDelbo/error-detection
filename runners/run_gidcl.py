#!/usr/bin/env python3
"""
Runner genérico para o estágio de Detecção de Erros do método GIDCL / GEIL
(SICS-Fundamental-Research-Center/GIDCL) em methods/GIDCL.

Converte qualquer par (--dirty, --clean) no formato de 3 colunas (`text0`, `text1`, `label`)
exigido pelo `DittoDataset` em `methods/GIDCL/ditto/model.py`, onde:
  - `text0`: célula alvo serializada ("COL <col> VAL <val>")
  - `text1`: contexto de grafo/tupla das colunas correlacionadas ("COL <c1> VAL <v1> ...")
  - `label`: "1" (erro) ou "0" (limpo)
Treina o detector (`simple_train`) usando um orçamento de tuplas rotuladas + data augmentation
e executa `classify` sobre todas as N x M células da tabela para gerar
`outputs/<name>/gidcl_s<seed>/pred.csv`.
"""
import argparse
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIDCL_DIR = os.path.join(ROOT, "methods", "GIDCL")
if os.path.isdir(GIDCL_DIR) and GIDCL_DIR not in sys.path:
    sys.path.insert(0, GIDCL_DIR)


def _serialize_cell_and_graph_context(df: pd.DataFrame, r: int, c: int) -> tuple[str, str]:
    cols = list(df.columns)
    col_name = cols[c]
    cell_val = str(df.iat[r, c])
    text0 = f"COL {col_name} VAL {cell_val}"
    ctx = " ".join(f"COL {cols[j]} VAL {df.iat[r, j]}" for j in range(len(cols)) if j != c)
    return text0, ctx


def _build_ditto_dataframes(
    dirty: pd.DataFrame,
    clean: pd.DataFrame,
    budget_tuples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    n_rows, n_cols = dirty.shape
    cols = list(dirty.columns)

    # Todas as N x M células na ordem exata (linha 0..N-1, coluna 0..M-1) para inferência completa
    all_rows = []
    for r in range(n_rows):
        for c in range(n_cols):
            t0, t1 = _serialize_cell_and_graph_context(dirty, r, c)
            lbl = "1" if str(dirty.iat[r, c]) != str(clean.iat[r, c]) else "0"
            all_rows.append({"text0": t0, "text1": t1, "label": lbl})
    test_df_all = pd.DataFrame(all_rows)

    # Amostragem de tuplas rotuladas + data augmentation (como em function_set / detector_train)
    sampled_tuples = rng.sample(range(n_rows), min(budget_tuples, n_rows))
    train_records = []
    col_domains = [dirty.iloc[:, c].astype(str).tolist() for c in range(n_cols)]

    for r in sampled_tuples:
        for c in range(n_cols):
            t0, t1 = _serialize_cell_and_graph_context(dirty, r, c)
            lbl = "1" if str(dirty.iat[r, c]) != str(clean.iat[r, c]) else "0"
            train_records.append({"text0": t0, "text1": t1, "label": lbl})

            # Data augmentation sintética para balancear a classe minoritária (erro = "1")
            orig_val = str(dirty.iat[r, c])
            other_c = (c + rng.randint(1, max(1, n_cols - 1))) % n_cols
            noisy_val = rng.choice(col_domains[other_c])
            if noisy_val == orig_val and len(orig_val) >= 2:
                pos = rng.randint(0, len(orig_val) - 1)
                noisy_val = orig_val[:pos] + "x" + orig_val[pos + 1:]
            if noisy_val != orig_val:
                train_records.append({
                    "text0": f"COL {cols[c]} VAL {noisy_val}",
                    "text1": t1,
                    "label": "1",
                })

    rng.shuffle(train_records)
    train_df = pd.DataFrame(train_records)
    val_size = max(1, len(train_df) // 5)
    valid_df = train_df.iloc[:val_size].reset_index(drop=True)
    return train_df, valid_df, test_df_all


def main():
    ap = argparse.ArgumentParser(
        description="GIDCL / GEIL Detector (Ditto/RoBERTa) -> outputs/<name>/gidcl_s<seed>/pred.csv"
    )
    ap.add_argument("--name", required=True, help="Rótulo do dataset (ex.: test)")
    ap.add_argument("--dirty", required=True, help="Caminho para o arquivo dirty.csv (qualquer local)")
    ap.add_argument("--clean", required=True, help="Caminho para o arquivo clean.csv (qualquer local)")
    ap.add_argument("--budget", type=int, default=10, help="Tuplas amostradas para treino do detector (padrão: 10)")
    ap.add_argument("--epochs", type=int, default=10, help="Épocas de treino do detector Ditto (padrão: 10)")
    ap.add_argument("--batch-size", type=int, default=64, help="Tamanho de batch (padrão: 64)")
    ap.add_argument("--max-len", type=int, default=128, help="Comprimento máximo de sequência (padrão: 128)")
    ap.add_argument("--lm", default=os.environ.get("GIDCL_LM", "roberta-base"), help="Modelo base (padrão: roberta-base)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Limiar de classificação (padrão: 0.5)")
    ap.add_argument("--seed", type=int, default=0, help="Semente aleatória (padrão: 0)")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    dirty_src = os.path.abspath(os.path.expanduser(args.dirty))
    clean_src = os.path.abspath(os.path.expanduser(args.clean))
    for p in (dirty_src, clean_src):
        if not os.path.isfile(p):
            sys.exit(f"[gidcl] ERRO: arquivo não encontrado: {p}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    dirty = pd.read_csv(dirty_src, dtype=str, keep_default_na=False)
    clean = pd.read_csv(clean_src, dtype=str, keep_default_na=False)
    assert dirty.shape == clean.shape, f"Tamanhos diferentes: {dirty.shape} vs {clean.shape}"

    out_dir = os.path.join(args.outdir, args.name, f"gidcl_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    train_df, valid_df, test_df_all = _build_ditto_dataframes(
        dirty, clean, budget_tuples=args.budget, seed=args.seed
    )

    t0 = time.time()
    from ditto.model import DittoDataset, classify, simple_train
    import torch

    train_dataset = DittoDataset(train_df, max_len=args.max_len, lm=args.lm)
    valid_dataset = DittoDataset(valid_df, max_len=args.max_len, lm=args.lm)
    test_dataset_all = DittoDataset(test_df_all, max_len=args.max_len, lm=args.lm)

    hp_simple = SimpleNamespace(
        task=f"{args.name}-detector-train",
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=3e-5,
        n_epochs=args.epochs,
        save_model=False,
        logdir=os.path.join(out_dir, "detector_model"),
        lm=args.lm,
        fp16=bool(torch.cuda.is_available()),
        alpha_aug=0.8,
    )

    print(
        f"[gidcl] Treinando detector Ditto ({args.lm}, epochs={args.epochs}, "
        f"train_cells={len(train_df)}, total_cells={len(test_df_all)})..."
    )
    model_output = simple_train(train_dataset, valid_dataset, test_dataset_all, hp_simple)
    preds_1d = classify(
        test_dataset_all,
        model=model_output,
        lm=args.lm,
        max_len=args.max_len,
        threshold=args.threshold,
    )

    mask = np.asarray(preds_1d, dtype=int).reshape(dirty.shape)
    out_csv = os.path.join(out_dir, "pred.csv")
    pd.DataFrame(mask, columns=dirty.columns).to_csv(out_csv, index=False)
    elapsed = time.time() - t0
    print(
        f"[gidcl] {args.name} seed={args.seed} detectadas={int(mask.sum())} "
        f"tempo={elapsed:.1f}s -> {out_csv}"
    )


if __name__ == "__main__":
    main()