#!/usr/bin/env python3
"""Cria a pasta 'visível' ao método: tabela suja + clean MASCARADO + k tuplas rotuladas.
A escolha das tuplas usa só a tabela suja (nunca o clean)."""
import argparse, json, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dirty", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--k", type=int, default=20, help="tuplas rotuladas (GIDCL: 20)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--select", choices=["random", "diverse"], default="diverse",
                help="random = sorteio; diverse = amostragem por distância máxima sobre o texto das linhas")
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
dirty, clean = load(args.dirty), load(args.clean)
assert dirty.shape == clean.shape, f"tamanhos diferentes: {dirty.shape} vs {clean.shape}"
clean.columns = dirty.columns
n, k = len(dirty), min(args.k, len(dirty))
rng = np.random.RandomState(args.seed)

if args.select == "random":
    rows = sorted(rng.choice(n, size=k, replace=False).tolist())
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    docs = dirty.astype(str).agg(" | ".join, axis=1)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2).fit_transform(docs)
    chosen = [int(rng.randint(n))]
    dist = cosine_distances(X, X[chosen[0]]).ravel()
    while len(chosen) < k:
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, cosine_distances(X, X[nxt]).ravel())
    rows = sorted(chosen)

os.makedirs(args.outdir, exist_ok=True)
masked = dirty.copy()
masked.loc[rows] = clean.loc[rows]                       # só as tuplas rotuladas têm valores limpos
dirty.to_csv(os.path.join(args.outdir, "dirty.csv"), index=False)
masked.to_csv(os.path.join(args.outdir, "clean_masked.csv"), index=False)
pd.DataFrame({"row": rows}).to_csv(os.path.join(args.outdir, "labeled_rows.csv"), index=False)

cells = [(i, c, dirty.at[i, c], clean.at[i, c]) for i in rows for c in dirty.columns]
pd.DataFrame(cells, columns=["row", "column", "dirty", "clean"]).to_csv(
    os.path.join(args.outdir, "labeled_cells.csv"), index=False)

n_err = sum(d != c for _, _, d, c in cells)
json.dump({"k": k, "seed": args.seed, "select": args.select, "rows": rows},
          open(os.path.join(args.outdir, "split_info.json"), "w"))
print(f"{k} tuplas rotuladas -> {args.outdir}  (células erradas nelas: {n_err} de {len(cells)})")
print("Não troque a semente por causa desse número: isso vazaria informação da avaliação.")