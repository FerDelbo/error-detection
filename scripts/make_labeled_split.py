#!/usr/bin/env python3
"""Separa o que o GIDCL pode ver: tabela suja, tuplas rotuladas e clusters.
A escolha das tuplas usa SÓ a tabela suja; o clean só é lido nas tuplas escolhidas."""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dirty", required=True)
ap.add_argument("--clean", required=True)
ap.add_argument("--k", type=int, default=20, help="tuplas rotuladas (artigo: 20)")
ap.add_argument("--clusters", type=int, default=None, help="nº de clusters (padrão: k; artigo: Hospital 20, Beers 10, Tax 30)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--select", choices=["cluster", "random"], default="cluster")
ap.add_argument("--embed", choices=["sbert", "tfidf"], default="sbert")
ap.add_argument("--embed-model", default="BAAI/bge-large-en")
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
dirty = load(args.dirty)
n = len(dirty)
k = min(args.k, n)
ncl = max(1, min(args.clusters or k, n))
rng = np.random.RandomState(args.seed)
docs = [" | ".join(f"{c}: {v}" for c, v in zip(dirty.columns, row)) for row in dirty.values.tolist()]


def embed(docs):
    if args.embed == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), sublinear_tf=True).fit_transform(docs)
        E = TruncatedSVD(n_components=max(2, min(128, X.shape[1] - 1)), random_state=args.seed).fit_transform(X)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embed_model, device="cuda" if torch.cuda.is_available() else "cpu")
        E = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def select_cluster(E, labels, k):
    """Clusters de menor similaridade média primeiro; em cada um, a tupla mais atípica."""
    stats = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        sz = len(idx)
        s = E[idx].sum(axis=0)
        if sz > 1:
            avg = float((s @ s - sz) / (sz * (sz - 1)))        # similaridade média entre pares
            rowmean = (E[idx] @ s - 1.0) / (sz - 1)            # similaridade média de cada tupla com as outras
        else:
            avg, rowmean = 1.0, np.array([1.0])
        stats.append((avg, idx, np.argsort(rowmean)))
    stats.sort(key=lambda t: t[0])
    chosen = []
    for rank in range(int(np.ceil(k / len(stats)))):          # se há menos clusters que k, mais de uma por cluster
        for _, idx, order in stats:
            if len(chosen) < k and rank < len(order):
                chosen.append(int(idx[order[rank]]))
    if len(chosen) < k:
        used = set(chosen)
        rest = [i for i in range(len(E)) if i not in used]
        chosen += rng.choice(rest, size=k - len(chosen), replace=False).tolist()
    return sorted(chosen)


E = embed(docs)
if ncl > 1:
    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=ncl, n_init=10, random_state=args.seed).fit_predict(E)
else:
    labels = np.zeros(n, dtype=int)
rows = sorted(rng.choice(n, size=k, replace=False).tolist()) if args.select == "random" else select_cluster(E, labels, k)

# --- só agora o clean é lido, e apenas nas tuplas escolhidas ---
clean = load(args.clean)
assert clean.shape == dirty.shape, f"tamanhos diferentes: {dirty.shape} vs {clean.shape}"
clean.columns = dirty.columns
os.makedirs(args.outdir, exist_ok=True)
masked = dirty.copy()
masked.loc[rows] = clean.loc[rows]
dirty.to_csv(os.path.join(args.outdir, "dirty.csv"), index=False)
masked.to_csv(os.path.join(args.outdir, "clean_masked.csv"), index=False)
pd.DataFrame({"row": rows}).to_csv(os.path.join(args.outdir, "labeled_rows.csv"), index=False)
pd.DataFrame({"row": range(n), "cluster": labels}).to_csv(os.path.join(args.outdir, "clusters.csv"), index=False)
cells = [(i, c, dirty.at[i, c], clean.at[i, c]) for i in rows for c in dirty.columns]
pd.DataFrame(cells, columns=["row", "column", "dirty", "clean"]).to_csv(
    os.path.join(args.outdir, "labeled_cells.csv"), index=False)
n_err = sum(d != c for _, _, d, c in cells)
json.dump({"k": k, "clusters": ncl, "seed": args.seed, "select": args.select, "embed": args.embed, "rows": rows},
          open(os.path.join(args.outdir, "split_info.json"), "w"))
print(f"{k} tuplas rotuladas, {ncl} clusters -> {args.outdir}  (células erradas nelas: {n_err} de {len(cells)})")
print("Não troque a semente por causa desse número: isso vazaria informação da avaliação.")