#!/usr/bin/env python3
"""
GIDCL (detecção) no formato comum do pipeline:
  saída: outputs/<name>/gidcl_b<budget>_k<clusters>_s<seed>/{pred.csv, metadata.json, labeled_rows.csv, ...}

Protocolo do artigo (Yan et al., SIGMOD 2024, Seções 4 e 5), em duas etapas:
  1) scripts/make_labeled_split.py   escolhe as θ tuplas rotuladas (embedding + k-means; usa só a tabela suja)
  2) methods/gidcl_det/gidcl_detect.py   laço criador-crítico (LLM gera f_det/f_gen; RoBERTa é o M_det)
O detector nunca recebe o clean completo: só as células das θ tuplas escolhidas.
O tempo gravado no metadata.json cobre as duas etapas.

Parâmetros do artigo (Seção 7): θ=20 (--budget), τ=0.85 (--tau), M_det=RoBERTa (--lm), LLM=gpt-4-turbo
(OPENAI_MODEL no .env) e clusters por base: Hospital 20, Flights 20, Beers 10, Rayyan 10, Tax 30, IMDB 50
(--clusters; sem a opção, usa esse valor quando --name é uma dessas bases e o próprio budget nas demais).
Os demais (épocas, lr, rodadas, Iter_max, coreset...) não são informados no artigo; ver --help.

Modo --from-file F: só converte uma detecção pronta (ex.: methods/GIDCL/GEIL_Data/<ds>/detector/detector.npy,
publicada pelos autores) -> outputs/<name>/gidcl_released/. Formatos: .npy/.npz (matriz n x m, n x colunas
mantidas ou vetor 1-D), .csv/.json/.pkl (matriz 0/1 ou lista de células).
"""
import argparse
import json
import os
import pickle
import shutil
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
sys.path.insert(0, os.path.join(ROOT, "runners"))
from pipeline_meta import guard_output, write_metadata  # noqa: E402

# número de clusters usado no artigo (Seção 7, "Configuration")
PAPER_CLUSTERS = {"hospital": 20, "flights": 20, "beers": 10, "rayyan": 10, "tax": 30, "imdb": 50}
LLM_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--name", required=True, help="rótulo do dataset (usado só nos nomes de saída)")
ap.add_argument("--dirty", required=True, help="caminho do CSV sujo")
ap.add_argument("--clean", required=True, help="caminho do CSV limpo (rotulador simulado: só as θ tuplas são lidas)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
ap.add_argument("--overwrite", action="store_true",
                help="permite sobrescrever uma pasta de saída que tenha resultado de outra configuração")
ap.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES das etapas (ex.: 0)")

g = ap.add_argument_group("seleção das tuplas rotuladas (artigo, Seção 4)")
g.add_argument("--budget", type=int, default=20, help="θ: tuplas rotuladas (artigo: 20)")
g.add_argument("--clusters", type=int,
               help="k do k-means (artigo: Hospital 20, Flights 20, Beers 10, Rayyan 10, Tax 30, IMDB 50)")
g.add_argument("--select", choices=["cluster", "random"], default="cluster",
               help="cluster: clusters menos coesos, tupla mais atípica de cada um (artigo); random: ablação")
g.add_argument("--embed", choices=["sbert", "tfidf"], default="sbert",
               help="sbert: SentenceBert da linha (variante 'SBert' da Tabela 6; o GSL com GNN não foi publicado); "
                    "tfidf: sem GPU/modelo")
g.add_argument("--embed-model", default="BAAI/bge-large-en", help="modelo do SentenceBert (artigo: bge-large-en)")

g = ap.add_argument_group("criador: funções geradas pelo LLM (artigo, Seção 5)")
g.add_argument("--tau", type=float, default=0.85, help="τ: F1 mínimo em T_label para aceitar f_det (artigo: 0.85)")
g.add_argument("--iter-max", type=int, default=4, help="Iter_max: refinamentos por função (valor não informado no artigo)")
g.add_argument("--rounds", type=int, default=3,
               help="teto de rodadas criador-crítico (o artigo para quando F_det não muda; sem teto)")
g.add_argument("--pseudo-per-col", type=int, default=20, help="células corrompidas por f_gen, por coluna e rodada")

g = ap.add_argument_group("crítico: M_det (artigo, Seção 5)")
g.add_argument("--lm", default="roberta-base", help="PLM do M_det (artigo: RoBERTa)")
g.add_argument("--epochs", type=int, default=8, help="épocas por rodada (artigo: 'a few epochs'; notebook dos autores: 20)")
g.add_argument("--batch", type=int, default=16, help="batch de treino (notebook dos autores: 64)")
g.add_argument("--lr", type=float, default=2e-5, help="learning rate (notebook dos autores: 3e-5)")
g.add_argument("--max-len", type=int, default=256, help="tokens por instância (notebook dos autores: 128)")
g.add_argument("--infer-batch", type=int, default=128, help="batch de inferência (não altera o resultado)")
g.add_argument("--coreset-n", type=int, default=200, help="células limpas confiáveis adicionadas a T_coreset por rodada")
g.add_argument("--coreset-conf", type=float, default=0.9, help="confiança mínima do M_det para entrar em T_coreset")
g.add_argument("--col-ctx-max", type=int, default=20, help="valores distintos do cluster no contexto de coluna")

g = ap.add_argument_group("modo --from-file (detecção pronta)")
g.add_argument("--from-file", help="arquivo de detecção já pronto")
g.add_argument("--skip-cols", type=int, nargs="*", default=[],
               help="colunas (0-based) ausentes do vetor, ex.: 0 1 no Beers (index e id)")
g.add_argument("--order", choices=["C", "F", "auto"], default="C",
               help="leitura do vetor 1-D: C = linha a linha, F = coluna a coluna, "
                    "auto = escolhe a de maior F1 (usa o clean só para decidir o layout)")
g.add_argument("--row-offset", type=int, default=0, help="para listas de células com linhas 1-based use 1")
g.add_argument("--run-time", type=float, help="tempo (s) da execução original")
g.add_argument("--prompt-tokens", type=int, help="tokens de entrada da execução original")
g.add_argument("--completion-tokens", type=int, help="tokens de saída da execução original")
args = ap.parse_args()

dirty_path, clean_path = os.path.abspath(args.dirty), os.path.abspath(args.clean)
for p in (dirty_path, clean_path):
    if not os.path.isfile(p):
        sys.exit(f"arquivo não encontrado: {p}")
load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)  # noqa: E731
dirty = load(dirty_path)
cols = list(dirty.columns)
n, m = dirty.shape


# ------------------------------------------------------------ conversão de um arquivo pronto para matriz 0/1
def array_to_matrix(arr, keep, truth):
    def place(sub):
        mat = np.zeros((n, m), dtype=int)
        mat[:, keep] = (sub.astype(float) >= 0.5).astype(int)
        return mat

    def f1_of(mat):
        p = mat.astype(bool)
        tp, fp, fn = (truth & p).sum(), (~truth & p).sum(), (truth & ~p).sum()
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    arr = np.asarray(arr)
    print("formato do array:", arr.shape, "| valores distintos:", np.unique(arr)[:8])
    if arr.shape == (n, m):
        return (arr.astype(float) >= 0.5).astype(int)
    if arr.ndim == 2 and arr.shape == (n, len(keep)):
        return place(arr)
    if arr.ndim == 1 and arr.size == n * len(keep):
        orders = ["C", "F"] if args.order == "auto" else [args.order]
        cands = {o: place(arr.reshape((n, len(keep)), order=o)) for o in orders}
        if len(cands) == 1:
            return cands[orders[0]]
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


def convert(path, keep, truth):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return array_to_matrix(np.load(path), keep, truth), 0
    if ext == ".npz":
        z = np.load(path)
        return array_to_matrix(z["predictions"] if "predictions" in z.files else z[z.files[0]], keep, truth), 0
    if ext in (".pkl", ".pickle"):
        obj = pickle.load(open(path, "rb"))              # arquivo gerado por você mesmo
        if isinstance(obj, (np.ndarray, pd.DataFrame)):
            return array_to_matrix(obj.values if isinstance(obj, pd.DataFrame) else obj, keep, truth), 0
        return cells_to_matrix(obj)
    if ext == ".json":
        return cells_to_matrix(json.load(open(path, encoding="utf-8")))
    df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    vals = set(df.values.ravel().tolist())
    if vals <= {"0", "1", "True", "False", "0.0", "1.0", ""} and df.shape in ((n, m), (n, len(keep))):
        return array_to_matrix(df.apply(lambda s: s.isin(["1", "True", "1.0"])).values.astype(int), keep, truth), 0
    low = {c.lower(): c for c in df.columns}
    rc = next((low[k] for k in ("row", "row_idx", "i", "index", "tuple", "row_id") if k in low), None)
    cc = next((low[k] for k in ("col", "column", "attr", "attribute", "j", "col_idx") if k in low), None)
    pairs = list(zip(df[rc], df[cc])) if rc and cc else (list(zip(df.iloc[:, 0], df.iloc[:, 1])) if df.shape[1] == 2 else [])
    return cells_to_matrix(pairs)


# ------------------------------------------------------------ modo --from-file: detecção pronta
def run_from_file():
    if None in (args.run_time, args.prompt_tokens, args.completion_tokens):
        sys.exit("--from-file exige --run-time, --prompt-tokens e --completion-tokens (vão para o metadata.json)")
    det_file = os.path.abspath(args.from_file)
    if not os.path.isfile(det_file):
        sys.exit(f"arquivo não encontrado: {det_file}")
    method = "gidcl_released"
    out = os.path.join(args.outdir, args.name, method)
    config = {"source": "released", "order": args.order, "skip_cols": args.skip_cols, "row_offset": args.row_offset}
    guard_output(out, config, args.overwrite)
    clean = load(clean_path)                              # só para --order auto (layout do vetor)
    if clean.shape != dirty.shape:
        sys.exit(f"tamanhos diferentes: dirty{dirty.shape} clean{clean.shape}")
    keep = [j for j in range(m) if j not in set(args.skip_cols)]
    mat, fora = convert(det_file, keep, dirty.values != clean.values)
    if int(mat.sum()) == 0:
        sys.exit(f"nenhuma célula marcada em {det_file} (fora_dos_limites={fora}). "
                 f"Confira --skip-cols/--order/--row-offset.")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(mat, columns=cols).to_csv(os.path.join(out, "pred.csv"), index=False)
    write_metadata(out, method, args.name, config, args.run_time, uses_llm=True,
                   prompt_tokens=args.prompt_tokens, completion_tokens=args.completion_tokens,
                   extra={"origem": det_file, "fora_dos_limites": fora})
    print(f"[gidcl] {args.name} {method} detectadas={int(mat.sum())} -> {out}/pred.csv")


# ------------------------------------------------------------ modo padrão: seleção + criador-crítico
def step(argv, env):
    print("$", " ".join(argv), flush=True)
    rc = subprocess.run(argv, cwd=ROOT, env=env).returncode
    if rc != 0:
        sys.exit(f"etapa falhou (código {rc}): {' '.join(argv)}")


def run_pipeline():
    miss = [v for v in LLM_VARS if not os.environ.get(v)]
    if miss:
        sys.exit("Defina no .env: " + ", ".join(miss))
    if args.budget < 1:
        sys.exit("--budget deve ser positivo")
    clusters = args.clusters or PAPER_CLUSTERS.get(args.name.lower(), args.budget)

    # o nome da pasta inclui os parâmetros do artigo que mudam o resultado; os demais ficam na config
    # (guard_output aborta se a pasta já tiver resultado de outra configuração)
    method = f"gidcl_b{args.budget}_k{clusters}"
    if args.select != "cluster":
        method += "_" + args.select
    if args.embed != "sbert":
        method += "_" + args.embed
    if args.lm != "roberta-base":
        method += "_" + os.path.basename(args.lm.rstrip("/"))
    method += f"_s{args.seed}"
    out = os.path.join(args.outdir, args.name, method)
    detect_cfg = {"lm": args.lm, "tau": args.tau, "iter_max": args.iter_max, "rounds": args.rounds,
                  "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "max_len": args.max_len,
                  "pseudo_per_col": args.pseudo_per_col, "coreset_n": args.coreset_n,
                  "coreset_conf": args.coreset_conf, "col_ctx_max": args.col_ctx_max}
    config = {"budget": args.budget, "clusters": clusters, "select": args.select, "embed": args.embed,
              "embed_model": args.embed_model if args.embed == "sbert" else None, "seed": args.seed,
              **detect_cfg, "llm_model": os.environ["OPENAI_MODEL"]}
    guard_output(out, config, args.overwrite)
    os.makedirs(out, exist_ok=True)
    split = os.path.join(out, "split")

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    py = sys.executable
    t0 = time.time()
    # 1) o método só enxerga a tabela suja + as θ tuplas rotuladas (o clean completo fica de fora)
    step([py, os.path.join("scripts", "make_labeled_split.py"), "--dirty", dirty_path, "--clean", clean_path,
          "--k", str(args.budget), "--clusters", str(clusters), "--seed", str(args.seed),
          "--select", args.select, "--embed", args.embed, "--embed-model", args.embed_model,
          "--outdir", split], env)
    # 2) criador-crítico (não recebe o caminho do clean)
    detect = [py, os.path.join("methods", "gidcl_det", "gidcl_detect.py"), "--dirty", dirty_path,
              "--labeled", os.path.join(split, "labeled_cells.csv"),
              "--clusters", os.path.join(split, "clusters.csv"), "--outdir", out, "--seed", str(args.seed),
              "--infer-batch", str(args.infer_batch)]
    for k, v in detect_cfg.items():
        detect += ["--" + k.replace("_", "-"), str(v)]
    step(detect, env)
    elapsed = time.time() - t0

    mat = np.load(os.path.join(out, "pred_matrix.npy")).astype(int)   # detecção do M_det (linha 23 da Fig. 3)
    if mat.shape != (n, m):
        sys.exit(f"pred_matrix.npy com forma {mat.shape}; esperado {(n, m)}")
    report = json.load(open(os.path.join(out, "report.json"), encoding="utf-8"))
    usage = report.get("uso_llm") or {}
    rounds = report.get("rodadas") or []
    pd.DataFrame(mat, columns=cols).to_csv(os.path.join(out, "pred.csv"), index=False)
    # tuplas cujo rótulo o método viu: avalie também sem elas (evaluate.py --exclude-rows)
    shutil.copyfile(os.path.join(split, "labeled_rows.csv"), os.path.join(out, "labeled_rows.csv"))
    extra = {"llm_calls": usage.get("calls"), "rodadas": len(rounds),
             "funcoes_aceitas": rounds[-1]["accepted"] if rounds else [],
             "pseudo": rounds[-1]["pseudo"] if rounds else 0,
             "detectadas_funcoes": report.get("marcadas_funcoes"), "segundos_deteccao": report.get("total_s")}
    write_metadata(out, method, args.name, config, elapsed, uses_llm=True,
                   prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                   extra=extra)
    print(f"[gidcl] {args.name} {method} detectadas={int(mat.sum())} tempo={elapsed:.1f}s -> {out}/pred.csv")


if args.from_file:
    run_from_file()
else:
    run_pipeline()
