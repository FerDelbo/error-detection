#!/usr/bin/env python3
"""
Variante do ZeroED em que o Jev (jevai.net) rotula os valores centrais no lugar do LLM.
Código em methods/zeroed_jev/ (cópia congelada do ZeroED; mudanças marcadas com "# JEV").
NÃO usa nem altera methods/zeroed/, runners/run_zeroed.py ou as saídas zeroed_s<N>.

Saída: outputs/<name>/zeroed_<variante>_qwen_s<N>/ com pred.csv, metadata.json e detected.json
(com JEV_BACKEND=mock o nome ganha _mock: é só teste de encanamento, não é resultado).

Variantes (--variant):                 pre_func  distri/guide  rotulador  func  err_gen  proteção
  base         ZeroED com o Qwen          LLM        LLM          LLM     LLM    LLM       não
  jevlab       troca só o rotulador       LLM        —            Jev     LLM    LLM       não
  jevlab_feat  + sem funções clean        —          —            Jev     LLM    LLM       não
  jevlab_aug   LLM só gera erros          —          —            Jev     —      LLM       sim
"proteção" = empty_func_guard: atributo sem funções não perde os rótulos "correto" (ver README).
A base é o ZeroED original com o mesmo Qwen das variantes: é a referência justa (os zeroed_s<N>
existentes usaram outro LLM) e confirma que a cópia se comporta como o original.

LLM (Qwen, como no GIDCL): QWEN_MODEL (padrão Qwen/Qwen2.5-14B-Instruct), QWEN_BASE_URL
(padrão http://localhost:8000/v1), QWEN_API_KEY (padrão EMPTY), QWEN_EXTRA_BODY (JSON opcional).
Temperatura: ZEROED_JEV_TEMPERATURE (padrão 0, como no artigo).
Jev: JEV_URL, JEV_API_KEY, JEV_MODEL (padrão jev-1.14.0), JEV_BACKEND (api | mock); ver
methods/zeroed_jev/jev_client.py.

Subcomandos (todos com --name, --dirty, --clean, --variant):
  prepare : copia dirty/clean para methods/zeroed_jev/data/ e gera configs/<tag>_<variante>.yaml
  fasttext: fastText reduzido (reaproveita, por link, o de methods/zeroed se já existir)
  run     : roda methods/zeroed_jev/main.py
  export  : detected.json -> pred.csv + metadata.json (tokens por etapa, custo do Jev, qualidade dos rótulos)
  all     : prepare + fasttext + run + export
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runners"))
from pipeline_meta import guard_output, write_metadata
ZDIR = os.path.join(ROOT, "methods", "zeroed_jev")
ZDIR_ORIG = os.path.join(ROOT, "methods", "zeroed")  # só leitura (fastText)

QWEN_DEFAULTS = {"QWEN_MODEL": "Qwen/Qwen2.5-14B-Instruct", "QWEN_BASE_URL": "http://localhost:8000/v1"}
JEV_DEFAULT_MODEL = "jev-1.14.0"
JEV_PRICE_PER_M = float(os.getenv("JEV_PRICE_PER_M", "0.084"))  # US$/M tokens de entrada; saída grátis

VARIANTS = {
    "base":        dict(labeler="llm", pre_func=True,  distri=True,  guide=True,  func=True,  err_gen=True, guard=False),
    "jevlab":      dict(labeler="jev", pre_func=True,  distri=False, guide=False, func=True,  err_gen=True, guard=False),
    "jevlab_feat": dict(labeler="jev", pre_func=False, distri=False, guide=False, func=True,  err_gen=True, guard=False),
    "jevlab_aug":  dict(labeler="jev", pre_func=False, distri=False, guide=False, func=False, err_gen=True, guard=True),
}
DEFAULT_CLUSTER = "RANDOM"


# ---------------------------------------------------------------- nomes
def is_mock(a):
    return VARIANTS[a.variant]["labeler"] == "jev" and os.getenv("JEV_BACKEND", "api") == "mock"


def suffix(a):
    s = f"_ft{a.ft_dim}" if a.ft_dim else ""
    if a.cluster != DEFAULT_CLUSTER:
        s += f"_{a.cluster.lower()}"
    return s


def method_of(a):
    return f"zeroed_{a.variant}_qwen" + suffix(a) + ("_mock" if is_mock(a) else "")


def tag_of(a):
    """Nome do dataset dentro do ZeroED (só letras/dígitos/_; o ZeroED monta caminhos com ele)."""
    t = re.sub(r"[^A-Za-z0-9_]", "", a.name)
    if not t:
        sys.exit(f"--name inválido: {a.name!r}")
    return t


def result_dir_of(a):
    """Pasta de resultados do main.py (methods/zeroed_jev/result/<isto>/): uma por variante."""
    return f"pipeline_{a.variant}" + suffix(a) + ("_mock" if is_mock(a) else "")


def config_path_of(a):
    return os.path.join(ZDIR, "configs", f"{tag_of(a)}_{result_dir_of(a)}.yaml")


def usage_files(a):
    base = os.path.join(ROOT, "logs", f"zeroed_jev_{tag_of(a)}_{result_dir_of(a)}")
    return base + "_llm.jsonl", base + "_jev.jsonl"


def ft_dim_of(a):
    if a.ft_dim:
        if not 1 <= a.ft_dim <= 300:
            sys.exit("--ft-dim deve estar entre 1 e 300 (300 = sem redução)")
        return a.ft_dim
    return len(pd.read_csv(a.dirty, nrows=0).columns)


def llm_model():
    return os.getenv("QWEN_MODEL") or QWEN_DEFAULTS["QWEN_MODEL"]


def check_inputs(a):
    for p in (a.dirty, a.clean):
        if not os.path.isfile(p):
            sys.exit(f"arquivo não encontrado: {p}")
    if not os.path.isfile(os.path.join(ZDIR, "main.py")):
        sys.exit(f"variante não encontrada em {ZDIR}")


# ---------------------------------------------------------------- subcomandos
def prepare(a):
    check_inputs(a)
    tag, v = tag_of(a), VARIANTS[a.variant]
    os.makedirs(os.path.join(ZDIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(ZDIR, "configs"), exist_ok=True)
    shutil.copy(a.dirty, os.path.join(ZDIR, "data", f"{tag}_error-01.csv"))
    # como no run_zeroed.py: o clean é regravado com o cabeçalho do dirty (por posição; valores intactos)
    dirty_cols = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, nrows=0).columns
    clean = pd.read_csv(a.clean, dtype=str, keep_default_na=False, low_memory=False)
    assert len(clean.columns) == len(dirty_cols), "dirty e clean têm números de colunas diferentes"
    clean.columns = dirty_cols
    clean.to_csv(os.path.join(ZDIR, "data", f"{tag}_clean.csv"), index=False)
    cfg = {
        "model": {
            "api_use": True, "pre_func_use": v["pre_func"], "related_attrs": True,
            "distri_analysis": v["distri"], "guide_use": v["guide"], "func_use": v["func"],
            "n_method": a.n_method, "err_gen_use": v["err_gen"], "rel_top": 2, "run_info": result_dir_of(a),
        },
        "jev": {"labeler": v["labeler"], "empty_func_guard": v["guard"],
                "threshold": a.threshold, "workers": a.jev_workers},
        # tudo False = execução do zero (sem reaproveitar resultados antigos)
        "read": {
            "pre_func": False, "distri_analysis": False, "related_attrs": False,
            "cluster": False, "guide": False, "error_checking": False, "func": False,
            "err_gen": False, "read_in_batch": False,
            "start_time": "01-01-00:00", "end_time": "01-01-00:00",
        },
        "data": {
            "base_dir": ".", "err_rate_list": ["01"], "all_set_num": a.runs,
            "datasets": [tag], "err_check_val_num_per_query": 20,
            "result_dir": result_dir_of(a),
        },
        "read_paths": {},
    }
    with open(config_path_of(a), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"config gerada: {config_path_of(a)} ({a.variant}, {a.runs} execução(ões) de '{tag}')")


def build_fasttext(a):
    """methods/zeroed_jev/cc.en.<dim>.bin: reaproveita por link o de methods/zeroed; senão reduz o de 300d."""
    dim = ft_dim_of(a)
    full = os.path.join(ZDIR, "cc.en.300.bin")
    if not os.path.lexists(full):  # *.bin fica fora do git: o link é recriado aqui
        os.symlink(os.path.join("..", "zeroed", "cc.en.300.bin"), full)
    out = os.path.join(ZDIR, f"cc.en.{dim}.bin")
    if os.path.exists(out):
        print(f"já existe: {out}")
        return
    orig = os.path.join(ZDIR_ORIG, f"cc.en.{dim}.bin")
    if os.path.exists(orig):
        os.symlink(os.path.relpath(orig, ZDIR), out)
        print(f"link para o fastText do ZeroED: {out} -> {orig}")
        return
    import fasttext
    import fasttext.util
    if not os.path.exists(full):
        sys.exit(f"Falta {full} (link para methods/zeroed/cc.en.300.bin, ~7 GB).")
    print(f"reduzindo o fastText para {dim} dimensões (usa MUITA memória; feche outros programas)...")
    model = fasttext.load_model(full)
    fasttext.util.reduce_model(model, dim)
    model.save_model(out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"gerado: {out}")


def run(a):
    check_inputs(a)
    if not os.path.exists(config_path_of(a)):
        sys.exit("Rode antes o subcomando 'prepare' (ou use 'all').")
    if not os.path.exists(os.path.join(ZDIR, f"cc.en.{ft_dim_of(a)}.bin")):
        sys.exit(f"Falta cc.en.{ft_dim_of(a)}.bin. Rode antes o subcomando 'fasttext' (ou use 'all').")
    if VARIANTS[a.variant]["labeler"] == "jev" and not is_mock(a) and not os.getenv("JEV_URL"):
        sys.exit("Defina JEV_URL e JEV_API_KEY (no .env ou no ambiente), ou JEV_BACKEND=mock para um teste sem API.")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    llm_uf, jev_uf = usage_files(a)
    for uf in (llm_uf, jev_uf):
        if os.path.exists(uf):
            os.remove(uf)  # uso deste dataset/variante começa do zero a cada 'run'
    env = dict(os.environ, ZEROED_USAGE_FILE=llm_uf, ZEROED_JEV_USAGE_FILE=jev_uf,
               ZEROED_PROCS=str(a.procs), ZEROED_CLUSTER=a.cluster, PYTHONUNBUFFERED="1")
    env.pop("ZEROED_FT_DIM", None)
    if a.ft_dim:
        env["ZEROED_FT_DIM"] = str(a.ft_dim)
    print(f"[{method_of(a)}] LLM: {llm_model()} em {os.getenv('QWEN_BASE_URL') or QWEN_DEFAULTS['QWEN_BASE_URL']}")
    subprocess.run([sys.executable, "main.py", "--config", os.path.relpath(config_path_of(a), ZDIR)],
                   cwd=ZDIR, env=env, check=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def label_quality(center_labels_path, dirty, clean):
    """Qualidade dos rótulos dos centros contra o ground truth (dirty != clean, como make_ground_truth.py).
    Só para o relatório: o clean não entra no método."""
    if not os.path.exists(center_labels_path):
        return None
    labels = json.load(open(center_labels_path))
    cols = list(dirty.columns)
    err = dirty.to_numpy() != clean.to_numpy()
    tp = fp = fn = tn = 0
    for attr, pairs in labels.items():
        j = cols.index(attr)
        for i, lab in pairs:
            truth = bool(err[i, j])
            if lab == 1 and truth:
                tp += 1
            elif lab == 1:
                fp += 1
            elif truth:
                fn += 1
            else:
                tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"centros": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def export(a):
    check_inputs(a)
    tag, v = tag_of(a), VARIANTS[a.variant]
    llm_uf, jev_uf = usage_files(a)
    llm_rows, jev_rows = read_jsonl(llm_uf), read_jsonl(jev_uf)
    tokens = {"prompt_tokens": sum(u["p"] for u in llm_rows), "completion_tokens": sum(u["c"] for u in llm_rows)}
    per_stage = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    for u in llm_rows:
        s = per_stage[u.get("s", "unknown")]
        s["calls"] += 1
        s["prompt_tokens"] += u["p"]
        s["completion_tokens"] += u["c"]
    jev = {"requests": sum(not u["cached"] for u in jev_rows), "cached": sum(u["cached"] for u in jev_rows),
           "questions": sum(u["n_questions"] for u in jev_rows if not u["cached"]),
           "input_tokens": sum(u["tokens_in"] for u in jev_rows),
           "tokens_estimated": any(u.get("estimated") for u in jev_rows)}
    jev["cost_usd"] = round(jev["input_tokens"] / 1e6 * JEV_PRICE_PER_M, 6)

    # pastas: methods/zeroed_jev/result/<result_dir>/<MM-DD> <tag>01-<n_method>-set<N>/
    runs = glob.glob(os.path.join(ZDIR, "result", result_dir_of(a), f"* {tag}01-*-set*"))
    latest = {}
    for r in runs:
        if not os.path.exists(os.path.join(r, "detected.json")):
            continue
        n = r.rsplit("-set", 1)[1]
        if n not in latest or os.path.getmtime(r) > os.path.getmtime(latest[n]):
            latest[n] = r
    if not latest:
        sys.exit(f"nenhuma execução concluída (detected.json) para '{tag}' em {result_dir_of(a)}")

    dirty = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, low_memory=False)
    clean = pd.read_csv(a.clean, dtype=str, keep_default_na=False, low_memory=False)
    k = len(latest)
    for n in sorted(latest, key=int):
        r = latest[n]
        mat = np.zeros(dirty.shape, dtype=int)
        skipped = 0
        for i, j in json.load(open(os.path.join(r, "detected.json"))):
            if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
                mat[i, j] = 1
            else:
                skipped += 1
        total = None
        tf = os.path.join(r, "0-time.txt")
        if os.path.exists(tf):
            for line in open(tf):
                if line.startswith("total:"):
                    total = round(float(line.split(":")[1]), 1)
        if total is None:
            sys.exit(f"tempo total não encontrado em {tf}; metadata.json não pode ser gerado")
        cfg = {"variant": a.variant, "labeler": v["labeler"], "pre_func": v["pre_func"],
               "distri_analysis": v["distri"], "guide": v["guide"], "func": v["func"], "err_gen": v["err_gen"],
               "empty_func_guard": v["guard"], "n_method": a.n_method, "cluster": a.cluster, "rel_top": 2,
               "batch": 20, "temperature": float(os.getenv("ZEROED_JEV_TEMPERATURE", "0")),
               "ft_dim": ft_dim_of(a), "runs": a.runs, "procs": a.procs, "llm_model": llm_model(), "set": int(n)}
        if v["labeler"] == "jev":
            cfg.update({"jev_model": os.getenv("JEV_MODEL") or JEV_DEFAULT_MODEL,
                        "jev_backend": os.getenv("JEV_BACKEND", "api"), "threshold": a.threshold})
        out = os.path.join(a.outdir, a.name, f"{method_of(a)}_s{n}")
        guard_output(out, cfg, a.overwrite)
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
        shutil.copy(os.path.join(r, "detected.json"), out)
        lq = label_quality(os.path.join(r, "center_labels.json"), dirty, clean)
        # os arquivos de uso somam todos os sets deste 'run': grava-se a média por set (como no run_zeroed.py)
        write_metadata(out, f"{method_of(a)}_s{n}", a.name, cfg, total, uses_llm=True,
                       prompt_tokens=tokens["prompt_tokens"] / k, completion_tokens=tokens["completion_tokens"] / k,
                       extra={"origem": r, "fora_dos_limites": skipped, "tokens_media_de_n_sets": k,
                              "tokens_total": tokens, "llm_por_etapa_total": dict(per_stage),
                              "jev_total": jev if v["labeler"] == "jev" else None,
                              "qualidade_rotulos_centros": lq})
        print(f"[{method_of(a)}] {a.name} set{n}: detectadas={int(mat.sum())} tempo={total}s -> {out}/pred.csv")
        if lq:
            print(f"   rótulos dos centros: P={lq['precision']} R={lq['recall']} F1={lq['f1']} ({lq['centros']} centros)")
    print("LLM (soma de todos os sets):", tokens)
    for s, u in sorted(per_stage.items()):
        print(f"   {s:14s} {u}")
    if v["labeler"] == "jev":
        print("Jev:", jev)


def all_steps(a):
    prepare(a)
    build_fasttext(a)
    run(a)
    export(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("prepare", prepare), ("fasttext", build_fasttext), ("run", run),
                     ("export", export), ("all", all_steps)]:
        p = sub.add_parser(name)
        p.add_argument("--name", required=True, help="rótulo do dataset (usado nas saídas)")
        p.add_argument("--dirty", required=True, help="caminho do CSV sujo")
        p.add_argument("--clean", required=True, help="caminho do CSV limpo")
        p.add_argument("--variant", required=True, choices=list(VARIANTS))
        p.add_argument("--runs", type=int, default=1, help="execuções (sets) por dataset")
        p.add_argument("--procs", default="2", help="ZEROED_PROCS (processos paralelos)")
        p.add_argument("--n-method", default="5%", help="proporção de linhas rotuladas (artigo: 5%%)")
        p.add_argument("--cluster", default=DEFAULT_CLUSTER,
                       choices=["RANDOM", "KMeans", "DBSCAN", "Agglomerative"],
                       help="amostragem das linhas rotuladas (RANDOM = código publicado; KMeans = artigo)")
        p.add_argument("--ft-dim", type=int, default=None,
                       help="dimensão do fastText (padrão: nº de colunas; 300 = sem redução)")
        p.add_argument("--threshold", type=float, default=0.5,
                       help="p(erro) acima do qual o Jev rotula o centro como erro (fixo; não ajustar com o GT)")
        p.add_argument("--jev-workers", type=int, default=4, help="requisições simultâneas ao Jev")
        p.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
        p.add_argument("--overwrite", action="store_true", help="sobrescreve saída de outra configuração")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.dirty, args.clean = os.path.abspath(args.dirty), os.path.abspath(args.clean)
    args.fn(args)
