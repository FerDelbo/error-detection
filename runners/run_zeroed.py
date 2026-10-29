#!/usr/bin/env python3
"""
ZeroED no formato comum do pipeline: dirty/clean entram por caminho e a saída é
outputs/<name>/zeroed_s<N>/pred.csv (matriz 0/1 do tamanho do dirty).
Com --ft-dim 300 (variante sem redução do fastText): outputs/<name>/zeroed_ft300_s<N>/.

Subcomandos (todos usam --name, --dirty, --clean):
  prepare : copia dirty/clean para methods/zeroed/data/, gera config_pipeline.yaml e aplica os patches
  fasttext: gera o fastText reduzido cc.en.<dim>.bin (1 vez por dimensão)
  run     : roda main.py (exige ZEROED_API_KEY, ZEROED_BASE_URL e ZEROED_MODEL)
  export  : converte detected.json -> pred.csv
  all     : prepare + fasttext + run + export

Execute com o Python do ambiente conda do ZeroED, a partir de qualquer pasta.
Pré-requisito: methods/zeroed com o código do ZeroED e cc.en.300.bin (pode ser link simbólico).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZDIR = os.path.join(ROOT, "methods", "zeroed")

# ---------------------------------------------------------------- patches (idempotentes)
PATCHES = {
    "feature.py": [],
    "prompt_gen.py": [
        (
            "def error_check_prompt(col_values, col_name):",
            "guide_gen_prompt = kb_gen_prompt  # PATCH\n\n\ndef error_check_prompt(col_values, col_name):",
        ),
    ],
    "main.py": [
        (
            "            measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n",
            "            measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n"
            "            with open(os.path.join(resp_path, 'detected.json'), 'w') as _f:  # PATCH\n"
            "                json.dump(sorted([int(i), all_attrs.index(a)] for i, a in det_wrong_list), _f)\n",
        ),
        (
            "feat_pool = mp.Pool()",
            "feat_pool = mp.Pool(int(os.environ.get('ZEROED_PROCS', '2')))  # PATCH",
        ),
        (
            "with multiprocessing.Pool(len(all_attrs)) as pool:\n            results = [pool.apply_async(cluster,",
            "with multiprocessing.Pool(int(os.environ.get('ZEROED_PROCS', '2'))) as pool:  # PATCH\n            results = [pool.apply_async(cluster,",
        ),
    ],
    "utility.py": [
        (
            "def get_ans_from_llm(prompt, api_use=False):",
            "def get_ans_from_llm(prompt, api_use=True):  # PATCH",
        ),
        (
            "def split_list_to_sublists(original_list, sublist_size):",
            "def load_reduced_fasttext(dim):  # PATCH\n"
            "    import fasttext\n"
            "    import fasttext.util\n"
            "    cached = f'./cc.en.{dim}.bin'\n"
            "    if os.path.exists(cached):\n"
            "        return fasttext.load_model(cached)\n"
            "    model = fasttext.load_model('./cc.en.300.bin')\n"
            "    fasttext.util.reduce_model(model, dim)\n"
            "    model.save_model(cached + '.tmp')\n"
            "    os.replace(cached + '.tmp', cached)\n"
            "    return model\n\n\n"
            "def split_list_to_sublists(original_list, sublist_size):",
        ),
        # dimensão configurável por variável de ambiente (variante sem redução: ZEROED_FT_DIM=300)
        (
            "    cached = f'./cc.en.{dim}.bin'\n",
            "    dim = int(os.environ.get('ZEROED_FT_DIM', dim))  # PATCH\n"
            "    cached = f'./cc.en.{dim}.bin'\n",
        ),
        # (
        #     "        model_type='qwen2.5-7b-instruct'\n",
        #     "        model_type = os.environ.get('ZEROED_MODEL', 'qwen2.5-7b-instruct')  # PATCH\n",
        # ),
        (
            "        api_key_list = [\n            ''\n        ]\n",
            "        api_key_list = [os.getenv('ZEROED_API_KEY', '')]  # PATCH\n",
        ),
        (
            "                    base_url='https://api.openai.com/v1'\n",
            "                    base_url=os.getenv('ZEROED_BASE_URL', 'https://api.openai.com/v1')  # PATCH\n",
        ),
        (
            "                response = completion_res.choices[0].message.content\n",
            "                response = completion_res.choices[0].message.content\n"
            "                try:  # PATCH\n"
            "                    with open(os.getenv('ZEROED_USAGE_FILE', 'llm_usage.jsonl'), 'a') as _u:\n"
            "                        _u.write('{\"p\": %d, \"c\": %d}\\n' % (completion_res.usage.prompt_tokens, completion_res.usage.completion_tokens))\n"
            "                except Exception:\n"
            "                    pass\n",
        ),
    ],
}

FT_PATTERN = re.compile(
    r"(?P<i1>[ \t]*)(?P<var>\w+) = fasttext\.load_model\('\./cc\.en\.300\.bin'\)[ \t]*\n"
    r"(?P<i2>[ \t]*)fasttext_dimension = (?P<dim>[^\n]*?)[ \t]*\n"
    r"(?P<i3>[ \t]*)fasttext\.util\.reduce_model\((?P=var), fasttext_dimension\)[ \t]*\n"
)
FT_EXPECTED = {"main.py": 3, "feature.py": 1}


def patch_fasttext(fname, text):
    if "load_reduced_fasttext" in text:
        return text

    def repl(m):
        return (f"{m['i1']}from utility import load_reduced_fasttext  # PATCH\n"
                f"{m['i1']}{m['var']} = load_reduced_fasttext({m['dim']})  # PATCH\n"
                f"{m['i2']}fasttext_dimension = {m['dim']}\n")
    new_text, n = FT_PATTERN.subn(repl, text)
    if n != FT_EXPECTED[fname]:
        sys.exit(f"ERRO: esperava {FT_EXPECTED[fname]} trechos de fastText em {fname}, achei {n}")
    return new_text


def apply_patches():
    for fname, reps in PATCHES.items():
        path = os.path.join(ZDIR, fname)
        src = open(path, encoding="utf-8", newline="").read()
        crlf = "\r\n" in src
        text = src.replace("\r\n", "\n")
        for old, new in reps:
            # assinatura = a linha com "# PATCH"; tolera diferenças de espaço e indentação
            sig = next((l.strip() for l in new.splitlines() if "# PATCH" in l), None)
            if new in text or (sig and sig in text):
                continue  # já aplicado        
	#for old, new in reps:
         #   if new in text:
          #      continue
            if old not in text:
                sys.exit(f"ERRO: trecho não encontrado em {fname}; o repositório mudou?\n---\n{old}")
            text = text.replace(old, new, 1)
        if fname in FT_EXPECTED:
            text = patch_fasttext(fname, text)
        if crlf:
            text = text.replace("\n", "\r\n")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print("patches aplicados (ou já presentes)")


# ---------------------------------------------------------------- utilidades do pipeline
def suffix(a):
    return f"_ft{a.ft_dim}" if a.ft_dim else ""


def method_of(a):
    return "zeroed" + suffix(a)


def tag_of(a):
    """Nome interno do ZeroED nos arquivos e pastas (sem hífen, espaço ou símbolos)."""
    t = re.sub(r"[^A-Za-z0-9_]", "", a.name)
    if not t:
        sys.exit(f"--name inválido: {a.name!r}")
    return t + suffix(a)


def ft_dim_of(a):
    if a.ft_dim:
        if not 1 <= a.ft_dim <= 300:
            sys.exit("--ft-dim deve estar entre 1 e 300 (300 = sem redução)")
        return a.ft_dim
    return len(pd.read_csv(a.dirty, nrows=0).columns)


def usage_file(a):
    return os.path.join(ROOT, "logs", f"zeroed_{tag_of(a)}_usage.jsonl")


def check_inputs(a):
    for p in (a.dirty, a.clean):
        if not os.path.isfile(p):
            sys.exit(f"arquivo não encontrado: {p}")
    if not os.path.isfile(os.path.join(ZDIR, "main.py")):
        sys.exit(f"ZeroED não encontrado em {ZDIR}")


# ---------------------------------------------------------------- subcomandos
def prepare(a):
    check_inputs(a)
    tag = tag_of(a)
    os.makedirs(os.path.join(ZDIR, "data"), exist_ok=True)
    shutil.copy(a.dirty, os.path.join(ZDIR, "data", f"{tag}_error-01.csv"))
    # O ZeroED acessa o clean pelos NOMES de coluna do dirty. Se os cabeçalhos diferem,
    # o clean é regravado com o cabeçalho do dirty (por posição; valores intactos).
    dirty_cols = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, nrows=0).columns
    clean = pd.read_csv(a.clean, dtype=str, keep_default_na=False, low_memory=False)
    assert len(clean.columns) == len(dirty_cols), "dirty e clean têm números de colunas diferentes"
    clean.columns = dirty_cols
    clean.to_csv(os.path.join(ZDIR, "data", f"{tag}_clean.csv"), index=False)
    cfg = {
        "model": {
            "api_use": True, "pre_func_use": True, "related_attrs": True,
            "distri_analysis": True, "guide_use": True, "func_use": True,
            "n_method": a.n_method, "err_gen_use": True, "rel_top": 2, "run_info": "pipeline",
        },
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
            "result_dir": "pipeline",
        },
        "read_paths": {},
    }
    with open(os.path.join(ZDIR, "config_pipeline.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    pass  # patches desativados (os arquivos já estão patchados)
    print(f"config gerada: {ZDIR}/config_pipeline.yaml ({a.runs} execução(ões) de '{tag}', "
          f"fastText {ft_dim_of(a)}d)")


def build_fasttext(a):
    """Gera methods/zeroed/cc.en.<dim>.bin (reduzido) uma vez por dimensão. Com 300, nada a fazer."""
    import fasttext
    import fasttext.util
    dim = ft_dim_of(a)
    out = os.path.join(ZDIR, f"cc.en.{dim}.bin")
    if os.path.exists(out):
        print(f"já existe: {out}")
        return
    full = os.path.join(ZDIR, "cc.en.300.bin")
    if not os.path.exists(full):
        sys.exit(f"Falta {full} (modelo fastText completo, ~7 GB). Crie um link simbólico ou baixe-o.")
    print(f"reduzindo o fastText para {dim} dimensões (usa MUITA memória; feche outros programas)...")
    model = fasttext.load_model(full)
    fasttext.util.reduce_model(model, dim)
    model.save_model(out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"gerado: {out}")


def run(a):
    check_inputs(a)
    cfg_path = os.path.join(ZDIR, "config_pipeline.yaml")
    if not os.path.exists(cfg_path):
        sys.exit("Rode antes o subcomando 'prepare' (ou use 'all').")
    cfg = yaml.safe_load(open(cfg_path))
    if cfg["data"]["datasets"] != [tag_of(a)]:
        sys.exit(f"config_pipeline.yaml é de {cfg['data']['datasets']}; rode 'prepare' para {tag_of(a)}")
    dim = ft_dim_of(a)
    if not os.path.exists(os.path.join(ZDIR, f"cc.en.{dim}.bin")):
        sys.exit(f"Falta cc.en.{dim}.bin. Rode antes o subcomando 'fasttext' (ou use 'all').")
    for v in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        if not os.getenv(v):
            sys.exit(f"Defina {v} (no .env ou no ambiente).")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    uf = usage_file(a)
    if os.path.exists(uf):
        os.remove(uf)  # tokens deste dataset começam do zero a cada 'run'
    env = dict(os.environ, ZEROED_USAGE_FILE=uf, ZEROED_PROCS=str(a.procs), PYTHONUNBUFFERED="1")
    env.pop("ZEROED_FT_DIM", None)
    if a.ft_dim:
        env["ZEROED_FT_DIM"] = str(a.ft_dim)
    subprocess.run([sys.executable, "main.py", "--config", "config_pipeline.yaml"],
                   cwd=ZDIR, env=env, check=True)


def export(a):
    check_inputs(a)
    tag = tag_of(a)
    tokens = {"prompt_tokens": 0, "completion_tokens": 0}
    uf = usage_file(a)
    if os.path.exists(uf):
        for line in open(uf):
            u = json.loads(line)
            tokens["prompt_tokens"] += u["p"]
            tokens["completion_tokens"] += u["c"]

    # pastas: methods/zeroed/result/pipeline/<MM-DD> <tag>01-<n_method>-set<N>/
    runs = glob.glob(os.path.join(ZDIR, "result", "pipeline", f"* {tag}01-*-set*"))
    latest = {}
    for r in runs:
        if not os.path.exists(os.path.join(r, "detected.json")):
            continue
        n = r.rsplit("-set", 1)[1]
        if n not in latest or os.path.getmtime(r) > os.path.getmtime(latest[n]):
            latest[n] = r
    if not latest:
        sys.exit(f"nenhuma execução concluída (detected.json) para '{tag}'")

    dirty = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, low_memory=False)
    for n in sorted(latest, key=int):
        r = latest[n]
        mat = np.zeros(dirty.shape, dtype=int)
        skipped = 0
        for i, j in json.load(open(os.path.join(r, "detected.json"))):
            if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
                mat[i, j] = 1
            else:
                skipped += 1
        out = os.path.join(a.outdir, a.name, f"{method_of(a)}_s{n}")
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
        shutil.copy(os.path.join(r, "detected.json"), out)
        total = None
        tf = os.path.join(r, "0-time.txt")
        if os.path.exists(tf):
            for line in open(tf):
                if line.startswith("total:"):
                    total = round(float(line.split(":")[1]), 1)
        json.dump({"total_s": total, "ft_dim": ft_dim_of(a), "n_sets_no_arquivo_de_tokens": len(latest), **tokens},
                  open(os.path.join(out, "runtime.json"), "w"))
        print(f"[{method_of(a)}] {a.name} set{n}: detectadas={int(mat.sum())} fora_dos_limites={skipped} "
              f"tempo={total}s -> {out}/pred.csv\n   métricas do próprio ZeroED: {os.path.join(r, 'func_det_res.txt')}")
    print("Tokens (soma de todas as execuções deste dataset):", tokens)


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
        p.add_argument("--runs", type=int, default=1, help="execuções (sets) por dataset")
        p.add_argument("--procs", default="2", help="ZEROED_PROCS (processos paralelos)")
        p.add_argument("--n-method", default="5%", help="proporção de linhas rotuladas pelo LLM")
        p.add_argument("--ft-dim", type=int, default=None,
                       help="dimensão do fastText (padrão: nº de colunas; 300 = sem redução)")
        p.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.dirty, args.clean = os.path.abspath(args.dirty), os.path.abspath(args.clean)
    args.fn(args)
