# #!/usr/bin/env python3
# """
# ZeroED no formato comum do pipeline: dirty/clean entram por caminho e a saída é
# outputs/<name>/zeroed_s<N>/pred.csv (matriz 0/1 do tamanho do dirty).

# Subcomandos (todos usam --name, --dirty, --clean):
#   prepare : copia dirty/clean para methods/zeroed/data/, gera config_pipeline.yaml e aplica os patches
#   fasttext: gera o fastText reduzido cc.en.<dim>.bin (1 vez por dimensão)
#   run     : roda main.py (exige ZEROED_API_KEY, ZEROED_BASE_URL, ZEROED_MODEL)
#   export  : converte detected.json -> pred.csv
#   all     : prepare + fasttext + run + export

# Execute com o Python do ambiente conda do ZeroED, a partir de qualquer pasta.
# Pré-requisito: methods/zeroed clonado e com cc.en.300.bin dentro (pode ser link simbólico).
# """
# import argparse
# import glob
# import json
# import os
# import re
# import shutil
# import subprocess
# import sys

# import numpy as np
# import pandas as pd
# import yaml

# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ZDIR = os.path.join(ROOT, "methods", "zeroed")

# # ---------------------------------------------------------------- patches (inalterados)
# PATCHES = {
#     "feature.py": [],
#     "prompt_gen.py": [
#         (
#             "def error_check_prompt(col_values, col_name):",
#             "guide_gen_prompt = kb_gen_prompt  # PATCH\n\n\ndef error_check_prompt(col_values, col_name):",
#         ),
#     ],
#     "main.py": [
#         (
#             "            measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n",
#             "            measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n"
#             "            with open(os.path.join(resp_path, 'detected.json'), 'w') as _f:  # PATCH\n"
#             "                json.dump(sorted([int(i), all_attrs.index(a)] for i, a in det_wrong_list), _f)\n",
#         ),
#         (
#             "feat_pool = mp.Pool()",
#             "feat_pool = mp.Pool(int(os.environ.get('ZEROED_PROCS', '2')))  # PATCH",
#         ),
#         (
#             "with multiprocessing.Pool(len(all_attrs)) as pool:\n            results = [pool.apply_async(cluster,",
#             "with multiprocessing.Pool(int(os.environ.get('ZEROED_PROCS', '2'))) as pool:  # PATCH\n            results = [pool.apply_async(cluster,",
#         ),
#     ],
#     "utility.py": [
#         (
#             "def get_ans_from_llm(prompt, api_use=False):",
#             "def get_ans_from_llm(prompt, api_use=True):  # PATCH",
#         ),
#         (
#             "def split_list_to_sublists(original_list, sublist_size):",
#             "def load_reduced_fasttext(dim):  # PATCH\n"
#             "    import fasttext\n"
#             "    import fasttext.util\n"
#             "    cached = f'./cc.en.{dim}.bin'\n"
#             "    if os.path.exists(cached):\n"
#             "        return fasttext.load_model(cached)\n"
#             "    model = fasttext.load_model('./cc.en.300.bin')\n"
#             "    fasttext.util.reduce_model(model, dim)\n"
#             "    model.save_model(cached + '.tmp')\n"
#             "    os.replace(cached + '.tmp', cached)\n"
#             "    return model\n\n\n"
#             "def split_list_to_sublists(original_list, sublist_size):",
#         ),
#         (
#             "        model_type='qwen2.5-7b-instruct'\n",
#             "        model_type = os.environ.get('ZEROED_MODEL', 'qwen2.5-7b-instruct')  # PATCH\n",
#         ),
#         (
#             "        api_key_list = [\n            ''\n        ]\n",
#             "        api_key_list = [os.environ.get('ZEROED_API_KEY', '')]  # PATCH\n",
#         ),
#         (
#             "                    base_url='https://api.openai.com/v1'\n",
#             "                    base_url=os.environ.get('ZEROED_BASE_URL', 'https://api.openai.com/v1')  # PATCH\n",
#         ),
#         (
#             "                response = completion_res.choices[0].message.content\n",
#             "                response = completion_res.choices[0].message.content\n"
#             "                try:  # PATCH\n"
#             "                    with open(os.environ.get('ZEROED_USAGE_FILE', 'llm_usage.jsonl'), 'a') as _u:\n"
#             "                        _u.write('{\"p\": %d, \"c\": %d}\\n' % (completion_res.usage.prompt_tokens, completion_res.usage.completion_tokens))\n"
#             "                except Exception:\n"
#             "                    pass\n",
#         ),
#     ],
# }

# FT_PATTERN = re.compile(
#     r"(?P<i1>[ \t]*)(?P<var>\w+) = fasttext\.load_model\('\./cc\.en\.300\.bin'\)[ \t]*\n"
#     r"(?P<i2>[ \t]*)fasttext_dimension = (?P<dim>[^\n]*?)[ \t]*\n"
#     r"(?P<i3>[ \t]*)fasttext\.util\.reduce_model\((?P=var), fasttext_dimension\)[ \t]*\n"
# )
# FT_EXPECTED = {"main.py": 3, "feature.py": 1}


# def patch_fasttext(fname, text):
#     if "load_reduced_fasttext" in text:
#         return text

#     def repl(m):
#         return (f"{m['i1']}from utility import load_reduced_fasttext  # PATCH\n"
#                 f"{m['i1']}{m['var']} = load_reduced_fasttext({m['dim']})  # PATCH\n"
#                 f"{m['i2']}fasttext_dimension = {m['dim']}\n")
#     new_text, n = FT_PATTERN.subn(repl, text)
#     if n != FT_EXPECTED[fname]:
#         sys.exit(f"ERRO: esperava {FT_EXPECTED[fname]} trechos de fastText em {fname}, achei {n}")
#     return new_text


# def apply_patches():
#     for fname, reps in PATCHES.items():
#         path = os.path.join(ZDIR, fname)
#         src = open(path, encoding="utf-8", newline="").read()
#         crlf = "\r\n" in src
#         text = src.replace("\r\n", "\n")
#         for old, new in reps:
#             if new in text:
#                 continue
#             if old not in text:
#                 sys.exit(f"ERRO: trecho não encontrado em {fname}; o repositório mudou?\n---\n{old}")
#             text = text.replace(old, new, 1)
#         if fname in FT_EXPECTED:
#             text = patch_fasttext(fname, text)
#         if crlf:
#             text = text.replace("\n", "\r\n")
#         with open(path, "w", encoding="utf-8", newline="") as f:
#             f.write(text)
#     print("patches aplicados (ou já presentes)")


# # ---------------------------------------------------------------- utilidades do pipeline
# def tag_of(name):
#     """Nome interno usado pelo ZeroED nos arquivos e pastas (sem hífen, espaço ou símbolos)."""
#     t = re.sub(r"[^A-Za-z0-9_]", "", name)
#     if not t:
#         sys.exit(f"--name inválido: {name!r}")
#     return t


# def usage_file(a):
#     return os.path.join(ROOT, "logs", f"zeroed_{tag_of(a.name)}_usage.jsonl")


# def check_inputs(a):
#     for p in (a.dirty, a.clean):
#         if not os.path.isfile(p):
#             sys.exit(f"arquivo não encontrado: {p}")
#     if not os.path.isfile(os.path.join(ZDIR, "main.py")):
#         sys.exit(f"ZeroED não encontrado em {ZDIR}")


# # ---------------------------------------------------------------- subcomandos
# def prepare(a):
#     check_inputs(a)
#     tag = tag_of(a.name)
#     os.makedirs(os.path.join(ZDIR, "data"), exist_ok=True)
#     shutil.copy(a.dirty, os.path.join(ZDIR, "data", f"{tag}_error-01.csv"))
#     # O ZeroED acessa o clean pelos NOMES de coluna do dirty. Se os cabeçalhos diferem,
#     # o clean é regravado com o cabeçalho do dirty (por posição; valores intactos).
#     dirty_cols = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, nrows=0).columns
#     clean = pd.read_csv(a.clean, dtype=str, keep_default_na=False, low_memory=False)
#     assert len(clean.columns) == len(dirty_cols), "dirty e clean têm números de colunas diferentes"
#     clean.columns = dirty_cols
#     clean.to_csv(os.path.join(ZDIR, "data", f"{tag}_clean.csv"), index=False)
#     cfg = {
#         "model": {
#             "api_use": True, "pre_func_use": True, "related_attrs": True,
#             "distri_analysis": True, "guide_use": True, "func_use": True,
#             "n_method": a.n_method, "err_gen_use": True, "rel_top": 2, "run_info": "pipeline",
#         },
#         "read": {
#             "pre_func": False, "distri_analysis": False, "related_attrs": False,
#             "cluster": False, "guide": False, "error_checking": False, "func": False,
#             "err_gen": False, "read_in_batch": False,
#             "start_time": "01-01-00:00", "end_time": "01-01-00:00",
#         },
#         "data": {
#             "base_dir": ".", "err_rate_list": ["01"], "all_set_num": a.runs,
#             "datasets": [tag], "err_check_val_num_per_query": 20,
#             "result_dir": "pipeline",
#         },
#         "read_paths": {},
#     }
#     with open(os.path.join(ZDIR, "config_pipeline.yaml"), "w") as f:
#         yaml.safe_dump(cfg, f, sort_keys=False)
#     apply_patches()
#     print(f"config gerada: {ZDIR}/config_pipeline.yaml ({a.runs} execução(ões) de '{tag}')")


# def build_fasttext(a):
#     """Gera methods/zeroed/cc.en.<dim>.bin (reduzido) uma vez por dimensão, em um único processo."""
#     import fasttext
#     import fasttext.util
#     dim = len(pd.read_csv(a.dirty, nrows=0).columns)
#     out = os.path.join(ZDIR, f"cc.en.{dim}.bin")
#     if os.path.exists(out):
#         print(f"já existe: {out}")
#         return
#     full = os.path.join(ZDIR, "cc.en.300.bin")
#     if not os.path.exists(full):
#         sys.exit(f"Falta {full} (modelo fastText completo, ~7 GB). Crie um link simbólico para ele.")
#     print(f"reduzindo o fastText para {dim} dimensões (usa MUITA memória; feche outros programas)...")
#     model = fasttext.load_model(full)
#     fasttext.util.reduce_model(model, dim)
#     model.save_model(out + ".tmp")
#     os.replace(out + ".tmp", out)
#     print(f"gerado: {out}")


# def run(a):
#     check_inputs(a)
#     cfg_path = os.path.join(ZDIR, "config_pipeline.yaml")
#     if not os.path.exists(cfg_path):
#         sys.exit("Rode antes o subcomando 'prepare' (ou use 'all').")
#     cfg = yaml.safe_load(open(cfg_path))
#     if cfg["data"]["datasets"] != [tag_of(a.name)]:
#         sys.exit(f"config_pipeline.yaml é de {cfg['data']['datasets']}; rode 'prepare' para {tag_of(a.name)}")
#     dim = len(pd.read_csv(a.dirty, nrows=0).columns)
#     if not os.path.exists(os.path.join(ZDIR, f"cc.en.{dim}.bin")):
#         sys.exit(f"Falta cc.en.{dim}.bin. Rode antes o subcomando 'fasttext' (ou use 'all').")
#     for v in ("ZEROED_API_KEY", "ZEROED_BASE_URL", "ZEROED_MODEL"):
#         if not os.environ.get(v):
#             sys.exit(f"Defina {v} (no .env ou no ambiente).")
#     os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
#     uf = usage_file(a)
#     if os.path.exists(uf):
#         os.remove(uf)  # tokens deste dataset começam do zero a cada 'run'
#     env = dict(os.environ, ZEROED_USAGE_FILE=uf, ZEROED_PROCS=str(a.procs), PYTHONUNBUFFERED="1")
#     subprocess.run([sys.executable, "main.py", "--config", "config_pipeline.yaml"],
#                    cwd=ZDIR, env=env, check=True)


# def export(a):
#     check_inputs(a)
#     tag = tag_of(a.name)
#     tokens = {"prompt_tokens": 0, "completion_tokens": 0}
#     uf = usage_file(a)
#     if os.path.exists(uf):
#         for line in open(uf):
#             u = json.loads(line)
#             tokens["prompt_tokens"] += u["p"]
#             tokens["completion_tokens"] += u["c"]

#     # pastas: methods/zeroed/result/pipeline/<MM-DD> <tag>01-<n_method>-set<N>/
#     runs = glob.glob(os.path.join(ZDIR, "result", "pipeline", f"* {tag}01-*-set*"))
#     latest = {}
#     for r in runs:
#         if not os.path.exists(os.path.join(r, "detected.json")):
#             continue
#         n = r.rsplit("-set", 1)[1]
#         if n not in latest or os.path.getmtime(r) > os.path.getmtime(latest[n]):
#             latest[n] = r
#     if not latest:
#         sys.exit(f"nenhuma execução concluída (detected.json) para '{tag}'")

#     dirty = pd.read_csv(a.dirty, dtype=str, keep_default_na=False, low_memory=False)
#     for n in sorted(latest, key=int):
#         r = latest[n]
#         mat = np.zeros(dirty.shape, dtype=int)
#         skipped = 0
#         for i, j in json.load(open(os.path.join(r, "detected.json"))):
#             if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1]:
#                 mat[i, j] = 1
#             else:
#                 skipped += 1
#         out = os.path.join(a.outdir, a.name, f"zeroed_s{n}")
#         os.makedirs(out, exist_ok=True)
#         pd.DataFrame(mat, columns=dirty.columns).to_csv(os.path.join(out, "pred.csv"), index=False)
#         shutil.copy(os.path.join(r, "detected.json"), out)
#         total = None
#         tf = os.path.join(r, "0-time.txt")
#         if os.path.exists(tf):
#             for line in open(tf):
#                 if line.startswith("total:"):
#                     total = round(float(line.split(":")[1]), 1)
#         json.dump({"total_s": total, "n_sets_no_arquivo_de_tokens": len(latest), **tokens},
#                   open(os.path.join(out, "runtime.json"), "w"))
#         print(f"[zeroed] {a.name} set{n}: detectadas={int(mat.sum())} fora_dos_limites={skipped} "
#               f"tempo={total}s -> {out}/pred.csv\n   métricas do próprio ZeroED: {os.path.join(r, 'func_det_res.txt')}")
#     print("Tokens (soma de todas as execuções deste dataset):", tokens)


# def all_steps(a):
#     prepare(a)
#     build_fasttext(a)
#     run(a)
#     export(a)


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     sub = ap.add_subparsers(dest="cmd", required=True)
#     for name, fn in [("prepare", prepare), ("fasttext", build_fasttext), ("run", run),
#                      ("export", export), ("all", all_steps)]:
#         p = sub.add_parser(name)
#         p.add_argument("--name", required=True, help="rótulo do dataset (usado nas saídas)")
#         p.add_argument("--dirty", required=True, help="caminho do CSV sujo")
#         p.add_argument("--clean", required=True, help="caminho do CSV limpo")
#         p.add_argument("--runs", type=int, default=1, help="execuções (sets) por dataset")
#         p.add_argument("--procs", default="2", help="ZEROED_PROCS (processos paralelos)")
#         p.add_argument("--n-method", default="5%", help="proporção de linhas rotuladas pelo LLM")
#         p.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
#         p.set_defaults(fn=fn)
#     args = ap.parse_args()
#     args.dirty, args.clean = os.path.abspath(args.dirty), os.path.abspath(args.clean)
#     args.fn(args)

#!/usr/bin/env python3
"""
Runner genérico para o método ZeroED (Ni et al., ICDE 2025).
Recebe --dirty e --clean de qualquer caminho no sistema de arquivos,
configura automaticamente os caminhos exigidos pelo main.py do ZeroED
(data/<name>_error-01.csv e data/<name>_clean.csv), aplica os patches de exportação
e converte detected.json para outputs/<name>/zeroed_s<seed>/pred.csv.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZDIR = os.path.join(ROOT, "methods", "zeroed")

DEFAULT_CONFIG = {
    "model": {
        "api_use": True,
        "pre_func_use": True,
        "related_attrs": True,
        "distri_analysis": True,
        "guide_use": True,
        "func_use": True,
        "n_method": "5%",
        "err_gen_use": True,
        "rel_top": 2,
        "run_info": "pipeline",
    },
    "read": {
        "pre_func": False,
        "distri_analysis": False,
        "related_attrs": False,
        "cluster": False,
        "guide": False,
        "error_checking": False,
        "func": False,
        "err_gen": False,
        "read_in_batch": False,
        "start_time": "01-01-00:00",
        "end_time": "01-01-00:00",
    },
    "data": {
        "base_dir": ".",
        "err_rate_list": ["01"],
        "all_set_num": 1,
        "datasets": ["test"],
        "err_check_val_num_per_query": 20,
        "result_dir": "pipeline",
        "read_paths": {},
    },
}

PATCHES = {
    "prompt_gen.py": [
        (
            "def error_check_prompt(col_values, col_name):",
            "guide_gen_prompt = kb_gen_prompt  # PATCH\n\ndef error_check_prompt(col_values, col_name):",
        ),
    ],
    "main.py": [
        (
            "        measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n",
            "        measure_detect(clean_path, dirty_path, list(det_wrong_list), det_res_path)\n"
            "        with open(os.path.join(resp_path, 'detected.json'), 'w') as _f:  # PATCH\n"
            "            json.dump(sorted([int(i), all_attrs.index(a)] for i, a in det_wrong_list), _f)\n",
        ),
        (
            "feat_pool = mp.Pool()",
            "feat_pool = mp.Pool(int(os.environ.get('ZEROED_PROCS', '2')))  # PATCH",
        ),
    ],
}


def apply_patches():
    for fname, pairs in PATCHES.items():
        path = os.path.join(ZDIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        changed = False
        for old, new in pairs:
            if old in code and new not in code:
                code = code.replace(old, new)
                changed = True
        if changed:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[zeroed] Patch aplicado em {fname}")


def cmd_prepare(args):
    dataset = args.name
    if not os.path.isdir(ZDIR):
        sys.exit(f"[zeroed] ERRO: diretório {ZDIR} não encontrado.")

    dirty_src = os.path.abspath(os.path.expanduser(args.dirty))
    clean_src = os.path.abspath(os.path.expanduser(args.clean))
    for p in (dirty_src, clean_src):
        if not os.path.isfile(p):
            sys.exit(f"[zeroed] ERRO: arquivo não encontrado: {p}")

    data_root = os.path.join(ZDIR, "data")
    os.makedirs(data_root, exist_ok=True)

    # 1. Formato esperado por methods/zeroed/main.py:
    #    data/<dataset>_error-01.csv e data/<dataset>_clean.csv
    shutil.copy2(dirty_src, os.path.join(data_root, f"{dataset}_error-01.csv"))
    shutil.copy2(clean_src, os.path.join(data_root, f"{dataset}_clean.csv"))

    # 2. Formato alternativo (data/<dataset>/clean_dirty/) por compatibilidade
    alt_dir = os.path.join(data_root, dataset, "clean_dirty")
    os.makedirs(alt_dir, exist_ok=True)
    shutil.copy2(dirty_src, os.path.join(alt_dir, "dirty.csv"))
    shutil.copy2(clean_src, os.path.join(alt_dir, "clean.csv"))

    apply_patches()

    # Carrega config_pipeline.yaml ou config.yaml existente como base
    cfg = None
    for candidate in ("config_pipeline.yaml", "config.yaml"):
        cpath = os.path.join(ZDIR, candidate)
        if os.path.isfile(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict) and "data" in loaded and "model" in loaded:
                cfg = loaded
                break

    if cfg is None:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))

    cfg.setdefault("model", {})["api_use"] = True
    cfg.setdefault("data", {})["base_dir"] = "."
    cfg["data"]["err_rate_list"] = ["01"]
    cfg["data"]["all_set_num"] = 1
    cfg["data"]["datasets"] = [dataset]
    cfg["data"]["result_dir"] = "pipeline"
    cfg["dataset"] = dataset
    cfg.setdefault("llm_param", {})["seed"] = args.seed

    for target_cfg in ("config_pipeline.yaml", "config.yaml"):
        out_cfg = os.path.join(ZDIR, target_cfg)
        with open(out_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[zeroed] Dataset '{dataset}' preparado em {data_root}")


def cmd_fasttext(args):
    pipe_cfg = os.path.join(ZDIR, "config_pipeline.yaml")
    cfg = {}
    if os.path.isfile(pipe_cfg):
        with open(pipe_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    dim = int(cfg.get("fasttext_dim", 300))
    emb_full = os.path.join(ZDIR, "cc.en.300.bin")
    emb_out = os.path.join(ZDIR, f"cc.en.{dim}.bin")

    if os.path.isfile(emb_out):
        print(f"[zeroed] Modelo FastText encontrado: {emb_out}")
        return

    if not os.path.isfile(emb_full):
        print(
            f"[zeroed] AVISO: {emb_full} não encontrado em {ZDIR}. "
            "Se o feature.py do ZeroED exigir embeddings FastText, crie um symlink para cc.en.300.bin."
        )
        return

    import fasttext
    import fasttext.util

    print(f"[zeroed] Reduzindo {emb_full} de 300d para {dim}d...")
    ft = fasttext.load_model(emb_full)
    fasttext.util.reduce_model(ft, dim)
    ft.save_model(emb_out)


def cmd_run(args):
    api_key = os.environ.get("ZEROED_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (
        os.environ.get("ZEROED_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    model = (
        os.environ.get("ZEROED_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-4o-mini"
    )

    if not api_key:
        sys.exit(
            "[zeroed] ERRO: defina OPENAI_API_KEY (ou ZEROED_API_KEY) antes de rodar o ZeroED."
        )

    env = dict(os.environ)
    env["ZEROED_API_KEY"] = api_key
    env["ZEROED_BASE_URL"] = base_url
    env["ZEROED_MODEL"] = model
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_MODEL"] = model

    print(f"[zeroed] Executando main.py no ZeroED (modelo={model})...")
    ret = subprocess.call([sys.executable, "main.py", "--config", "config.yaml"], cwd=ZDIR, env=env)
    if ret != 0:
        sys.exit(f"[zeroed] ERRO: main.py do ZeroED falhou com código {ret}.")


def _find_detected_json(dataset: str) -> str:
    # 1. Caminho padrão de main.py: result/pipeline/<MM-DD> <dataset>01-.../detected.json
    patterns = [
        os.path.join(ZDIR, "result", "pipeline", f"*{dataset}*", "detected.json"),
        os.path.join(ZDIR, "result", "*", f"*{dataset}*", "detected.json"),
        os.path.join(ZDIR, "response", dataset, "temp_*", "detected.json"),
        os.path.join(ZDIR, "result", "**", "detected.json"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        sys.exit(
            f"[zeroed] ERRO: Nenhum arquivo detected.json encontrado para '{dataset}' em {ZDIR}/result."
        )
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def cmd_export(args):
    det_json = _find_detected_json(args.name)
    with open(det_json, "r", encoding="utf-8") as f:
        err_pairs = json.load(f)

    dirty_src = os.path.abspath(os.path.expanduser(args.dirty))
    df_dirty = pd.read_csv(dirty_src, dtype=str, keep_default_na=False)
    mask = np.zeros(df_dirty.shape, dtype=int)
    for r, c in err_pairs:
        if 0 <= int(r) < mask.shape[0] and 0 <= int(c) < mask.shape[1]:
            mask[int(r), int(c)] = 1

    out_dir = os.path.join(args.outdir, args.name, f"zeroed_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pred.csv")

    pd.DataFrame(mask, columns=df_dirty.columns).to_csv(out_csv, index=False)
    print(
        f"[zeroed] {args.name} seed={args.seed} detectadas={int(mask.sum())} "
        f"(de {det_json}) -> {out_csv}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Pipeline ZeroED genérico -> outputs/<name>/zeroed_s<seed>/pred.csv"
    )
    p.add_argument("command", nargs="?", default="all", choices=["prepare", "fasttext", "run", "export", "all"])
    p.add_argument("--name", required=True, help="Nome do dataset (ex.: test)")
    p.add_argument("--dirty", required=True, help="Caminho para o arquivo dirty.csv")
    p.add_argument("--clean", required=True, help="Caminho para o arquivo clean.csv")
    p.add_argument("--seed", type=int, default=0, help="Semente aleatória (padrão: 0)")
    p.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    args = p.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "fasttext":
        cmd_fasttext(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "all":
        cmd_prepare(args)
        cmd_fasttext(args)
        cmd_run(args)
        cmd_export(args)


if __name__ == "__main__":
    main()