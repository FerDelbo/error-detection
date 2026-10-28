#!/usr/bin/env python3
"""
Runner genérico para FM_ED / FMED (Narayan et al., VLDB 2023 - HazyResearch/fm_data_tasks).
Avalia 100% das linhas da tabela (--n-eval 0 por padrão), constrói demonstrações k-shot
a partir de --demo-rows tuplas rotuladas via --clean e executa inferência paralela
via API da OpenAI com cache local em JSONL.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FM_REPO = os.path.join(ROOT, "methods", "fm_data_tasks")
if os.path.isdir(FM_REPO) and FM_REPO not in sys.path:
    sys.path.insert(0, FM_REPO)


def load_dotenv(path):
    if not os.path.isfile(path):
        return

    with open(path, encoding="utf-8") as dotenv:
        for line_number, raw_line in enumerate(dotenv, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()

            key, separator, value = line.partition("=")
            key = key.strip()
            if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Formato inválido no .env, linha {line_number}.")

            value = value.strip()
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError(
                        f"Valor entre aspas não fechado no .env, linha {line_number}."
                    )
                value = value[1:-1]
            else:
                value = re.sub(r"\s+#.*$", "", value).rstrip()

            os.environ.setdefault(key, value)


load_dotenv(os.path.join(ROOT, ".env"))


def main():
    ap = argparse.ArgumentParser(description="FM_ED (detecção na tabela completa) -> pred.csv")
    ap.add_argument("--name", required=True, help="Rótulo do dataset")
    ap.add_argument("--dirty", required=True, help="Caminho para o arquivo dirty.csv (qualquer local)")
    ap.add_argument("--clean", required=True, help="Caminho para o arquivo clean.csv (qualquer local)")
    ap.add_argument(
        "--n-eval",
        type=int,
        default=0,
        help="Número de linhas avaliadas (0 = avaliar 100%% da tabela; padrão: 0)",
    )
    ap.add_argument("--row-seed", type=int, default=0, help="Semente caso --n-eval > 0")
    ap.add_argument("-k", "--k", type=int, default=10, help="Demonstrações no prompt (0 = zero-shot)")
    ap.add_argument(
        "--demo-rows",
        type=int,
        default=2,
        help="Tuplas rotuladas de onde saem as demonstrações few-shot (padrão: 2)",
    )
    ap.add_argument("--seed", type=int, default=0, help="Semente das demonstrações")
    ap.add_argument("--workers", type=int, default=8, help="Workers paralelos para chamadas à API OpenAI")
    ap.add_argument(
        "--model",
        default=os.environ.get("FMED_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        help="Modelo da API OpenAI (padrão: FMED_MODEL / OPENAI_MODEL / gpt-4o-mini)",
    )
    ap.add_argument(
        "--base-url",
        default=(
            os.environ.get("FMED_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ),
        help="Endpoint base compatível com OpenAI",
    )
    ap.add_argument("--dry-run", action="store_true", help="Mostra 2 prompts e encerra sem chamar a API")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()

    api_key = (
        os.environ.get("FMED_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ZEROED_API_KEY")
    )
    if not api_key and not args.dry_run:
        sys.exit(
            "[fmed] ERRO: defina OPENAI_API_KEY (ou FMED_API_KEY) no ambiente antes de executar."
        )

    dirty_path = os.path.abspath(os.path.expanduser(args.dirty))
    clean_path = os.path.abspath(os.path.expanduser(args.clean))
    for p in (dirty_path, clean_path):
        if not os.path.isfile(p):
            sys.exit(f"[fmed] ERRO: arquivo não encontrado: {p}")

    load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
    dirty, clean = load(dirty_path), load(clean_path)
    assert dirty.shape == clean.shape, f"Tamanhos diferentes: {dirty.shape} vs {clean.shape}"
    clean.columns = dirty.columns
    cols = list(dirty.columns)

    rng = random.Random(args.seed)
    is_err = lambda i, c: dirty.at[i, c] != clean.at[i, c]

    def serialize(i: int) -> str:
        return " ".join(f"{c}: {dirty.at[i, c]}." for c in cols)

    def question(i: int, c: str) -> str:
        return f"{serialize(i)}\nIs there an error in {c}: {dirty.at[i, c]}?"

    # Seleção das tuplas de demonstração (k-shot) e das linhas de avaliação
    all_indices = list(dirty.index)
    if args.n_eval and 0 < args.n_eval < len(dirty):
        eval_rows = list(
            dirty.sample(n=args.n_eval, random_state=args.row_seed).sort_index().index
        )
        eval_set = set(eval_rows)
        candidate_demo_rows = [i for i in all_indices if i not in eval_set]
    else:
        # Avaliação na tabela inteira (100% das linhas)
        eval_rows = all_indices
        candidate_demo_rows = all_indices

    demo_rows = rng.sample(candidate_demo_rows, min(args.demo_rows, len(candidate_demo_rows)))
    demo_text = ""
    if args.k > 0 and demo_rows:
        pool_err = [(i, c) for i in demo_rows for c in cols if is_err(i, c)]
        pool_ok = [(i, c) for i in demo_rows for c in cols if not is_err(i, c)]
        n_err = min(len(pool_err), args.k // 2)
        n_ok = min(len(pool_ok), args.k - n_err)
        demos = rng.sample(pool_err, n_err) + rng.sample(pool_ok, n_ok)
        rng.shuffle(demos)
        demo_text = "\n\n".join(
            f"{question(i, c)}\n{'Yes' if is_err(i, c) else 'No'}" for i, c in demos
        )
        print(
            f"[fmed] Demonstrações k-shot: {len(demos)} (com erro: {n_err}, limpas: {n_ok}) "
            f"extraídas de {len(demo_rows)} tuplas."
        )

    system_msg = (
        "You are a tabular data error detection system. "
        "Given a tuple from a database and a target attribute, answer 'Yes' if the attribute "
        "value contains an error or noise, and 'No' if it is correct. "
        "Answer with ONLY 'Yes' or 'No'."
    )

    eval_pairs = [(i, c) for i in eval_rows for c in cols]
    if args.dry_run:
        for i, c in eval_pairs[:2]:
            user_prompt = f"{demo_text}\n\n{question(i, c)}" if demo_text else question(i, c)
            print("=== PROMPT EXEMPLO ===")
            print(user_prompt)
        return

    outdir = os.path.join(args.outdir, args.name, f"fmed_s{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, "cache.jsonl")

    # Carrega cache existente para retomada segura sem custo extra de API
    cache = {}
    if os.path.isfile(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec["ans"]
                except Exception:
                    pass

    cache_lock = threading.Lock()
    base_url = args.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def query_cell(pair):
        i, c = pair
        user_prompt = f"{demo_text}\n\n{question(i, c)}" if demo_text else question(i, c)
        key = hashlib.sha256(f"{args.model}|{system_msg}|{user_prompt}".encode("utf-8")).hexdigest()
        if key in cache:
            return i, c, cache[key], False

        payload = json.dumps({
            "model": args.model,
            "temperature": 0.0,
            "max_tokens": 5,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(f"{base_url}/chat/completions", data=payload, headers=headers)
        ans = ""
        failed = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    ans = res["choices"][0]["message"]["content"].strip()
                    break
            except Exception as e:
                if attempt == 2:
                    failed = True
                time.sleep(1.5 * (attempt + 1))

        if not failed and ans:
            with cache_lock:
                cache[key] = ans
                with open(cache_path, "a", encoding="utf-8") as cf_out:
                    cf_out.write(json.dumps({"key": key, "i": int(i), "col": c, "ans": ans}) + "\n")
        return i, c, ans, failed

    print(
        f"[fmed] Avaliando {len(eval_pairs)} células ({len(eval_rows)} linhas x {len(cols)} colunas) "
        f"com modelo={args.model} e workers={args.workers}..."
    )
    pred = pd.DataFrame(0, index=dirty.index, columns=cols, dtype=int)
    ok_count, err_count, fail_count = 0, 0, 0
    start_time = time.time()

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(query_cell, pair) for pair in eval_pairs]
        for idx, fut in enumerate(cf.as_completed(futures), 1):
            i, c, ans, failed = fut.result()
            if failed:
                fail_count += 1
            elif ans.lower().startswith("yes"):
                pred.at[i, c] = 1
                err_count += 1
            elif ans:
                ok_count += 1

            if idx % 100 == 0 or idx == len(eval_pairs):
                elapsed = time.time() - start_time
                print(
                    f"  [fmed] {idx}/{len(eval_pairs)} células em {elapsed:.1f}s "
                    f"(erros={err_count}, falhas={fail_count})",
                    flush=True,
                )

    out_csv = os.path.join(outdir, "pred.csv")
    pred.to_csv(out_csv, index=False)
    print(f"[fmed] Matriz salva em: {out_csv} (erros={err_count}, limpos={ok_count}, falhas={fail_count})")


if __name__ == "__main__":
    main()