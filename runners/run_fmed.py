#!/usr/bin/env python3
"""FM_ED: detecção de erros por prompting (Narayan et al., VLDB'23), adaptada ao pipeline."""
import argparse, concurrent.futures as cf, hashlib, json, os, random, sys, threading, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--name", required=True)
ap.add_argument("--dirty", required=True, help="tabela COMPLETA suja")
ap.add_argument("--clean", required=True, help="tabela COMPLETA limpa")
ap.add_argument("--n-eval", type=int, default=50, help="linhas avaliadas (protocolo do artigo: 50)")
ap.add_argument("--row-seed", type=int, default=0, help="semente das linhas avaliadas")
ap.add_argument("--k", type=int, default=10, help="demonstrações no prompt (0 = zero-shot)")
ap.add_argument("--demo-rows", type=int, default=2, help="tuplas rotuladas das quais saem as demonstrações")
ap.add_argument("--seed", type=int, default=0, help="semente das demonstrações")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--dry-run", action="store_true", help="mostra 2 prompts e sai, sem chamar a API")
ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
args = ap.parse_args()

load = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
dirty, clean = load(args.dirty), load(args.clean)
assert dirty.shape == clean.shape, f"tamanhos diferentes: {dirty.shape} vs {clean.shape}"
clean.columns = dirty.columns
cols = list(dirty.columns)

# linhas avaliadas: mesma amostragem do make_subsample.py (mesma semente => mesmas linhas)
eval_idx = dirty.sample(n=min(args.n_eval, len(dirty)), random_state=args.row_seed).sort_index().index
eval_set = set(eval_idx)
rest = [i for i in dirty.index if i not in eval_set]   # demonstrações nunca vêm das linhas avaliadas
rng = random.Random(args.seed)
is_err = lambda i, c: dirty.at[i, c] != clean.at[i, c]


def serialize(i):
    return " ".join(f"{c}: {dirty.at[i, c]}." for c in cols)


def question(i, c):
    return f"{serialize(i)}\nIs there an error in {c}: {dirty.at[i, c]}?"


demo_rows = rng.sample(rest, min(args.demo_rows, len(rest)))
pool_err = [(i, c) for i in demo_rows for c in cols if is_err(i, c)]
pool_ok = [(i, c) for i in demo_rows for c in cols if not is_err(i, c)]
n_err = min(len(pool_err), args.k // 2)
demos = rng.sample(pool_err, n_err) + rng.sample(pool_ok, min(len(pool_ok), args.k - n_err))
rng.shuffle(demos)
demo_text = "\n\n".join(f"{question(i, c)}\n{'Yes' if is_err(i, c) else 'No'}" for i, c in demos)
print(f"demonstrações: {len(demos)} (com erro: {n_err}), tiradas de {len(demo_rows)} tuplas rotuladas")
if args.k > 0 and n_err == 0:
    print("AVISO: nenhuma demonstração com erro; o modelo tende a responder só 'No'. "
          "Aumente --demo-rows ou use outra --seed.")


def build(i, c):
    return (demo_text + "\n\n" if demo_text else "") + question(i, c)


cells = [(i, c) for i in eval_idx for c in cols]
if args.dry_run:
    for i, c in cells[:2]:
        print("-" * 60)
        print(build(i, c))
    print(f"\n{len(cells)} chamadas ao LLM ({len(eval_idx)} linhas x {len(cols)} colunas)")
    sys.exit(0)

from openai import OpenAI

env = lambda *names: next((os.environ[n] for n in names if os.environ.get(n)), None)
api_key = env("FMED_API_KEY", "ZEROED_API_KEY")
base_url = env("FMED_BASE_URL", "ZEROED_BASE_URL") or "https://api.openai.com/v1"
model = env("FMED_MODEL", "ZEROED_MODEL")
if not api_key or not model:
    sys.exit("Defina FMED_API_KEY e FMED_MODEL (ou ZEROED_*) no .env")
client = OpenAI(api_key=api_key, base_url=base_url)

out = os.path.join(args.outdir, args.name, f"fmed_s{args.seed}")
os.makedirs(out, exist_ok=True)
cache_path = os.path.join(out, "cache.jsonl")
cache, lock = {}, threading.Lock()
if os.path.exists(cache_path):
    for line in open(cache_path, encoding="utf-8"):
        r = json.loads(line)
        cache[r["k"]] = r["a"]
tokens = {"prompt_tokens": 0, "completion_tokens": 0}
SYSTEM = "Answer with Yes or No only."


def ask(prompt):
    k = hashlib.md5((model + SYSTEM + prompt).encode()).hexdigest()
    if k in cache:
        return cache[k]
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=5,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            ans = (r.choices[0].message.content or "").strip()
            with lock:
                cache[k] = ans
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "a": ans}) + "\n")
                if r.usage:
                    tokens["prompt_tokens"] += r.usage.prompt_tokens
                    tokens["completion_tokens"] += r.usage.completion_tokens
            return ans
        except Exception as e:
            wait = 2 ** attempt
            print(f"erro ({e.__class__.__name__}); nova tentativa em {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("a API falhou 6 vezes seguidas; rode de novo (o cache preserva o progresso)")


t0 = time.time()
answers = {}
with cf.ThreadPoolExecutor(args.workers) as ex:
    futs = {ex.submit(ask, build(i, c)): (i, c) for i, c in cells}
    for n, f in enumerate(cf.as_completed(futs), 1):
        answers[futs[f]] = f.result()
        if n % 100 == 0:
            print(f"{n}/{len(cells)}", flush=True)
elapsed = time.time() - t0

mat = np.zeros((len(eval_idx), len(cols)), dtype=int)
unclear = 0
for r, i in enumerate(eval_idx):
    for j, c in enumerate(cols):
        a = answers[(i, c)].lower()
        mat[r, j] = int(a.startswith("yes"))
        unclear += not (a.startswith("yes") or a.startswith("no"))   # fora do padrão: conta como "No"

pd.DataFrame(mat, columns=cols).to_csv(os.path.join(out, "pred.csv"), index=False)
dirty.loc[eval_idx].to_csv(os.path.join(out, "eval_dirty.csv"), index=False)
clean.loc[eval_idx].to_csv(os.path.join(out, "eval_clean.csv"), index=False)
json.dump({"model": model, "k": args.k, "demos_com_erro": n_err, "n_eval": len(eval_idx),
           "chamadas": len(cells), "respostas_fora_do_padrao": unclear, "total_s": round(elapsed, 1),
           **tokens}, open(os.path.join(out, "runtime.json"), "w"))
print(f"[fmed] {args.name} seed={args.seed} marcadas={int(mat.sum())} de {mat.size} "
      f"fora_do_padrao={unclear} tempo={elapsed:.0f}s tokens={tokens} -> {out}/pred.csv")