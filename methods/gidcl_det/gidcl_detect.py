#!/usr/bin/env python3
"""
GIDCL, detecção de erros (fluxo criador-crítico), reimplementada a partir do artigo
(Yan et al., SIGMOD 2024). Entradas: tabela suja, células das tuplas rotuladas e clusters.
O clean completo NUNCA é lido por este script.

Saídas em --outdir:
  pred_matrix.npy     detecção do M_det (a oficial)         pred_functions.npy  só as funções F_det
  pred_union.npy      M_det OU F_det                         probs.npy           probabilidade de erro por célula
  functions.json      funções geradas pelo LLM               report.json         log das rodadas, tempo, tokens
Requer no ambiente: OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_KEY.
"""
import argparse
import json
import os
import random
import re
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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dirty", required=True)
ap.add_argument("--labeled", required=True, help="labeled_cells.csv (row,column,dirty,clean)")
ap.add_argument("--clusters", help="clusters.csv (row,cluster)")
ap.add_argument("--outdir", required=True)
ap.add_argument("--lm", default="roberta-base")
ap.add_argument("--theta", type=float, default=0.85, help="F1 mínimo para aceitar uma função (artigo: 0.85)")
ap.add_argument("--iter-max", type=int, default=4, help="refinamentos por função (artigo: 4)")
ap.add_argument("--rounds", type=int, default=3, help="máximo de rodadas criador-crítico")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--infer-batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--pseudo-per-col", type=int, default=20)
ap.add_argument("--coreset-n", type=int, default=200, help="células limpas confiáveis adicionadas por rodada")
ap.add_argument("--col-ctx-max", type=int, default=20, help="valores distintos do cluster no contexto de coluna")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

TABLE = "table"
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
_client = None


# ------------------------------------------------------------------ LLM
def llm(messages):
    global _client
    from openai import OpenAI
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    for attempt in range(6):
        try:
            r = _client.chat.completions.create(model=os.environ["OPENAI_MODEL"], messages=messages)
            USAGE["calls"] += 1
            if r.usage:
                USAGE["prompt_tokens"] += r.usage.prompt_tokens
                USAGE["completion_tokens"] += r.usage.completion_tokens
            return r.choices[0].message.content or ""
        except Exception as e:
            wait = 2 ** attempt
            print(f"erro do LLM ({e.__class__.__name__}); nova tentativa em {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("o LLM falhou 6 vezes seguidas")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


# ------------------------------------------------------------------ execução isolada de funções do LLM
WORKER = r'''
import sys, json, builtins, re, random, string, math
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
except Exception:
    pass
req = json.load(sys.stdin)
random.seed(req.get("seed", 0))
ALLOWED = {"re": re, "random": random, "string": string, "math": math}
def _imp(name, *a, **k):
    base = name.split(".")[0]
    if base in ALLOWED:
        return ALLOWED[base]
    raise ImportError("import não permitido: " + name)
names = ["len", "range", "str", "int", "float", "bool", "min", "max", "sum", "abs", "any", "all", "enumerate",
         "zip", "sorted", "set", "frozenset", "list", "dict", "tuple", "isinstance", "round", "ord", "chr",
         "reversed", "map", "filter", "repr", "divmod", "pow", "iter", "next", "format", "Exception",
         "ValueError", "TypeError", "IndexError", "KeyError", "AttributeError"]
safe = {n: getattr(builtins, n) for n in names}
safe["__import__"] = _imp
ns = {"__builtins__": safe, "__name__": "llm_fn"}
ns.update(ALLOWED)
exec(req["code"], ns)
fn = ns[req["entry"]]
out = []
for x in req["inputs"]:
    try:
        out.append(fn(x))
    except Exception:
        out.append(None)
print(json.dumps(out, default=str))
'''


def sandbox(code, entry, inputs, seed=0, timeout=30):
    try:
        p = subprocess.run([sys.executable, "-c", WORKER], capture_output=True, text=True, timeout=timeout,
                           input=json.dumps({"code": code, "entry": entry, "inputs": inputs, "seed": seed}),
                           env={"PATH": os.environ.get("PATH", "")})
    except subprocess.TimeoutExpired:
        return None, "tempo esgotado"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["erro"])[-1][:200]
    try:
        return json.loads(p.stdout), None
    except Exception:
        return None, "saída inválida"


def eval_detect(code, cells):
    """cells = [(valor, é_sujo)]. Score = F1 da classe 'sujo' (ou acurácia se não há sujos)."""
    out, err = sandbox(code, "detect", [v for v, _ in cells])
    if out is None:
        return 0.0, {"error": err, "wrong": []}
    pred = [bool(o) for o in out]
    truth = [t for _, t in cells]
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum((not p) and t for p, t in zip(pred, truth))
    tn = len(truth) - tp - fp - fn
    score = (2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0) if tp + fn > 0 else tn / max(1, len(truth))
    wrong = [(v, t) for (v, t), p in zip(cells, pred) if p != t]
    return score, {"error": None, "wrong": wrong}


# ------------------------------------------------------------------ prompts (baseados no exemplo do README; os do artigo estão no apêndice)
def fl(xs, k):
    return "[" + ", ".join(repr(x) for x in xs[:k]) + "]"


def prompt_first(col, pairs, clean_only, values):
    if pairs:
        s = (f"The input\n\n{fl(pairs, 12)}\n\nare [clean,dirty] cell pairs from table {TABLE} column {col}, "
             f"and {fl(values, 25)} are examples of all cells from this column.")
    else:
        s = (f"The labeled tuples contain no dirty cell in column {col} of table {TABLE}. {fl(values, 25)} are "
             f"examples of all cells from this column.")
    if clean_only:
        s += f" These cells are known to be clean: {fl(clean_only, 15)}."
    return (s + " Please conclude a general pattern for dirty and clean cells, and write a general function with "
            "regular expression to detect whether a given cell is dirty or not. Name the function `detect(cell)`: "
            "it receives the cell as a string and returns True if the cell is dirty and False if it is clean. "
            "Use only the Python standard library (re). Reply with a single ```python code block and no explanation.")


def prompt_refine(wrong, err):
    if err:
        return (f"Your code failed ({err}). Please fix it and rewrite the function `detect(cell)` following the same "
                "rules. Reply with a single ```python code block and no explanation.")
    items = "; ".join(f"{v!r} is {'dirty' if t else 'clean'}" for v, t in wrong[:15])
    return ("The function misclassified these cells: " + items + ". Please conclude a general pattern for dirty and "
            "clean cells, regarding the wrongly detected cells, and rewrite the function `detect(cell)` following "
            "the same rules. Reply with a single ```python code block and no explanation.")


PROMPT_GEN = ("Now write a function `corrupt(cell)` that receives a CLEAN value (string) and returns a DIRTY version "
              "of it, imitating the kinds of errors seen in this column, so that `detect` flags the result as dirty. "
              "It may use the random module. Reply with a single ```python code block and no explanation.")


# ------------------------------------------------------------------ dados
def main():
    miss = [v for v in ("OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY") if not os.environ.get(v)]
    if miss:
        sys.exit("Defina no .env: " + ", ".join(miss))
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    t_start = time.time()
    rnd = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    D = pd.read_csv(args.dirty, dtype=str, keep_default_na=False, low_memory=False)
    L = pd.read_csv(args.labeled, dtype=str, keep_default_na=False)
    cols = list(D.columns)
    n, m = D.shape
    col_idx = {c: j for j, c in enumerate(cols)}
    V = D.values.tolist()
    if args.clusters:
        cdf = pd.read_csv(args.clusters)
        assert len(cdf) == n, "clusters.csv com número de linhas diferente do dirty"
        C = [int(x) for x in cdf["cluster"]]
    else:
        C = [0] * n
    groups = {}
    for i, c in enumerate(C):
        groups.setdefault(c, []).append(i)

    lab_cells = [(int(r), col_idx[c], d, cl) for r, c, d, cl in zip(L["row"], L["column"], L["dirty"], L["clean"])]
    lab_rows = sorted({i for i, _, _, _ in lab_cells})
    lab_set = set(lab_rows)
    clean_row = {i: [None] * m for i in lab_rows}
    for i, j, _, cl in lab_cells:
        clean_row[i][j] = cl
    print(f"tabela {n}x{m} | {len(lab_rows)} tuplas rotuladas | {len(groups)} clusters", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("AVISO: sem GPU; o treino será muito lento.", flush=True)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(args.lm)
    model = AutoModelForSequenceClassification.from_pretrained(args.lm, num_labels=2).to(device)

    # ---------------- serialização (artigo, Seção 5): contexto de linha e de coluna
    def row_text(vals):
        return " ".join(f"[COL] {c} [VAL] {v}" for c, v in zip(cols, vals))

    colctx_cache = {}

    def colctx(cl, j):
        key = (cl, j)
        if key not in colctx_cache:
            vals = pd.Series([V[i][j] for i in groups[cl]]).value_counts().index.tolist()[:args.col_ctx_max]
            colctx_cache[key] = f"[COL] {cols[j]} " + " ".join(f"[VAL] {v}" for v in vals)
        return colctx_cache[key]

    def inst(rowvals, cl, j, v, y):
        rv = list(rowvals)
        rv[j] = v
        return [(row_text(rv), f"[COL] {cols[j]} [VAL] {v}", y), (colctx(cl, j), f"[VAL] {v}", y)]

    # ---------------- critic: treino e inferência
    def train(data, epochs):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = epochs * ((len(data) + args.batch - 1) // args.batch)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        losses = []
        model.train()
        for ep in range(epochs):
            rnd.shuffle(data)
            tot = 0.0
            for s in range(0, len(data), args.batch):
                b = data[s:s + args.batch]
                enc = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=args.max_len,
                          padding=True, return_tensors="pt").to(device)
                y = torch.tensor([x[2] for x in b], device=device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(**enc).logits
                loss = F.cross_entropy(logits.float(), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
            losses.append(round(tot / len(data), 4))
            print(f"  época {ep + 1}/{epochs} perda={losses[-1]}", flush=True)
        return losses

    @torch.no_grad()
    def predict_pairs(pairs):
        uniq, idx = {}, []
        for p in pairs:
            idx.append(uniq.setdefault((p[0], p[1]), len(uniq)))
        keys = list(uniq)
        order = sorted(range(len(keys)), key=lambda t: len(keys[t][0]) + len(keys[t][1]))
        probs = np.zeros(len(keys), dtype=np.float32)
        model.eval()
        for s in range(0, len(order), args.infer_batch):
            b = order[s:s + args.infer_batch]
            enc = tok([keys[t][0] for t in b], [keys[t][1] for t in b], truncation=True,
                      max_length=args.max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(**enc).logits
            probs[b] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        return probs[np.array(idx)]

    def predict_table(chunk=1000):
        P = np.zeros((n, m), dtype=np.float32)
        for s in range(0, n, chunk):
            rows = range(s, min(n, s + chunk))
            pairs = []
            for i in rows:
                a = row_text(V[i])
                for j in range(m):
                    v = V[i][j]
                    pairs.append((a, f"[COL] {cols[j]} [VAL] {v}"))
                    pairs.append((colctx(C[i], j), f"[VAL] {v}"))
            P[s:s + len(rows)] = predict_pairs(pairs).reshape(len(rows), m, 2).mean(axis=2)
            print(f"  inferência {min(n, s + chunk)}/{n} linhas", flush=True)
        return P

    # ---------------- creator: função de detecção (com refinamento) e de geração, por coluna
    def creator(j, extra):
        col = cols[j]
        pairs = [(cl, d) for i, jj, d, cl in lab_cells if jj == j and d != cl]
        extra_vals = [V[i][j] for i, jj in extra if jj == j][:30]
        clean_only = [d for i, jj, d, cl in lab_cells if jj == j and d == cl] + extra_vals
        cells = [(d, d != cl) for i, jj, d, cl in lab_cells if jj == j] + [(v, False) for v in extra_vals]
        vc = pd.Series([r[j] for r in V]).value_counts()
        values = vc.index[:15].tolist() + rnd.sample(vc.index[15:].tolist(), min(10, max(0, len(vc) - 15)))
        messages = [{"role": "user", "content": prompt_first(col, pairs, clean_only, values)}]
        score, detail, reply, code, it = 0.0, {"error": None, "wrong": []}, "", "", 0
        for it in range(args.iter_max + 1):
            reply = llm(messages)
            code = extract_code(reply)
            score, detail = eval_detect(code, cells)
            if score >= args.theta or it == args.iter_max:
                break
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": prompt_refine(detail["wrong"], detail["error"])}]
        res = {"column": col, "detect": code, "score": round(score, 3), "accepted": score >= args.theta,
               "refinements": it, "gen": None}
        if res["accepted"]:
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": PROMPT_GEN}]
            res["gen"] = extract_code(llm(messages))
        return res

    def make_pseudo(j, res, extra):
        sources = [(("L", i), clean_row[i], C[i], clean_row[i][j]) for i in lab_rows]
        sources += [(("C", i), V[i], C[i], V[i][j]) for i, jj in extra if jj == j]
        picks = [rnd.choice(sources) for _ in range(args.pseudo_per_col)]
        out, err = sandbox(res["gen"], "corrupt", [s[3] for s in picks], seed=args.seed + j)
        if out is None:
            return {}
        valid = [(s, o) for s, o in zip(picks, out) if isinstance(o, str) and o != s[3]]
        if not valid:
            return {}
        flags, _ = sandbox(res["detect"], "detect", [o for _, o in valid])
        if flags is None:
            return {}
        return {(s[0], j, o): (s[1], s[2], j, o) for (s, o), f in zip(valid, flags) if f}

    # ---------------- laço criador-crítico (Fig. 3 do artigo)
    pseudo, extra, extra_set, funcs, prev_sig, rounds_log, P = {}, [], set(), {}, None, [], None
    per_col = max(1, args.coreset_n // m)
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        print(f"\n=== Rodada {r}: criador ===", flush=True)
        funcs = {}
        for j in range(m):
            try:
                funcs[j] = creator(j, extra)
            except Exception as e:
                print(f"  [{cols[j]}] falhou: {e}", flush=True)
                funcs[j] = {"column": cols[j], "detect": "", "score": 0.0, "accepted": False, "refinements": 0, "gen": None}
                continue
            print(f"  [{cols[j]}] F1={funcs[j]['score']} aceita={funcs[j]['accepted']} refinamentos={funcs[j]['refinements']}", flush=True)
            if funcs[j]["accepted"] and funcs[j]["gen"]:
                pseudo.update(make_pseudo(j, funcs[j], extra))

        print(f"=== Rodada {r}: crítico ===", flush=True)
        data = []
        for i, j, d, cl in lab_cells:
            data += inst(V[i], C[i], j, d, int(d != cl))
            if d != cl:
                data += inst(clean_row[i], C[i], j, cl, 0)
        for rowvals, cl, j, v in pseudo.values():
            data += inst(rowvals, cl, j, v, 1)
        for i, j in extra:
            data += inst(V[i], C[i], j, V[i][j], 0)
        pos = sum(x[2] for x in data) / len(data)
        print(f"  instâncias={len(data)} fração de erros={pos:.2f} pseudo={len(pseudo)} coreset={len(extra)}", flush=True)
        losses = train(data, args.epochs)

        P = predict_table()
        new = []
        for j in range(m):                                   # coreset: células limpas com maior confiança
            conf = 1.0 - P[:, j] + 1e-6 * np.random.rand(n)
            k = 0
            for i in np.argsort(-conf):
                if conf[i] < 0.9 or k >= per_col:
                    break
                if int(i) in lab_set or (int(i), j) in extra_set:
                    continue
                new.append((int(i), j))
                k += 1
        extra += new
        extra_set |= set(new)
        sig = {j: f["detect"].strip() for j, f in funcs.items() if f["accepted"]}
        rounds_log.append({"round": r, "accepted": sorted(cols[j] for j in sig), "pseudo": len(pseudo),
                           "novas_celulas_coreset": len(new), "fracao_erros_treino": round(pos, 3),
                           "perda": losses, "marcadas_pelo_modelo": int((P >= 0.5).sum()),
                           "segundos": round(time.time() - t0)})
        if sig == prev_sig:
            print("funções inalteradas: fim do laço", flush=True)
            break
        prev_sig = sig

    # ---------------- saídas
    pred = (P >= 0.5).astype(np.int8)
    fm = np.zeros((n, m), dtype=np.int8)
    for j, f in funcs.items():
        if not f["accepted"]:
            continue
        col_vals = [r[j] for r in V]
        uniq = sorted(set(col_vals))
        out, _ = sandbox(f["detect"], "detect", uniq, timeout=120)
        if out is not None:
            flag = {u: bool(o) for u, o in zip(uniq, out)}
            fm[:, j] = [flag[v] for v in col_vals]
    np.save(os.path.join(args.outdir, "pred_matrix.npy"), pred)
    np.save(os.path.join(args.outdir, "pred_functions.npy"), fm)
    np.save(os.path.join(args.outdir, "pred_union.npy"), np.maximum(pred, fm))
    np.save(os.path.join(args.outdir, "probs.npy"), P)
    json.dump({cols[j]: f for j, f in funcs.items()}, open(os.path.join(args.outdir, "functions.json"), "w"),
              indent=1, ensure_ascii=False)
    json.dump({"modelo_llm": os.environ["OPENAI_MODEL"], "lm": args.lm, "args": vars(args), "uso_llm": USAGE,
               "rodadas": rounds_log, "marcadas": int(pred.sum()), "marcadas_funcoes": int(fm.sum()),
               "total_s": round(time.time() - t_start)},
              open(os.path.join(args.outdir, "report.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n[gidcl_detect] marcadas={int(pred.sum())} de {pred.size} | funções={int(fm.sum())} | "
          f"LLM: {USAGE['calls']} chamadas, {USAGE['prompt_tokens'] + USAGE['completion_tokens']} tokens | "
          f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
