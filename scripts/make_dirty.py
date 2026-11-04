#!/usr/bin/env python3
"""Gerador de sujeira sintética: a partir de um clean.csv produz um dirty.csv alinhado
(mesmo shape, mesma ordem de linhas/colunas), pronto para os runners e o evaluate.py.

Tipos de erro (cada célula sorteada recebe um tipo compatível com sua coluna):
  typo     – erro de digitação: troca por tecla vizinha, inserção, remoção, transposição
  missing  – valor apagado ou trocado por marcador de ausência ("", NULL, N/A, ...)
  format   – mesma informação em outro formato: datas, telefones, caixa, espaços, abreviações
  outlier  – colunas numéricas: valor fora da distribuição (escala, sinal, deslocamento)
  domain   – colunas categóricas: valor válido de outra linha (viola dependências, ex. ZIPCODE→BORO)

Saída em <outdir>/: clean.csv, dirty.csv, errors.csv (linha, coluna, tipo, clean, dirty) e meta.json.

Exemplo:
  python scripts/make_dirty.py --clean data/dataset/restaurants/clean.csv \
         --sample 2000 --rate 0.05 --seed 0
"""
import argparse
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

TYPES = ["typo", "missing", "format", "outlier", "domain"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--clean", required=True)
ap.add_argument("--outdir", help="padrão: <pasta do clean>/dirty_n<sample>_r<rate>_s<seed>")
ap.add_argument("--rate", type=float, default=0.05, help="fração das células não vazias a sujar")
ap.add_argument("--sample", type=int, help="amostrar N linhas antes de sujar (o clean.csv de saída é a amostra)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--types", default="typo=0.35,missing=0.15,format=0.2,outlier=0.1,domain=0.2",
                help="pesos por tipo, ex.: typo=1,missing=1 (tipos omitidos ficam desligados)")
ap.add_argument("--cols", help="só sujar estas colunas (separadas por vírgula)")
ap.add_argument("--exclude-cols", help="nunca sujar estas colunas (ex.: chaves)")
ap.add_argument("--max-per-row", type=int, default=0, help="máximo de células sujas por linha (0 = sem limite)")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
rnd = random.Random(args.seed)

weights = {}
for part in args.types.split(","):
    k, v = part.split("=")
    assert k.strip() in TYPES, f"tipo desconhecido: {k}"
    weights[k.strip()] = float(v)

df = pd.read_csv(args.clean, dtype=str, keep_default_na=False, low_memory=False)
if args.sample and args.sample < len(df):
    df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
cols = list(df.columns)
target_cols = args.cols.split(",") if args.cols else cols
if args.exclude_cols:
    target_cols = [c for c in target_cols if c not in args.exclude_cols.split(",")]
missing_cols = set(target_cols) - set(cols)
assert not missing_cols, f"colunas inexistentes: {missing_cols}"

# ---------- perfil das colunas ----------
DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def profile(s):
    v = s[s != ""]
    if len(v) == 0:
        return "empty"
    if v.str.match(DATE_RE).mean() > 0.95:
        return "date"
    if v.str.match(NUM_RE).mean() > 0.95:
        # códigos (CAMIS, PHONE, ZIPCODE, BIN...) também são numéricos; outlier só faz sentido em medidas
        is_code = v.str.len().nunique() == 1 or not v.str.contains(r"\.").any() and v.str.len().median() >= 5
        return "code" if is_code else "numeric"
    if v.nunique() <= max(200, 0.01 * len(v)):
        return "categorical"
    return "text"


kinds = {c: profile(df[c]) for c in cols}
domains = {c: df[c][df[c] != ""].unique() for c in cols}

# tipos aplicáveis por perfil
APPLICABLE = {
    "date": ["typo", "missing", "format"],
    "numeric": ["typo", "missing", "format", "outlier"],
    "code": ["typo", "missing", "format"],
    "categorical": ["typo", "missing", "format", "domain"],
    "text": ["typo", "missing", "format"],
}

# ---------- geradores de erro ----------
KEYBOARD = "qwertyuiop asdfghjkl zxcvbnm".split()
NEIGH = {}
for r, row in enumerate(KEYBOARD):
    for i, ch in enumerate(row):
        n = [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
        for rr in (r - 1, r + 1):
            if 0 <= rr < len(KEYBOARD):
                n += [KEYBOARD[rr][j] for j in (i - 1, i, i + 1) if 0 <= j < len(KEYBOARD[rr])]
        NEIGH[ch] = n
DIGIT_NEIGH = {d: [str((int(d) + k) % 10) for k in (-1, 1)] for d in "0123456789"}
MISSING_TOKENS = ["", "", "NULL", "N/A", "nan", "-", "?", "unknown"]
ABBREV = {"STREET": "ST", "AVENUE": "AVE", "ROAD": "RD", "BOULEVARD": "BLVD", "PLACE": "PL",
          "WEST": "W", "EAST": "E", "NORTH": "N", "SOUTH": "S", "DRIVE": "DR", "Manhattan": "MN",
          "Brooklyn": "BK", "Queens": "QN", "Bronx": "BX", "Staten Island": "SI",
          "Critical": "Crit", "Not Critical": "Not Crit", "American": "Amer."}


def keep_case(orig, ch):
    return ch.upper() if orig.isupper() else ch


def typo(v, kind):
    if not v:
        return v
    i = rnd.randrange(len(v))
    c = v[i]
    op = rnd.choice(["sub", "ins", "del", "swap"] if len(v) > 1 else ["sub", "ins"])
    if op == "swap":
        j = i + 1 if i + 1 < len(v) else i - 1
        a, b = sorted((i, j))
        return v[:a] + v[b] + v[a] + v[b + 1:]
    if op == "del":
        return v[:i] + v[i + 1:]
    if c.isdigit():
        new = rnd.choice(DIGIT_NEIGH[c])
    elif c.lower() in NEIGH:
        new = keep_case(c, rnd.choice(NEIGH[c.lower()]))
    else:
        new = keep_case(v[0], rnd.choice("abcdefghijklmnopqrstuvwxyz"))
    if op == "sub":
        return v[:i] + new + v[i + 1:]
    return v[:i] + new + v[i:]


def missing(v, kind):
    return rnd.choice(MISSING_TOKENS)


def fmt(v, kind):
    if kind == "date":
        m = DATE_RE.match(v)
        if m:
            mm, dd, yy = m.groups()
            return rnd.choice([f"{yy}-{mm}-{dd}", f"{dd}/{mm}/{yy}", f"{mm}-{dd}-{yy}",
                               f"{mm}/{dd}/{yy[2:]}", f"{int(mm)}/{int(dd)}/{yy}"])
    if kind == "code" and v.isdigit():
        if len(v) == 10:  # telefone
            return rnd.choice([f"({v[:3]}) {v[3:6]}-{v[6:]}", f"{v[:3]}-{v[3:6]}-{v[6:]}", f"+1{v}"])
        return rnd.choice([v + ".0", "0" + v, " " + v])
    if kind == "numeric":
        try:
            x = float(v)
            return rnd.choice([f"{x:.2f}", f"{x:.1e}", v.replace(".", ",") if "." in v else v + ".0"])
        except ValueError:
            pass
    opts = []
    for full, ab in ABBREV.items():
        if re.search(rf"\b{re.escape(full)}\b", v):
            opts.append(lambda v=v, f=full, a=ab: re.sub(rf"\b{re.escape(f)}\b", a, v, count=1))
    opts += [str.lower, str.upper, str.title, lambda s: " " + s, lambda s: s + "  ",
             lambda s: s.replace(" ", "  ", 1)]
    return rnd.choice(opts)(v)


def outlier(v, kind):
    try:
        x = float(v)
    except ValueError:
        return typo(v, kind)
    dec = len(v.split(".")[1]) if "." in v else 0
    y = rnd.choice([x * 10, x * 100, -abs(x) - 1, x + rnd.choice([-1, 1]) * (abs(x) + 1) * 5, 9999])
    return f"{y:.{dec}f}"


def domain(v, kind, col):
    dom = domains[col]
    if len(dom) < 2:
        return typo(v, kind)
    for _ in range(10):
        w = dom[rng.integers(len(dom))]
        if w != v:
            return w
    return typo(v, kind)


GEN = {"typo": typo, "missing": missing, "format": fmt, "outlier": outlier}

# ---------- sorteio das células ----------
vals = df.values  # objeto; cópia via astype abaixo
dirty = vals.copy()
cidx = [cols.index(c) for c in target_cols if kinds[c] != "empty"]
cand = np.argwhere(vals[:, cidx] != "")
cand[:, 1] = np.array(cidx)[cand[:, 1]]
n_err = int(round(args.rate * len(cand)))
order = rng.permutation(len(cand))

per_row = Counter()
log = []
for k in order:
    if len(log) >= n_err:
        break
    r, c = cand[k]
    if args.max_per_row and per_row[r] >= args.max_per_row:
        continue
    col = cols[c]
    kind = kinds[col]
    opts = [t for t in APPLICABLE[kind] if weights.get(t, 0) > 0]
    if not opts:
        continue
    w = np.array([weights[t] for t in opts])
    v = vals[r, c]
    for _ in range(5):  # garante que a célula realmente mudou
        t = opts[rng.choice(len(opts), p=w / w.sum())]
        new = domain(v, kind, col) if t == "domain" else GEN[t](v, kind)
        if new != v:
            break
    else:
        continue
    dirty[r, c] = new
    per_row[r] += 1
    log.append((int(r), col, t, v, new))

# ---------- saída ----------
tag = f"dirty_n{len(df)}_r{args.rate}_s{args.seed}"
outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.clean)), tag)
os.makedirs(outdir, exist_ok=True)
df.to_csv(os.path.join(outdir, "clean.csv"), index=False)
dirty_df = pd.DataFrame(dirty, columns=cols)
dirty_df.to_csv(os.path.join(outdir, "dirty.csv"), index=False)
pd.DataFrame(log, columns=["row", "column", "type", "clean", "dirty"]).to_csv(
    os.path.join(outdir, "errors.csv"), index=False)

# confere que o par relido bate com o log (o evaluate.py relê os CSVs)
rc = pd.read_csv(os.path.join(outdir, "clean.csv"), dtype=str, keep_default_na=False)
rd = pd.read_csv(os.path.join(outdir, "dirty.csv"), dtype=str, keep_default_na=False)
n_diff = int((rc.values != rd.values).sum())
assert rc.shape == rd.shape == df.shape and n_diff == len(log), (n_diff, len(log))

meta = dict(source=args.clean, shape=list(df.shape), seed=args.seed, rate=args.rate,
            weights=weights, n_errors=len(log), cell_error_rate=round(len(log) / df.size, 4),
            dirty_rows=len(per_row), column_kinds=kinds,
            by_type=dict(Counter(t for _, _, t, _, _ in log)),
            by_column=dict(Counter(c for _, c, _, _, _ in log)))
with open(os.path.join(outdir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"{outdir}: {len(log)} erros em {len(per_row)} linhas ({meta['cell_error_rate']:.2%} das células)")
print("por tipo:", meta["by_type"])
