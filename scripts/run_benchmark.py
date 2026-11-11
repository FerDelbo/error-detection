#!/usr/bin/env python3
"""Roda Raha, dBoost, ZeroED e GIDCL numa mesma base, em todas as combinações de configuração,
e grava um CSV com modelo, precisão, recall, F1 e a configuração usada.

O pipeline de cada método é o mesmo dos scripts/run_*.sh (runners/ + evaluate.py); a diferença é
que cada configuração ganha sua própria pasta de saída, então nada é sobrescrito:
  outputs/<nome>/bench/<método>/<id da configuração>/...

Uso:
  python scripts/run_benchmark.py --name restaurants_2k \
         --dirty data/dataset/restaurants/dirty_n2000_r0.05_s0/dirty.csv \
         --clean data/dataset/restaurants/dirty_n2000_r0.05_s0/clean.csv

  --methods raha,dboost          só alguns métodos
  --grid minha_grade.json        troca a grade padrão (mesmo formato de DEFAULT_GRID; ver --show-grid)
  --resume                       pula as configurações que já têm status=ok no CSV
  --dry-run                      só mostra os comandos

Os pythons de cada método vêm de ~/miniconda3/envs/{raha,zeroed,gidcl} (ou de --py-raha/--py-zeroed/--py-gidcl).
ZeroED usa a API de LLM do .env: cada configuração custa tokens.
"""
import argparse
import csv
import glob
import itertools
import json
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = os.path.expanduser("~/miniconda3/envs")

# Cada método: lista de valores por parâmetro; o produto cartesiano dá as configurações.
DEFAULT_GRID = {
    "raha": {"budget": [2, 5, 10, 20], "seed": [0, 1, 2]},
    "dboost": [  # lista de sub-grades (gaussian e histogram têm parâmetros diferentes)
        {"model": ["gaussian"], "threshold": [1.0, 1.5, 2.0, 2.5, 3.0]},
        {"model": ["histogram"], "peak": [0.1, 0.5, 0.9], "outlier": [0.1, 0.5, 0.9]},
    ],
    "zeroed": {"n_method": ["5%", "10%"], "ft_dim": [None]},
    "gidcl": {"k": [10, 20], "embed": ["tfidf", "sbert"], "rounds": [3], "epochs": [8], "seed": [0]},
}

# run_time_s e tokens vêm do metadata.json do método; tempo_parede_s inclui preparação e conversões
FIELDS = ["dataset", "modelo", "precisao", "recall", "f1", "configuracao", "avaliacao",
          "tp", "fp", "fn", "erros_reais", "detectadas", "run_time_s", "uses_llm", "prompt_tokens",
          "completion_tokens", "tempo_parede_s", "status", "pred"]


def expand(grid):
    subs = grid if isinstance(grid, list) else [grid]
    for g in subs:
        keys = list(g)
        for vals in itertools.product(*(g[k] for k in keys)):
            yield dict(zip(keys, vals))


def cfg_id(cfg):
    return "_".join(f"{k}{v}" for k, v in cfg.items() if v is not None).replace("%", "pct").replace("/", "-")


def default_py(env):
    p = os.path.join(ENVS, env, "bin", "python")
    return p if os.path.isfile(p) else "python"


def load_dotenv(path):
    env = {}
    if os.path.isfile(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


# ---------------------------------------------------------------- comandos de cada método
# Cada função devolve [(argv, cwd)] e as avaliações [(rótulo, pred.csv, extra_args_evaluate)].
def cmds_raha(a, cfg, out):
    argv = [a.py_raha, "runners/run_raha.py", "--name", a.name, "--dirty", a.dirty, "--clean", a.clean,
            "--budget", str(cfg["budget"]), "--seed", str(cfg["seed"]), "--outdir", out]
    return [argv], [("todas", os.path.join(out, a.name, f"raha_b{cfg['budget']}_s{cfg['seed']}", "pred.csv"), [])]


def cmds_dboost(a, cfg, out):
    params = [str(cfg["threshold"])] if cfg["model"] == "gaussian" else [str(cfg["peak"]), str(cfg["outlier"])]
    argv = [a.py_raha, "runners/run_dboost.py", "--name", a.name, "--dirty", a.dirty,
            "--config", cfg["model"], *params, "--outdir", out]
    tag = "dboost_" + "_".join([cfg["model"], *params])
    return [argv], [("todas", os.path.join(out, a.name, tag, "pred.csv"), [])]


def cmds_zeroed(a, cfg, out):
    argv = [a.py_zeroed, "runners/run_zeroed.py", "all", "--name", a.name, "--dirty", a.dirty,
            "--clean", a.clean, "--runs", "1", "--n-method", cfg["n_method"], "--procs", str(a.zeroed_procs),
            "--outdir", out]
    method = "zeroed"
    if cfg.get("ft_dim"):
        argv += ["--ft-dim", str(cfg["ft_dim"])]
        method += f"_ft{cfg['ft_dim']}"
    return [argv], [("todas", os.path.join(out, a.name, f"{method}_s1", "pred.csv"), [])]


def cmds_gidcl(a, cfg, out):
    s, split = cfg["seed"], os.path.join(out, "split")
    split_cmd = [a.py_gidcl, "scripts/make_labeled_split.py", "--dirty", a.dirty, "--clean", a.clean,
                 "--k", str(cfg["k"]), "--clusters", str(cfg.get("clusters") or cfg["k"]), "--seed", str(s),
                 "--select", "cluster", "--embed", cfg["embed"], "--outdir", split]
    extra = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in cfg.items()
                     if k not in ("k", "clusters", "embed", "seed") and v is not None)
    # chaves {…} são preenchidas pelo runners/run_gidcl.py
    det = ("gidcl_detect.py --dirty {dirty} --labeled {labeled_csv} --clusters {split}/clusters.csv "
           "--outdir {outdir} --seed {seed} " + extra)
    run_cmd = [a.py_gidcl, "runners/run_gidcl.py", "--name", a.name, "--dirty", a.dirty, "--clean", a.clean,
               "--seed", str(s), "--gpu", str(a.gpu), "--split-dir", split, "--python", a.py_gidcl,
               "--gidcl-dir", "methods/gidcl_det", "--cmd", det, "--output", "{outdir}/pred_matrix.npy",
               "--outdir", out]
    pred = os.path.join(out, a.name, f"gidcl_s{s}", "pred.csv")
    evals = [("todas", pred, []),
             # sem as K tuplas rotuladas (que o método viu com o valor limpo)
             ("sem_rotuladas", pred, ["--exclude-rows", os.path.join(split, "labeled_rows.csv")])]
    return [split_cmd, run_cmd], evals


BUILDERS = {"raha": cmds_raha, "dboost": cmds_dboost, "zeroed": cmds_zeroed, "gidcl": cmds_gidcl}


# ---------------------------------------------------------------- execução
def run_cmd(argv, log, env, timeout):
    log.write(f"\n$ {shlex.join(argv)}\n")
    log.flush()
    p = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def evaluate(a, pred, extra, env):
    """Roda o evaluate.py (pred + GROUND_TRUTH + metadata.json) e devolve a linha produzida."""
    tmp = pred + ".eval.csv"
    if os.path.exists(tmp):
        os.remove(tmp)
    argv = [a.py_raha, "evaluate.py", "--pred", pred, "--ground-truth", a.ground_truth,
            "--metadata", os.path.join(os.path.dirname(pred), "metadata.json"), "--results", tmp, *extra]
    p = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(tmp):
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    with open(tmp) as f:
        row = list(csv.DictReader(f))[-1]
    os.remove(tmp)
    return row


def ensure_ground_truth(a):
    gt = a.ground_truth
    stale = not os.path.isfile(gt) or any(os.path.getmtime(x) > os.path.getmtime(gt) for x in (a.dirty, a.clean))
    if stale:
        p = subprocess.run([a.py_raha, "scripts/make_ground_truth.py", "--dirty", a.dirty, "--clean", a.clean,
                            "--out", gt], cwd=ROOT)
        if p.returncode != 0:
            sys.exit(f"não foi possível gerar {gt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="rótulo da base (pastas de saída e ZeroED: só letras, números, _)")
    ap.add_argument("--dirty")
    ap.add_argument("--clean")
    ap.add_argument("--methods", default="raha,dboost,zeroed,gidcl")
    ap.add_argument("--grid", help="JSON com a grade (substitui a padrão só dos métodos presentes nele)")
    ap.add_argument("--out", help="CSV de saída (padrão: outputs/<nome>/benchmark.csv)")
    ap.add_argument("--ground-truth", help="GROUND_TRUTH.csv (padrão: na pasta do dirty; gerado se faltar)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-grid", action="store_true", help="mostra a grade padrão em JSON e sai")
    ap.add_argument("--timeout", type=float, help="limite em segundos por configuração")
    ap.add_argument("--py-raha", default=default_py("raha"), help="python do Raha/dBoost (e do evaluate.py)")
    ap.add_argument("--py-zeroed", default=default_py("zeroed"))
    ap.add_argument("--py-gidcl", default=default_py("gidcl"))
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES do GIDCL")
    ap.add_argument("--zeroed-procs", default="2")
    a = ap.parse_args()

    if a.show_grid:
        print(json.dumps(DEFAULT_GRID, indent=2))
        return
    if not (a.name and a.dirty and a.clean):
        ap.error("--name, --dirty e --clean são obrigatórios")
    a.dirty, a.clean = os.path.abspath(a.dirty), os.path.abspath(a.clean)
    for p in (a.dirty, a.clean):
        if not os.path.isfile(p):
            sys.exit(f"arquivo não encontrado: {p}")

    a.ground_truth = os.path.abspath(a.ground_truth or os.path.join(os.path.dirname(a.dirty), "GROUND_TRUTH.csv"))
    if not a.dry_run:
        ensure_ground_truth(a)

    grid = dict(DEFAULT_GRID)
    if a.grid:
        grid.update(json.load(open(a.grid)))
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    unknown = set(methods) - set(BUILDERS)
    if unknown:
        sys.exit(f"métodos desconhecidos: {unknown}")

    out_csv = a.out or os.path.join(ROOT, "outputs", a.name, "benchmark.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    done = set()
    if a.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            done = {(r["modelo"], r["configuracao"]) for r in csv.DictReader(f) if r["status"] == "ok"}

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update({k: v for k, v in load_dotenv(os.path.join(ROOT, ".env")).items() if k not in os.environ})

    jobs = [(m, cfg) for m in methods for cfg in expand(grid[m])]
    print(f"{len(jobs)} configurações: " + ", ".join(f"{m}={sum(j[0] == m for j in jobs)}" for m in methods))
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)

    new_file = not os.path.exists(out_csv)
    if not new_file:
        with open(out_csv) as f:
            header = next(csv.reader(f), [])
        if header != FIELDS:
            sys.exit(f"{out_csv} tem colunas antigas/diferentes; use outro --out ou renomeie o arquivo")
    with open(out_csv, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, (m, cfg) in enumerate(jobs, 1):
            cfg_str = json.dumps(cfg, ensure_ascii=False)
            if (m, cfg_str) in done:
                print(f"[{i}/{len(jobs)}] {m} {cfg_str}: já feito, pulando")
                continue
            out = os.path.join(ROOT, "outputs", a.name, "bench", m, cfg_id(cfg))
            cmds, evals = BUILDERS[m](a, cfg, out)
            if a.dry_run:
                print(f"[{i}/{len(jobs)}] {m} {cfg_str}")
                for c in cmds:
                    print("   ", shlex.join(c))
                continue

            print(f"[{i}/{len(jobs)}] {m} {cfg_str} ...", end=" ", flush=True)
            os.makedirs(out, exist_ok=True)
            log_path = os.path.join(out, "run.log")
            t0, status = time.time(), "ok"
            with open(log_path, "a") as log:
                try:
                    for c in cmds:
                        rc = run_cmd(c, log, env, a.timeout)
                        if rc != 0:
                            status = f"erro (código {rc}; ver {os.path.relpath(log_path, ROOT)})"
                            break
                except subprocess.TimeoutExpired:
                    status = f"timeout ({a.timeout}s)"
            elapsed = round(time.time() - t0, 1)

            for label, pred, extra in evals:
                row = dict(dataset=a.name, modelo=m, configuracao=cfg_str, avaliacao=label,
                           tempo_parede_s=elapsed, status=status, pred=os.path.relpath(pred, ROOT))
                if status == "ok":
                    try:
                        r = evaluate(a, pred, extra, env)
                        row.update(precisao=r["precision"], recall=r["recall"], f1=r["f1"], tp=r["tp"],
                                   fp=r["fp"], fn=r["fn"], erros_reais=r["erros_reais"],
                                   detectadas=r["detectadas"], run_time_s=r["run_time_s"],
                                   uses_llm=r["uses_llm"], prompt_tokens=r["prompt_tokens"],
                                   completion_tokens=r["completion_tokens"])
                    except Exception as e:  # noqa: BLE001 — registra e segue para a próxima configuração
                        row["status"] = f"erro na avaliação: {str(e).splitlines()[-1][:200]}"
                writer.writerow(row)
                fcsv.flush()
                if label == "todas":
                    print(f"P={row.get('precisao', '-')} R={row.get('recall', '-')} F1={row.get('f1', '-')} "
                          f"({elapsed}s) {'' if row['status'] == 'ok' else row['status']}")
    if not a.dry_run:
        print(f"resultados: {out_csv}")


if __name__ == "__main__":
    main()
