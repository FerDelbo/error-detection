#!/usr/bin/env python3
"""
Runner genérico para o método TabReformer (Nashaat et al., 2021).
Usa por padrão o adapter `runners/tabreformer_adapter.py:predict_cells`,
permitindo rodar diretamente com:
  python runners/run_tabreformer.py --name <name> --dirty <dirty.csv> --clean <clean.csv>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), here, here.parent]
    for candidate in candidates:
        if (candidate / "methods").is_dir() or (candidate / "evaluate.py").is_file():
            return candidate
    return here.parent


ROOT = _project_root()
DEFAULT_ADAPTER = f"{Path(__file__).resolve().parent / 'tabreformer_adapter.py'}:predict_cells"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TabReformer cell-error detection -> outputs/<name>/tabreformer_s<seed>/pred.csv"
    )
    parser.add_argument("--name", required=True, help="Rótulo do dataset")
    parser.add_argument("--dirty", required=True, help="Caminho para o arquivo dirty.csv (qualquer local)")
    parser.add_argument("--clean", default=None, help="Caminho para o arquivo clean.csv (qualquer local)")
    parser.add_argument("--seed", type=int, default=0, help="Semente aleatória (padrão: 0)")
    parser.add_argument("--budget", type=int, default=2, help="Tuplas rotuladas para fine-tuning (padrão: 2)")
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--repo",
        default=os.environ.get("TABREFORMER_REPO", str(ROOT / "methods" / "TabReformer")),
        help="Caminho para methods/TabReformer",
    )
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help="Módulo:função do adapter Python (padrão: runners/tabreformer_adapter.py:predict_cells)",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Opcional: caminho para uma máscara binária CSV pré-computada",
    )
    return parser.parse_args()


def load_csv(path: str, label: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} CSV não encontrado: {resolved}")
    return pd.read_csv(resolved, dtype=str, keep_default_na=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_callable(spec: str, repo_path: Path):
    if ":" not in spec:
        raise ValueError("--adapter deve ter o formato 'caminho/arquivo.py:funcao' ou 'modulo:funcao'")
    module_spec, function_name = spec.rsplit(":", 1)
    for candidate in (repo_path, repo_path / "src", repo_path / "tabReformer"):
        if candidate.is_dir() and str(candidate.resolve()) not in sys.path:
            sys.path.insert(0, str(candidate.resolve()))

    module_path = Path(module_spec).expanduser()
    if module_path.suffix == ".py" or module_path.exists():
        if not module_path.is_absolute():
            module_path = (Path.cwd() / module_path).resolve()
        dynamic_name = f"_tabreformer_adapter_{abs(hash(str(module_path)))}"
        spec_obj = importlib.util.spec_from_file_location(dynamic_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Não foi possível carregar o adapter: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[dynamic_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(f"Função '{function_name}' não encontrada em '{module_spec}'")
    return function


def main() -> int:
    args = parse_args()
    start = time.time()
    set_seed(args.seed)
    os.environ["TABREFORMER_BUDGET"] = str(args.budget)

    dirty_df = load_csv(args.dirty, "Dirty")
    clean_df = load_csv(args.clean, "Clean") if args.clean else None

    outdir = Path(args.outdir) / args.name / f"tabreformer_s{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "pred.csv"

    if args.predictions:
        pred_path = Path(args.predictions).expanduser().resolve()
        raw = pd.read_csv(pred_path, dtype=int).to_numpy()
    else:
        repo_path = Path(args.repo).expanduser().resolve()
        adapter = load_callable(args.adapter, repo_path)
        raw = adapter(
            dirty=dirty_df.copy(),
            clean=None if clean_df is None else clean_df.copy(),
            repo_path=repo_path,
            seed=args.seed,
            work_dir=outdir.resolve(),
        )

    mask = np.asarray(raw, dtype=int).reshape(dirty_df.shape)
    pred_df = pd.DataFrame(mask, columns=dirty_df.columns)
    pred_df.to_csv(out_csv, index=False)

    duration = time.time() - start
    print(
        f"[tabreformer] {args.name} seed={args.seed} detectadas={int(mask.sum())} "
        f"tempo={duration:.1f}s -> {out_csv}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())