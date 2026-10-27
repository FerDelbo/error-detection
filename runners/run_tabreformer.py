#!/usr/bin/env python3
"""Pipeline wrapper for TabReformer cell-error predictions.

IMPORTANT
---------
The public TabReformer README describes the model and installation, but does
not document a stable Python inference API or a command that accepts a dirty
CSV and writes cell predictions. This wrapper therefore does NOT guess the
model's internal API. Use one of these explicit integration paths:

1. --adapter path/to/adapter.py:predict_cells
   The callable must accept keyword arguments:
       dirty: pandas.DataFrame
       clean: pandas.DataFrame | None
       repo_path: pathlib.Path
       seed: int
       work_dir: pathlib.Path
   and return either:
       * a 2-D matrix / DataFrame of 0/1 with the same shape as `dirty`; OR
       * an iterable of (row_index, column_index) pairs (0-based) flagged dirty.

2. --predictions path/to/mask.csv
   If the repository's own inference code has already generated a binary
   matrix, this wrapper validates it and writes it to the pipeline's output
   location.

The output is always a CSV with the same row/column layout as --dirty and
values 0 (clean) / 1 (dirty), with no index column.
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
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _project_root() -> Path:
    """Find the project root, accommodating runners/run_tabreformer.py."""
    here = Path(__file__).resolve().parent
    candidates = [Path.cwd().resolve(), here, here.parent]
    for candidate in candidates:
        if (candidate / "methods").is_dir() or (candidate / "evaluate.py").is_file():
            return candidate
    # When this file is stored under runners/, its parent is the repository root.
    return here.parent


ROOT = _project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TabReformer cell-error detection -> outputs/<name>/tabreformer_s<seed>/pred.csv"
    )
    parser.add_argument("--name", required=True, help="Dataset label used in output directory")
    parser.add_argument("--dirty", required=True, help="Path to the dirty/input CSV")
    parser.add_argument(
        "--clean",
        default=None,
        help=(
            "Optional clean/reference CSV. It is passed to the adapter only if provided; "
            "it is not automatically used as a label source."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--repo",
        default=os.environ.get("TABREFORMER_REPO", str(ROOT / "methods" / "TabReformer")),
        help="Path to the cloned TabReformer repository (default: methods/TabReformer)",
    )
    backend = parser.add_mutually_exclusive_group(required=True)
    backend.add_argument(
        "--adapter",
        help=(
            "Python callable implementing the actual TabReformer inference, formatted as "
            "'path/to/adapter.py:function' or 'module:function'."
        ),
    )
    backend.add_argument(
        "--predictions",
        help="Path to an already-generated binary cell-mask CSV (same shape as --dirty)",
    )
    return parser.parse_args()


def load_csv(path: str, label: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {resolved}")
    return pd.read_csv(resolved, dtype=str, keep_default_na=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # Optional dependency; only set the seed if available.

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_callable(spec: str, repo_path: Path):
    """Load `file.py:function` or `importable.module:function`."""
    if ":" not in spec:
        raise ValueError("--adapter must be written as 'path/to/file.py:function' or 'module:function'")
    module_spec, function_name = spec.rsplit(":", 1)
    if not function_name:
        raise ValueError("--adapter must include a callable name after ':'")

    # Make the cloned repository importable, including common src layouts.
    for candidate in (repo_path, repo_path / "src", repo_path / "tabReformer"):
        candidate_str = str(candidate.resolve())
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    module_path = Path(module_spec).expanduser()
    if module_path.suffix == ".py" or module_path.exists():
        if not module_path.is_absolute():
            module_path = (Path.cwd() / module_path).resolve()
        if not module_path.is_file():
            raise FileNotFoundError(f"Adapter module not found: {module_path}")
        dynamic_name = f"_tabreformer_adapter_{abs(hash(str(module_path)))}"
        spec_obj = importlib.util.spec_from_file_location(dynamic_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load adapter module: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[dynamic_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    function = getattr(module, function_name, None)
    if not callable(function):
        raise AttributeError(f"Adapter callable '{function_name}' was not found in '{module_spec}'")
    return function


def _normalize_binary_matrix(raw: Any, expected_shape: tuple[int, int], columns: list[str]) -> np.ndarray:
    """Validate and normalize an adapter's prediction result to a 0/1 matrix."""
    # A list of tuples is the documented sparse-cell format. Check it first,
    # otherwise a coordinate list with shape (n_rows, 2) could be mistaken for a mask.
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, tuple) and len(item) == 2 for item in raw):
        result = np.zeros(expected_shape, dtype=np.int8)
        for row, col in raw:
            if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
                raise ValueError(f"Cell coordinate indexes must be integers, got {(row, col)!r}")
            if row < 0 or row >= expected_shape[0] or col < 0 or col >= expected_shape[1]:
                raise IndexError(f"Predicted cell {(row, col)!r} is outside matrix shape {expected_shape}")
            result[row, col] = 1
        return result

    if isinstance(raw, pd.DataFrame):
        if raw.shape != expected_shape:
            raise ValueError(f"Prediction DataFrame has shape {raw.shape}; expected {expected_shape}")
        # Column order matters for a cell-level mask.
        if list(raw.columns) != columns:
            raise ValueError(
                "Prediction DataFrame columns/order do not match the dirty CSV. "
                f"Expected {columns!r}; got {list(raw.columns)!r}."
            )
        values = raw.to_numpy()
    else:
        values = np.asarray(raw)

    if values.shape == expected_shape:
        # Accept numeric 0/1 and common string/boolean representations, not probabilities.
        mapping = {
            "0": 0, "1": 1,
            "false": 0, "true": 1,
            "False": 0, "True": 1,
        }
        try:
            flattened = []
            for value in values.ravel().tolist():
                if isinstance(value, (bool, np.bool_)):
                    flattened.append(int(value))
                elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                    flattened.append(int(value))
                elif isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
                    flattened.append(int(value))
                elif str(value) in mapping:
                    flattened.append(mapping[str(value)])
                else:
                    raise ValueError(f"Non-binary prediction value encountered: {value!r}")
            return np.asarray(flattened, dtype=np.int8).reshape(expected_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The adapter/predictions file must contain only binary values 0 and 1. "
                "Probability scores need an explicit thresholding rule first."
            ) from exc

    # If result is a sequence of cell coordinates, mark those positions as dirty.
    try:
        coordinates = list(raw)
    except TypeError as exc:
        raise ValueError(
            f"Predictions have shape {values.shape}; expected a binary matrix {expected_shape} "
            "or an iterable of (row, column) coordinates."
        ) from exc

    result = np.zeros(expected_shape, dtype=np.int8)
    for coordinate in coordinates:
        if not isinstance(coordinate, (tuple, list, np.ndarray)) or len(coordinate) != 2:
            raise ValueError(
                "Coordinate output must contain (row_index, column_index) pairs, using zero-based indexes."
            )
        row, col = coordinate
        if not isinstance(row, (int, np.integer)) or not isinstance(col, (int, np.integer)):
            raise ValueError(f"Cell coordinate indexes must be integers, got {coordinate!r}")
        if row < 0 or row >= expected_shape[0] or col < 0 or col >= expected_shape[1]:
            raise IndexError(f"Predicted cell {coordinate!r} is outside matrix shape {expected_shape}")
        result[row, col] = 1
    return result


def run_adapter(
    adapter_spec: str,
    dirty: pd.DataFrame,
    clean: pd.DataFrame | None,
    repo_path: Path,
    seed: int,
    work_dir: Path,
) -> Any:
    if not repo_path.is_dir():
        raise FileNotFoundError(
            f"TabReformer repository not found: {repo_path}\n"
            "Clone it there or pass --repo /path/to/TabReformer."
        )
    adapter = load_callable(adapter_spec, repo_path)
    # Standard adapter contract: named keyword arguments, so the implementation is explicit.
    return adapter(
        dirty=dirty.copy(),
        clean=None if clean is None else clean.copy(),
        repo_path=repo_path.resolve(),
        seed=seed,
        work_dir=work_dir.resolve(),
    )


def main() -> int:
    args = parse_args()
    start = time.time()
    set_seed(args.seed)

    dirty_path = Path(args.dirty).expanduser().resolve()
    dirty = load_csv(str(dirty_path), "Dirty/input")
    clean = load_csv(args.clean, "Clean/reference") if args.clean else None

    if clean is not None:
        if clean.shape != dirty.shape:
            raise ValueError(
                f"Dirty and clean CSV shapes differ: dirty={dirty.shape}, clean={clean.shape}. "
                "The reference CSV must align cell-by-cell with the dirty CSV."
            )
        if list(clean.columns) != list(dirty.columns):
            raise ValueError("Dirty and clean CSV columns/order do not match.")

    repo_path = Path(args.repo).expanduser().resolve()
    output_dir = Path(args.outdir).expanduser().resolve() / args.name / f"tabreformer_s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions:
        raw = load_csv(args.predictions, "Predictions")
        if list(raw.columns) != list(dirty.columns):
            raise ValueError(
                "Prediction CSV columns/order do not match the dirty CSV. "
                f"Expected {list(dirty.columns)!r}; got {list(raw.columns)!r}."
            )
        matrix = _normalize_binary_matrix(raw, dirty.shape, list(dirty.columns))
    else:
        raw = run_adapter(args.adapter, dirty, clean, repo_path, args.seed, work_dir)
        matrix = _normalize_binary_matrix(raw, dirty.shape, list(dirty.columns))

    output_path = output_dir / "pred.csv"
    pd.DataFrame(matrix, columns=dirty.columns).to_csv(output_path, index=False)
    elapsed = time.time() - start
    print(
        f"[tabreformer] {args.name} seed={args.seed} "
        f"detectadas={int(matrix.sum())} tempo={elapsed:.1f}s -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, ImportError, AttributeError, IndexError, TypeError) as exc:
        print(f"[tabreformer] ERRO: {exc}", file=sys.stderr)
        raise SystemExit(2)