"""metadata.json comum a todos os runners (lido pelo evaluate.py).

Formato:
{
  "method": "raha_b20_s0",        # nome da pasta de saída
  "dataset": "beers",
  "run_time_s": 12.3,             # obrigatório
  "uses_llm": false,              # se true, prompt_tokens e completion_tokens são obrigatórios
  "prompt_tokens": null,          # tokens de entrada
  "completion_tokens": null,      # tokens de saída
  "config": {"budget": 2, "seed": 0},   # obrigatório: configuração do algoritmo
  "extra": {...}                  # opcional: qualquer informação adicional do método
}
"""
import json
import os

REQUIRED = ("method", "dataset", "run_time_s", "uses_llm", "config")


def guard_output(out_dir, config, overwrite=False):
    """Aborta se out_dir já tem resultado de uma configuração diferente.

    Chamar ANTES de gravar o pred.csv. Mesma configuração (reexecução) é permitida;
    configuração diferente só com overwrite=True (--overwrite nos runners).
    """
    path = os.path.join(out_dir, "metadata.json")
    if overwrite or not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            old = json.load(f).get("config")
    except (OSError, ValueError):
        old = None
    # normaliza pelo JSON para comparar tipos iguais aos gravados (tuplas -> listas etc.)
    new = json.loads(json.dumps(config, default=str))
    if old != new:
        raise SystemExit(f"{out_dir} já contém resultado de outra configuração:\n"
                         f"  existente: {old}\n  atual:     {new}\n"
                         f"Mude os parâmetros, use outro --outdir ou passe --overwrite.")


def write_metadata(out_dir, method, dataset, config, run_time_s,
                   uses_llm=False, prompt_tokens=None, completion_tokens=None, extra=None):
    meta = {
        "method": method,
        "dataset": dataset,
        "run_time_s": round(float(run_time_s), 2),
        "uses_llm": bool(uses_llm),
        "prompt_tokens": None if prompt_tokens is None else int(prompt_tokens),
        "completion_tokens": None if completion_tokens is None else int(completion_tokens),
        "config": config,
        "extra": extra or {},
    }
    validate(meta)
    path = os.path.join(out_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=str)
    return path


def validate(meta, source="metadata"):
    """Levanta ValueError se faltar algo obrigatório."""
    miss = [k for k in REQUIRED if k not in meta or meta[k] is None]
    if miss:
        raise ValueError(f"{source}: campos obrigatórios ausentes: {miss}")
    if not isinstance(meta["config"], dict):
        raise ValueError(f"{source}: 'config' deve ser um objeto JSON")
    if meta["uses_llm"]:
        miss = [k for k in ("prompt_tokens", "completion_tokens") if meta.get(k) is None]
        if miss:
            raise ValueError(f"{source}: uses_llm=true exige {miss}")
