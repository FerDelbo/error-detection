"""Perfil de uma coluna calculado só a partir da tabela suja (sem LLM e sem o clean).

Substitui, como contexto para o Jev, as diretrizes e a análise de distribuição que o
ZeroED original pede ao LLM (etapas 4 e 5). Tudo aqui é estatística simples e determinística.
"""
import re
from collections import Counter

NULL_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-"}


def value_pattern(v):
    """Padrão de formato: letra minúscula -> a, maiúscula -> A, dígito -> 9, repetições colapsadas.
    Ex.: '35233' -> '9', 'Al-12' -> 'Aa-9', '1300 s main st' -> '9 a a a'."""
    s = re.sub(r"[a-z]", "a", str(v))
    s = re.sub(r"[A-Z]", "A", s)
    s = re.sub(r"[0-9]", "9", s)
    return re.sub(r"(.)\1+", r"\1", s)


def profile(series, top_values=10, top_patterns=8, max_len=60):
    vals = [str(v) for v in series.tolist()]
    n = len(vals)
    if n == 0:
        return "empty column"
    nulls = sum(v.strip().lower() in NULL_TOKENS for v in vals)
    counts = Counter(vals)
    lens = sorted(len(v) for v in vals)
    pats = Counter(value_pattern(v) for v in vals)

    def cut(v):
        return v if len(v) <= max_len else v[:max_len] + "..."

    lines = [
        f"rows: {n}; distinct values: {len(counts)}; null-like values: {nulls} ({nulls / n:.1%})",
        f"value length: min {lens[0]}, median {lens[n // 2]}, max {lens[-1]}",
        "most frequent values (count): " + "; ".join(f"'{cut(v)}' ({c})" for v, c in counts.most_common(top_values)),
        "most frequent formats (share; a=lowercase run, A=uppercase run, 9=digit run): "
        + "; ".join(f"'{cut(p)}' ({c / n:.1%})" for p, c in pats.most_common(top_patterns)),
    ]
    return "\n".join(lines)


def profile_table(dirty_csv):
    return {col: profile(dirty_csv[col]) for col in dirty_csv.columns}
