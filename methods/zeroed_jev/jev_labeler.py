"""Rotulação dos valores centrais pelo Jev (substitui a etapa 6 do ZeroED: subtask_det_initial).

Entrada: os mesmos centros que o ZeroED mandaria ao LLM (cluster_index_dict[attr][0]) com as
mesmas colunas (o atributo e os atributos relacionados). Cada lote de `batch` centros vira UMA
requisição: o `state` leva o perfil da coluna (column_profile.py) e as linhas; há uma pergunta de
escolha ("error"/"correct") por linha.

Saída: center_index_value_label_dict no formato do main.py, {attr: [(idx, valor_dict, 0|1), ...]},
montado pelo índice da linha (sem o parser de texto por regex do ZeroED original). Para auditoria,
grava em error_checking/:
  jev_labels_<attr>.jsonl   idx, valor, escolha, confiança, p_erro e rótulo
  error_checking_<attr>.txt o mesmo resultado no formato de texto que o LLM do ZeroED produz
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

CHOICES = ["error", "correct"]


def _row_str(row):
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in row.items()) + "}"


def build_state(attr, related_attrs, profile_text, rows):
    rel = ", ".join(related_attrs) if related_attrs else "(none)"
    lines = [
        "You are checking a data table for errors (typos, missing values, pattern violations, "
        "violations of dependencies between columns, outliers).",
        f"Column under inspection: '{attr}'. Related columns shown for context: {rel}.",
        "Only the value of the column under inspection is judged; problems in other columns are ignored. "
        "Letter case is ignored.",
        "",
        f"Profile of column '{attr}' computed over the whole table:",
        profile_text,
        "",
        "Rows to check:",
    ]
    lines += [f"[r{k}] {_row_str(r)}" for k, r in enumerate(rows)]
    return "\n".join(lines)


def build_questions(attr, rows):
    return [{"id": f"r{k}",
             "instructions": f"Is the value of '{attr}' in row [r{k}] ('{r[attr]}') erroneous? "
                             f"Answer 'error' if it is wrong, 'correct' otherwise.",
             "choices": CHOICES} for k, r in enumerate(rows)]


def p_error(answer):
    """Probabilidade de erro a partir da escolha e da confiança devolvidas pelo Jev."""
    c = min(max(float(answer["confidence"]), 0.0), 1.0)
    if answer["choice"] == "error":
        return c
    if answer["choice"] == "correct":
        return 1.0 - c
    raise ValueError(f"escolha inesperada do Jev: {answer['choice']!r}")


def _audit_text(attr, recs):
    """Mesmo formato de texto do LLM no ZeroED (lido por extract_llm_label_res), só para auditoria."""
    entries = []
    for r in recs:
        entries.append("    {\n"
                       f'"value_row": "{_row_str(r["value"])}",\n'
                       f'"error_analysis": "Jev: {r["choice"]} (confidence {r["confidence"]:.3f})",\n'
                       f'"has_error_in_{attr}_value": {"true" if r["label"] == 1 else "false"},\n'
                       "    }")
    return ("```json\n{\n" f'"column_name": "{attr}",\n  "entries": [\n' + ",\n".join(entries) + "\n  ]\n}\n```\n")


def label_all(client, all_attrs, dirty_csv, related_attrs_dict, cluster_index_dict, center_value_dict,
              profiles, out_dir, batch=20, threshold=0.5, workers=4):
    jobs = []  # (attr, [(pos, idx, value), ...])
    for attr in all_attrs:
        centers = [(pos, int(idx), center_value_dict[attr][pos])
                   for pos, idx in enumerate(cluster_index_dict[attr][0])]
        for s in range(0, len(centers), batch):
            jobs.append((attr, centers[s:s + batch]))

    def run(job):
        attr, items = job
        rows = [v for _, _, v in items]
        state = build_state(attr, list(related_attrs_dict[attr]), profiles[attr], rows)
        answers = client.ask(state, build_questions(attr, rows))
        return attr, items, answers

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        results = list(ex.map(run, jobs))

    per_attr = {attr: [] for attr in all_attrs}
    for attr, items, answers in results:
        for (pos, idx, value), ans in zip(items, answers):
            p = p_error(ans)
            per_attr[attr].append({"pos": pos, "idx": idx, "value": value, "choice": ans["choice"],
                                   "confidence": ans["confidence"], "p_error": round(p, 4),
                                   "label": int(p > threshold)})

    os.makedirs(out_dir, exist_ok=True)
    center_index_value_label_dict = {}
    for attr in all_attrs:
        recs = sorted(per_attr[attr], key=lambda r: r["pos"])
        with open(os.path.join(out_dir, f"jev_labels_{attr}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, f"error_checking_{attr}.txt"), "w", encoding="utf-8") as f:
            f.write(_audit_text(attr, recs))
        # mesma tupla que extract_llm_label_res monta: (idx, valor_dict, rótulo)
        center_index_value_label_dict[attr] = [(r["idx"], r["value"], r["label"]) for r in recs]
    return center_index_value_label_dict
