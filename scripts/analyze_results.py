from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "si", "sí"}


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def _as_int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, "0") or 0))


def _has_error(row: dict[str, str]) -> bool:
    return bool(row.get("error_type", "").strip())


def _distinct(rows: list[dict[str, str]], key: str) -> str:
    values = sorted({row.get(key, "") or "no registrado" for row in rows})
    return ", ".join(values) if values else "sin datos"


def _time_value(row: dict[str, str], key: str) -> float:
    if row.get(key):
        return _as_float(row, key)
    if key == "total_time_ms":
        return _as_float(row, "time_ms")
    return 0.0


def _clip(value: str, limit: int = 140) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _cell(value: str, limit: int = 140) -> str:
    return _clip(value.replace("|", "/"), limit)


def _summarize(rows: list[dict[str, str]], keys: Iterable[str]) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)

    summary: list[dict[str, str]] = []
    for group_key, vals in sorted(groups.items()):
        n = len(vals)
        out = {key: value for key, value in zip(keys, group_key)}
        out.update(
            {
                "n": str(n),
                "validez": f"{sum(_as_bool(v.get('valid', 'false')) for v in vals) / n:.2%}",
                "hard_score": f"{mean(_as_float(v, 'hard_score') for v in vals):.4f}",
                "semantic_score": f"{mean(_as_float(v, 'semantic_score') for v in vals):.4f}",
                "total_score": f"{mean(_as_float(v, 'total_score') for v in vals):.4f}",
                "violaciones": f"{mean(_as_int(v, 'violations') for v in vals):.2f}",
                "intentos": f"{mean(_as_int(v, 'attempts') for v in vals):.2f}",
                "errores": str(sum(_has_error(v) for v in vals)),
                "generacion_ms": f"{mean(_time_value(v, 'generation_time_ms') for v in vals):.2f}",
                "restricciones_ms": f"{mean(_time_value(v, 'constraint_eval_time_ms') for v in vals):.2f}",
                "semantica_ms": f"{mean(_time_value(v, 'semantic_eval_time_ms') for v in vals):.2f}",
                "total_ms": f"{mean(_time_value(v, 'total_time_ms') for v in vals):.2f}",
            }
        )
        summary.append(out)
    return summary


def _markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "Sin datos.\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.get(col, "") for col in columns) + " |" for row in rows]
    return "\n".join([header, sep, *body]) + "\n"


def _diagnostic_rows(rows: list[dict[str, str]], error: bool, limit: int = 15) -> list[dict[str, str]]:
    if error:
        selected = [row for row in rows if _has_error(row)]
    else:
        selected = [row for row in rows if not _has_error(row) and not _as_bool(row.get("valid", "false"))]
    out: list[dict[str, str]] = []
    for row in selected[:limit]:
        out.append(
            {
                "instance_id": row.get("instance_id", ""),
                "strategy": row.get("strategy", ""),
                "difficulty": row.get("difficulty", ""),
                "error_type": row.get("error_type", ""),
                "reason": _cell(row.get("error_message") or row.get("invalid_reason", ""), 180),
                "trace": _trace_summary(row),
                "query": _cell(row.get("query", ""), 180),
            }
        )
    return out


def _trace_summary(row: dict[str, str]) -> str:
    raw = row.get("attempt_trace", "")
    if not raw:
        return ""
    try:
        trace = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(trace, list):
        return ""

    chunks: list[str] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        failed = item.get("failed") or []
        failed_text = "; ".join(str(x) for x in failed) if isinstance(failed, list) else str(failed)
        chunks.append(
            f"{item.get('kind')}#{item.get('attempt')} "
            f"valid={item.get('valid')} words={item.get('word_count')} failed={failed_text}"
        )
    return _cell(" / ".join(chunks), 220)


def _failed_constraint_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], str] = {}
    for row in rows:
        raw = row.get("failed_constraints", "")
        if not raw:
            continue
        try:
            failures = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(failures, list):
            continue
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            key = (str(failure.get("name", "")), str(failure.get("source", "")))
            if not key[0]:
                continue
            counts[key] += 1
            examples.setdefault(key, _cell(str(failure.get("details", "")), 120))

    return [
        {
            "constraint": name,
            "source": source,
            "count": str(count),
            "example": examples.get((name, source), ""),
        }
        for (name, source), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def analyze(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    columns = [
        "strategy",
        "difficulty",
        "n",
        "validez",
        "hard_score",
        "semantic_score",
        "total_score",
        "violaciones",
        "intentos",
        "errores",
        "generacion_ms",
        "restricciones_ms",
        "semantica_ms",
        "total_ms",
    ]
    by_strategy = _summarize(rows, ["strategy"])
    by_difficulty = _summarize(rows, ["difficulty"])
    by_both = _summarize(rows, ["strategy", "difficulty"])
    error_rows = _diagnostic_rows(rows, error=True)
    invalid_rows = _diagnostic_rows(rows, error=False)
    diagnostic_columns = ["instance_id", "strategy", "difficulty", "error_type", "reason", "trace", "query"]
    failed_summary = _failed_constraint_summary(rows)

    lines = [
        f"# Analisis de resultados: {path}",
        "",
        f"Filas analizadas: {len(rows)}",
        "",
        f"Evaluador de restricciones abiertas: {_distinct(rows, 'open_constraint_evaluator')}",
        f"Evaluador semantico: {_distinct(rows, 'semantic_evaluator')}",
        f"Version de prompt: {_distinct(rows, 'prompt_version')}",
        "",
        "## Resumen por estrategia",
        "",
        _markdown_table(by_strategy, [c for c in columns if c != "difficulty"]),
        "## Resumen por dificultad",
        "",
        _markdown_table(by_difficulty, [c for c in columns if c != "strategy"]),
        "## Resumen por estrategia y dificultad",
        "",
        _markdown_table(by_both, columns),
        "## Restricciones fallidas",
        "",
        _markdown_table(failed_summary, ["constraint", "source", "count", "example"]),
        "## Errores registrados",
        "",
        _markdown_table(error_rows, diagnostic_columns),
        "## Candidatos no validos",
        "",
        _markdown_table(invalid_rows, diagnostic_columns),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza CSV de experimentos messagegen")
    parser.add_argument("--csv", default="results/groq_final_results.csv")
    parser.add_argument("--out", default=None, help="Ruta .md opcional para guardar el análisis")
    args = parser.parse_args()

    report = analyze(Path(args.csv))
    print(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
