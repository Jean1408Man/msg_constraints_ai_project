from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, List

from .evaluator import MessageEvaluator
from .llm import PROMPT_VERSION, GroqRateLimitError, build_generator
from .models import ConstraintResult, Evaluation, ExperimentRow, MessageInstance
from .strategies import build_strategy


DEFAULT_STRATEGIES = ["unified"]


def _stable_offset(value: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value)) % 10000


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "si", "sí"}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _query_for(instance: MessageInstance) -> str:
    return instance.raw_request or instance.to_json()


def _failed_constraints(results: list[ConstraintResult]) -> list[dict[str, object]]:
    return [
        {
            "name": result.name,
            "source": result.source,
            "details": result.details,
            "missing": result.missing,
        }
        for result in results
        if not result.passed
    ]


def _invalid_reason(ev: Evaluation) -> str:
    failures = _failed_constraints(ev.constraint_results)
    if not failures:
        return "válido"
    return " | ".join(
        f"{item['name']}[{item['source']}]: {item['details']}"
        for item in failures
    )


def _truncate_error(text: object, limit: int = 500) -> str:
    clean = " ".join(str(text).split())
    return clean[:limit]


class ExperimentRunner:
    def __init__(
        self,
        seed: int | None = 42,
        generator_name: str = "groq",
        model: str | None = None,
        groq_base_url: str | None = None,
        temperature: float = 0.2,
        open_constraint_evaluator: str = "llm",
        semantic_evaluator: str = "llm",
        max_attempts: int = 3,
        population_size: int = 2,
    ):
        self.seed = seed
        self.generator_name = generator_name
        self.model = model
        self.groq_base_url = groq_base_url
        self.temperature = temperature
        self.open_constraint_evaluator = open_constraint_evaluator
        self.semantic_evaluator = semantic_evaluator
        self.max_attempts = max_attempts
        self.population_size = population_size

    def run(
        self,
        instances: Iterable[MessageInstance],
        strategies: List[str] | None = None,
        skip_keys: set[tuple[str, str]] | None = None,
    ) -> List[ExperimentRow]:
        return list(self.iter_rows(instances, strategies=strategies, skip_keys=skip_keys))

    def iter_rows(
        self,
        instances: Iterable[MessageInstance],
        strategies: List[str] | None = None,
        skip_keys: set[tuple[str, str]] | None = None,
        on_task_start: Callable[[MessageInstance, str], None] | None = None,
    ) -> Iterator[ExperimentRow]:
        strategies = strategies or DEFAULT_STRATEGIES
        skip_keys = skip_keys or set()
        evaluator = MessageEvaluator(
            open_constraint_evaluator=self.open_constraint_evaluator,
            semantic_evaluator=self.semantic_evaluator,
            model=self.model,
            groq_base_url=self.groq_base_url,
        )

        for instance in instances:
            for sname in strategies:
                if (instance.instance_id, sname) in skip_keys:
                    continue
                if on_task_start is not None:
                    on_task_start(instance, sname)
                task_start = time.perf_counter()
                try:
                    generator = build_generator(
                        self.generator_name,
                        seed=(self.seed or 0) + _stable_offset(instance.instance_id + sname),
                        model=self.model,
                        groq_base_url=self.groq_base_url,
                        temperature=self.temperature,
                    )
                    strategy = build_strategy(
                        sname,
                        generator,
                        evaluator,
                        max_attempts=self.max_attempts,
                        population_size=self.population_size,
                    )
                    candidate = strategy.run(instance)
                    ev = candidate.evaluation
                    if ev is None:
                        ev = evaluator.evaluate(candidate.text, instance)
                        candidate.metadata["constraint_eval_time_ms"] = ev.constraint_eval_time_ms
                        candidate.metadata["semantic_eval_time_ms"] = ev.semantic_eval_time_ms

                    yield self._build_row(instance, sname, candidate.attempts, candidate.text, ev, candidate.metadata)
                except GroqRateLimitError as exc:
                    elapsed_ms = (time.perf_counter() - task_start) * 1000
                    yield self._build_error_row(instance, sname, exc, elapsed_ms)
                    return
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - task_start) * 1000
                    yield self._build_error_row(instance, sname, exc, elapsed_ms)

    def run_to_csv(
        self,
        path: str | Path,
        instances: Iterable[MessageInstance],
        strategies: List[str] | None = None,
        resume: bool = False,
        on_start: Callable[[int, int, int], None] | None = None,
        on_task_start: Callable[[int, int, MessageInstance, str], None] | None = None,
        on_row_complete: Callable[[int, int, ExperimentRow], None] | None = None,
    ) -> List[ExperimentRow]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        strategies = strategies or DEFAULT_STRATEGIES
        instance_list = list(instances)
        total = len(instance_list) * len(strategies)
        requested_keys = {(instance.instance_id, strategy) for instance in instance_list for strategy in strategies}

        loaded_rows = self._read_existing_rows(p) if resume and p.exists() else []
        existing_rows = [
            row
            for row in loaded_rows
            if self._is_compatible_resume_row(row) and (row.instance_id, row.strategy) in requested_keys
        ]
        incompatible_count = len(loaded_rows) - len(existing_rows)
        skip_keys = {(row.instance_id, row.strategy) for row in existing_rows}
        rows = list(existing_rows)
        done = len(skip_keys)
        if on_start is not None:
            on_start(total, done, incompatible_count)

        def notify_task_start(instance: MessageInstance, strategy: str) -> None:
            if on_task_start is not None:
                on_task_start(done, total, instance, strategy)

        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames())
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row.__dict__)
            f.flush()

            for row in self.iter_rows(
                instance_list,
                strategies=strategies,
                skip_keys=skip_keys,
                on_task_start=notify_task_start,
            ):
                writer.writerow(row.__dict__)
                f.flush()
                rows.append(row)
                done += 1
                if on_row_complete is not None:
                    on_row_complete(done, total, row)
        return rows

    @staticmethod
    def _build_row(
        instance: MessageInstance,
        strategy: str,
        attempts: int,
        message: str,
        ev: Evaluation,
        metadata: dict[str, object],
    ) -> ExperimentRow:
        generation_time_ms = _as_float(metadata.get("generation_time_ms"))
        constraint_eval_time_ms = _as_float(metadata.get("constraint_eval_time_ms"), ev.constraint_eval_time_ms)
        semantic_eval_time_ms = _as_float(metadata.get("semantic_eval_time_ms"), ev.semantic_eval_time_ms)
        measured_total = _as_float(metadata.get("total_time_ms"))
        total_time_ms = measured_total or generation_time_ms + constraint_eval_time_ms + semantic_eval_time_ms
        total_time_ms = round(total_time_ms, 3)
        return ExperimentRow(
            instance_id=instance.instance_id,
            strategy=strategy,
            difficulty=instance.difficulty,
            prompt_version=PROMPT_VERSION,
            query=_query_for(instance),
            instance_json=instance.to_json(),
            open_constraint_evaluator=ev.open_constraint_evaluator,
            semantic_evaluator=ev.semantic_evaluator,
            valid=ev.valid,
            hard_score=ev.hard_score,
            semantic_score=ev.semantic_score,
            total_score=ev.total_score,
            violations=ev.violations,
            attempts=attempts,
            attempt_trace=json.dumps(metadata.get("attempt_trace", []), ensure_ascii=False),
            failed_constraints=json.dumps(_failed_constraints(ev.constraint_results), ensure_ascii=False),
            invalid_reason=_invalid_reason(ev),
            error_type="",
            error_message="",
            evaluation_notes=json.dumps(ev.notes, ensure_ascii=False),
            generation_time_ms=round(generation_time_ms, 3),
            constraint_eval_time_ms=round(constraint_eval_time_ms, 3),
            semantic_eval_time_ms=round(semantic_eval_time_ms, 3),
            total_time_ms=total_time_ms,
            time_ms=total_time_ms,
            message=message,
        )

    def _build_error_row(
        self,
        instance: MessageInstance,
        strategy: str,
        error: Exception,
        elapsed_ms: float,
    ) -> ExperimentRow:
        error_type = type(error).__name__
        error_message = _truncate_error(error)
        return ExperimentRow(
            instance_id=instance.instance_id,
            strategy=strategy,
            difficulty=instance.difficulty,
            prompt_version=PROMPT_VERSION,
            query=_query_for(instance),
            instance_json=instance.to_json(),
            open_constraint_evaluator=self.open_constraint_evaluator,
            semantic_evaluator=self.semantic_evaluator,
            valid=False,
            hard_score=0.0,
            semantic_score=0.0,
            total_score=0.0,
            violations=0,
            attempts=0,
            attempt_trace="[]",
            failed_constraints="[]",
            invalid_reason=f"error durante la ejecución: {error_type}: {error_message}",
            error_type=error_type,
            error_message=error_message,
            evaluation_notes="[]",
            generation_time_ms=0.0,
            constraint_eval_time_ms=0.0,
            semantic_eval_time_ms=0.0,
            total_time_ms=round(elapsed_ms, 3),
            time_ms=round(elapsed_ms, 3),
            message="",
        )

    @staticmethod
    def fieldnames() -> list[str]:
        return list(ExperimentRow.__dataclass_fields__.keys())

    def _is_compatible_resume_row(self, row: ExperimentRow) -> bool:
        return (
            row.open_constraint_evaluator == self.open_constraint_evaluator
            and row.semantic_evaluator == self.semantic_evaluator
            and bool(row.query)
            and bool(row.instance_json)
            and row.prompt_version == PROMPT_VERSION
            and not row.error_type
        )

    @classmethod
    def write_csv(cls, path: str | Path, rows: List[ExperimentRow]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cls.fieldnames())
            writer.writeheader()
            for row in rows:
                writer.writerow(row.__dict__)

    @classmethod
    def _read_existing_rows(cls, path: Path) -> list[ExperimentRow]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [cls._row_from_dict(row) for row in csv.DictReader(f)]

    @staticmethod
    def _row_from_dict(row: dict[str, str]) -> ExperimentRow:
        total_time_ms = _as_float(row.get("total_time_ms"), _as_float(row.get("time_ms")))
        generation_time_ms = _as_float(row.get("generation_time_ms"), total_time_ms)
        constraint_eval_time_ms = _as_float(row.get("constraint_eval_time_ms"))
        semantic_eval_time_ms = _as_float(row.get("semantic_eval_time_ms"))
        return ExperimentRow(
            instance_id=row.get("instance_id", ""),
            strategy=row.get("strategy", ""),
            difficulty=row.get("difficulty", ""),
            prompt_version=row.get("prompt_version", ""),
            query=row.get("query", ""),
            instance_json=row.get("instance_json", ""),
            open_constraint_evaluator=row.get("open_constraint_evaluator", "legacy"),
            semantic_evaluator=row.get("semantic_evaluator", "legacy"),
            valid=_as_bool(row.get("valid", "false")),
            hard_score=_as_float(row.get("hard_score")),
            semantic_score=_as_float(row.get("semantic_score")),
            total_score=_as_float(row.get("total_score")),
            violations=_as_int(row.get("violations")),
            attempts=_as_int(row.get("attempts")),
            attempt_trace=row.get("attempt_trace", ""),
            failed_constraints=row.get("failed_constraints", ""),
            invalid_reason=row.get("invalid_reason", ""),
            error_type=row.get("error_type", ""),
            error_message=row.get("error_message", ""),
            evaluation_notes=row.get("evaluation_notes", ""),
            generation_time_ms=round(generation_time_ms, 3),
            constraint_eval_time_ms=round(constraint_eval_time_ms, 3),
            semantic_eval_time_ms=round(semantic_eval_time_ms, 3),
            total_time_ms=round(total_time_ms, 3),
            time_ms=round(total_time_ms, 3),
            message=row.get("message", ""),
        )

    @staticmethod
    def summarize(rows: List[ExperimentRow]) -> str:
        if not rows:
            return "Sin resultados."
        by_strategy: dict[str, list[ExperimentRow]] = {}
        for r in rows:
            by_strategy.setdefault(r.strategy, []).append(r)
        lines = ["Resumen experimental:"]
        for strategy, vals in sorted(by_strategy.items()):
            n = len(vals)
            valid_rate = sum(v.valid for v in vals) / n
            avg_total = sum(v.total_score for v in vals) / n
            avg_viol = sum(v.violations for v in vals) / n
            avg_time = sum(v.total_time_ms for v in vals) / n
            lines.append(
                f"- {strategy}: validez={valid_rate:.2%}, score={avg_total:.3f}, "
                f"violaciones={avg_viol:.2f}, tiempo={avg_time:.2f} ms"
            )
        return "\n".join(lines)
