from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from .evaluator import MessageEvaluator
from .llm import BaseGenerator
from .models import Candidate, ConstraintResult, Evaluation, MessageInstance
from .utils import word_count


class Strategy(Protocol):
    name: str
    def run(self, instance: MessageInstance) -> Candidate: ...


def _empty_timing() -> dict[str, float]:
    return {
        "generation_time_ms": 0.0,
        "constraint_eval_time_ms": 0.0,
        "semantic_eval_time_ms": 0.0,
        "total_time_ms": 0.0,
    }


def _round_timing(timing: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 3) for key, value in timing.items()}


def _add_evaluation_timing(timing: dict[str, float], evaluation: Evaluation) -> None:
    timing["constraint_eval_time_ms"] += evaluation.constraint_eval_time_ms
    timing["semantic_eval_time_ms"] += evaluation.semantic_eval_time_ms


def _candidate(
    text: str,
    evaluation: Evaluation,
    attempts: int,
    timing: dict[str, float],
    attempt_trace: list[dict[str, object]] | None = None,
) -> Candidate:
    metadata: dict[str, object] = _round_timing(timing)
    if attempt_trace is not None:
        metadata["attempt_trace"] = attempt_trace
    return Candidate(text=text, evaluation=evaluation, attempts=attempts, metadata=metadata)


def _is_better(candidate: Candidate, best: Candidate | None) -> bool:
    if best is None or best.evaluation is None:
        return True
    if candidate.evaluation is None:
        return False

    cand_ev = candidate.evaluation
    best_ev = best.evaluation
    if cand_ev.valid != best_ev.valid:
        return cand_ev.valid
    if cand_ev.violations != best_ev.violations:
        return cand_ev.violations < best_ev.violations
    return cand_ev.total_score > best_ev.total_score


def _feedback(results: list[ConstraintResult]) -> list[str]:
    feedback: list[str] = []
    for result in results:
        if result.passed:
            continue
        missing = f"; faltante={', '.join(result.missing)}" if result.missing else ""
        feedback.append(f"{result.name}: {result.details}{missing}")
    return feedback


def _trace_entry(kind: str, attempt: int, text: str, evaluation: Evaluation) -> dict[str, object]:
    return {
        "kind": kind,
        "attempt": attempt,
        "valid": evaluation.valid,
        "violations": evaluation.violations,
        "word_count": word_count(text),
        "failed": _feedback(evaluation.constraint_results),
        "text": text,
    }


@dataclass
class DirectStrategy:
    generator: BaseGenerator
    evaluator: MessageEvaluator
    name: str = "direct"

    def run(self, instance: MessageInstance) -> Candidate:
        timing = _empty_timing()
        trace: list[dict[str, object]] = []
        start = time.perf_counter()
        text = self.generator.generate(instance, attempt=1)
        timing["generation_time_ms"] += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        evaluation = self.evaluator.evaluate(text, instance)
        _add_evaluation_timing(timing, evaluation)
        trace.append(_trace_entry("generate", 1, text, evaluation))
        timing["total_time_ms"] += (time.perf_counter() - start) * 1000 + timing["generation_time_ms"]
        return _candidate(text, evaluation, attempts=1, timing=timing, attempt_trace=trace)


@dataclass
class VerifyStrategy:
    generator: BaseGenerator
    evaluator: MessageEvaluator
    max_attempts: int = 3
    name: str = "verify"

    def run(self, instance: MessageInstance) -> Candidate:
        total_start = time.perf_counter()
        timing = _empty_timing()
        trace: list[dict[str, object]] = []
        best: Candidate | None = None
        attempts_used = 0

        for attempt in range(1, self.max_attempts + 1):
            attempts_used = attempt
            start = time.perf_counter()
            text = self.generator.generate(instance, attempt=attempt)
            timing["generation_time_ms"] += (time.perf_counter() - start) * 1000

            evaluation = self.evaluator.evaluate(text, instance)
            _add_evaluation_timing(timing, evaluation)
            trace.append(_trace_entry("generate", attempt, text, evaluation))
            cand = _candidate(text, evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)
            if _is_better(cand, best):
                best = cand
            if evaluation.valid:
                timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
                return _candidate(text, evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)

        timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
        if best is None:
            metadata: dict[str, object] = _round_timing(timing)
            metadata["attempt_trace"] = trace
            return Candidate(text="", attempts=attempts_used, metadata=metadata)
        return _candidate(best.text, best.evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)  # type: ignore[arg-type]


@dataclass
class RepairStrategy:
    generator: BaseGenerator
    evaluator: MessageEvaluator
    max_attempts: int = 3
    name: str = "repair"

    def run(self, instance: MessageInstance) -> Candidate:
        total_start = time.perf_counter()
        timing = _empty_timing()
        trace: list[dict[str, object]] = []

        start = time.perf_counter()
        text = self.generator.generate(instance, attempt=1)
        timing["generation_time_ms"] += (time.perf_counter() - start) * 1000
        attempts = 1

        evaluation = self.evaluator.evaluate(text, instance)
        _add_evaluation_timing(timing, evaluation)
        trace.append(_trace_entry("generate", attempts, text, evaluation))
        current = _candidate(text, evaluation, attempts=attempts, timing=timing, attempt_trace=trace)
        best = current

        while attempts < self.max_attempts and current.evaluation is not None and not current.evaluation.valid:
            feedback = _feedback(current.evaluation.constraint_results)
            start = time.perf_counter()
            repaired_text = self.generator.repair(current.text, instance, feedback)
            timing["generation_time_ms"] += (time.perf_counter() - start) * 1000
            attempts += 1

            evaluation = self.evaluator.evaluate(repaired_text, instance)
            _add_evaluation_timing(timing, evaluation)
            trace.append(_trace_entry("repair", attempts, repaired_text, evaluation))
            current = _candidate(repaired_text, evaluation, attempts=attempts, timing=timing, attempt_trace=trace)
            if _is_better(current, best):
                best = current
            if evaluation.valid:
                timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
                return _candidate(current.text, evaluation, attempts=attempts, timing=timing, attempt_trace=trace)

        timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
        return _candidate(best.text, best.evaluation, attempts=attempts, timing=timing, attempt_trace=trace)  # type: ignore[arg-type]


@dataclass
class MultiCandidateStrategy:
    generator: BaseGenerator
    evaluator: MessageEvaluator
    max_attempts: int = 3
    population_size: int = 6
    name: str = "multi"

    def run(self, instance: MessageInstance) -> Candidate:
        total_start = time.perf_counter()
        timing = _empty_timing()
        trace: list[dict[str, object]] = []
        best: Candidate | None = None
        attempts_used = 0

        for attempt in range(1, self.population_size + 1):
            attempts_used = attempt
            start = time.perf_counter()
            text = self.generator.generate(instance, attempt=attempt)
            timing["generation_time_ms"] += (time.perf_counter() - start) * 1000

            evaluation = self.evaluator.evaluate(text, instance)
            _add_evaluation_timing(timing, evaluation)
            trace.append(_trace_entry("candidate", attempt, text, evaluation))
            cand = _candidate(text, evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)
            if _is_better(cand, best):
                best = cand

        assert best is not None
        current = best
        while attempts_used < self.max_attempts and current.evaluation is not None and not current.evaluation.valid:
            feedback = _feedback(current.evaluation.constraint_results)
            start = time.perf_counter()
            repaired_text = self.generator.repair(current.text, instance, feedback)
            timing["generation_time_ms"] += (time.perf_counter() - start) * 1000
            attempts_used += 1

            evaluation = self.evaluator.evaluate(repaired_text, instance)
            _add_evaluation_timing(timing, evaluation)
            trace.append(_trace_entry("repair_best", attempts_used, repaired_text, evaluation))
            current = _candidate(repaired_text, evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)
            if _is_better(current, best):
                best = current
            if evaluation.valid:
                timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
                return _candidate(current.text, evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)

        timing["total_time_ms"] = (time.perf_counter() - total_start) * 1000
        return _candidate(best.text, best.evaluation, attempts=attempts_used, timing=timing, attempt_trace=trace)  # type: ignore[arg-type]


def build_strategy(
    name: str,
    generator: BaseGenerator,
    evaluator: MessageEvaluator,
    max_attempts: int = 3,
    population_size: int = 3,
) -> Strategy:
    if name == "direct":
        return DirectStrategy(generator, evaluator)
    if name == "verify":
        return VerifyStrategy(generator, evaluator, max_attempts=max_attempts)
    if name == "repair":
        return RepairStrategy(generator, evaluator, max_attempts=max_attempts)
    if name == "multi":
        return MultiCandidateStrategy(generator, evaluator, max_attempts=max_attempts, population_size=population_size)
    raise ValueError(f"Estrategia desconocida: {name}")
