from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import json


@dataclass
class MessageInstance:
    """Instancia del problema: solicitud + restricciones."""

    instance_id: str
    message_type: str
    recipient: str
    tone: str
    min_words: int
    max_words: int
    required_concepts: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    required_structure: List[str] = field(default_factory=lambda: ["saludo", "cuerpo", "cierre"])
    must_end_with_gratitude: bool = False
    difficulty: str = "media"
    raw_request: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MessageInstance":
        return MessageInstance(**data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "MessageInstance":
        return MessageInstance.from_dict(json.loads(line))


@dataclass
class ConstraintResult:
    name: str
    passed: bool
    details: str = ""
    source: str = "deterministic"
    missing: List[str] = field(default_factory=list)


@dataclass
class Evaluation:
    hard_score: float
    semantic_score: float
    total_score: float
    valid: bool
    constraint_results: List[ConstraintResult]
    open_constraint_evaluator: str = "llm"
    semantic_evaluator: str = "llm"
    constraint_eval_time_ms: float = 0.0
    semantic_eval_time_ms: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def violations(self) -> int:
        return sum(1 for r in self.constraint_results if not r.passed)


@dataclass
class Candidate:
    text: str
    evaluation: Optional[Evaluation] = None
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRow:
    instance_id: str
    strategy: str
    difficulty: str
    prompt_version: str
    query: str
    instance_json: str
    open_constraint_evaluator: str
    semantic_evaluator: str
    valid: bool
    hard_score: float
    semantic_score: float
    total_score: float
    violations: int
    attempts: int
    attempt_trace: str
    failed_constraints: str
    invalid_reason: str
    error_type: str
    error_message: str
    evaluation_notes: str
    generation_time_ms: float
    constraint_eval_time_ms: float
    semantic_eval_time_ms: float
    total_time_ms: float
    time_ms: float
    message: str
