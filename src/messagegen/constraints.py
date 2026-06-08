from __future__ import annotations

import re
from typing import List

from .models import ConstraintResult, MessageInstance
from .utils import contains_concept, normalize, word_count


def check_length(text: str, instance: MessageInstance) -> ConstraintResult:
    n = word_count(text)
    ok = instance.min_words <= n <= instance.max_words
    return ConstraintResult(
        name="longitud",
        passed=ok,
        details=f"{n} palabras; esperado entre {instance.min_words} y {instance.max_words}",
        source="deterministic",
        missing=[] if ok else [f"rango {instance.min_words}-{instance.max_words}"],
    )


def check_required_concepts(text: str, instance: MessageInstance) -> ConstraintResult:
    missing = [c for c in instance.required_concepts if not contains_concept(text, c)]
    return ConstraintResult(
        name="conceptos_obligatorios",
        passed=not missing,
        details="faltan: " + ", ".join(missing) if missing else "todos presentes",
        source="deterministic",
        missing=missing,
    )


def check_forbidden_terms(text: str, instance: MessageInstance) -> ConstraintResult:
    nt = normalize(text)
    found = []
    for term in instance.forbidden_terms:
        pattern = r"\b" + re.escape(normalize(term)) + r"\b"
        if re.search(pattern, nt):
            found.append(term)
    return ConstraintResult(
        name="terminos_prohibidos",
        passed=not found,
        details="aparecen: " + ", ".join(found) if found else "ninguno encontrado",
        source="deterministic",
        missing=found,
    )


def check_deterministic(text: str, instance: MessageInstance) -> List[ConstraintResult]:
    """Comprueba solo restricciones objetivas y enumerables.

    Las restricciones abiertas de lenguaje natural, como reconocer saludos,
    cuerpo, cierre y agradecimiento final, se evalúan en `evaluator.py` con LLM.
    """

    return [
        check_length(text, instance),
        check_required_concepts(text, instance),
        check_forbidden_terms(text, instance),
    ]


def check_all(text: str, instance: MessageInstance) -> List[ConstraintResult]:
    """Alias de compatibilidad: desde ahora `check_all` significa checks objetivos."""

    return check_deterministic(text, instance)
