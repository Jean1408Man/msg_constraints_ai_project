from __future__ import annotations

import json
import re
import time
from typing import Any, List

from .constraints import check_deterministic
from .llm import GroqConnectionError, GroqGenerator
from .models import ConstraintResult, Evaluation, MessageInstance
from .utils import normalize, word_count


OPEN_CONSTRAINTS = ("saludo", "cuerpo", "cierre", "cierre_agradecimiento")

FORMAL_MARKERS = ["estimado", "estimada", "usted", "le solicito", "agradezco", "atentamente", "cordialmente"]
INFORMAL_MARKERS = ["hola", "oye", "gracias", "un abrazo", "nos vemos", "genial"]
EMPATHY_MARKERS = ["comprendo", "entiendo", "lamento", "disculpa", "disculpe", "agradezco", "valoro"]
PERSUASIVE_MARKERS = ["beneficio", "oportunidad", "propongo", "conviene", "mejor", "solucion"]


def _extract_json(raw: str, label: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise GroqConnectionError(f"Groq no devolvió JSON evaluable para {label}: {raw[:200]}")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise GroqConnectionError(f"JSON inválido para {label}: {raw[:200]}") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "si", "sí", "yes"}
    return bool(value)


def _as_missing_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or value == "":
        return []
    return [str(value).strip()]


def _split_segments(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]


class GroqOpenConstraintJudge:
    """Evalúa restricciones lingüísticas abiertas mediante un juez LLM."""

    def __init__(self, model: str | None = None, groq_base_url: str | None = None):
        self.client = GroqGenerator(model=model, base_url=groq_base_url, temperature=0.0)
        self._cache: dict[tuple[str, str], list[ConstraintResult]] = {}

    def evaluate(self, text: str, instance: MessageInstance) -> list[ConstraintResult]:
        cache_key = (instance.to_json(), text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un juez técnico de restricciones lingüísticas abiertas en español. "
                    "Devuelve solamente JSON válido, sin markdown ni explicaciones externas."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(text, instance),
            },
        ]
        raw = self.client.chat(messages, max_completion_tokens=520)
        obj = _extract_json(raw, "restricciones abiertas")
        results = self._to_results(obj, instance)
        self._cache[cache_key] = results
        return results

    @staticmethod
    def _is_required(name: str, instance: MessageInstance) -> bool:
        if name == "cierre_agradecimiento":
            return instance.must_end_with_gratitude
        return name in set(instance.required_structure)

    def _to_results(self, obj: dict[str, Any], instance: MessageInstance) -> list[ConstraintResult]:
        results: list[ConstraintResult] = []
        for name in OPEN_CONSTRAINTS:
            if not self._is_required(name, instance):
                results.append(
                    ConstraintResult(
                        name=name,
                        passed=True,
                        details="no requerido",
                        source="llm_constraints",
                    )
                )
                continue

            data = obj.get(name, {})
            if not isinstance(data, dict):
                data = {}
            missing = _as_missing_list(data.get("missing"))
            passed = _as_bool(data.get("passed", False))
            details = str(data.get("details", "")).strip()
            if not details:
                details = "cumple" if passed else "no cumple la restricción abierta"
            results.append(
                ConstraintResult(
                    name=name,
                    passed=passed,
                    details=details,
                    source="llm_constraints",
                    missing=missing if not passed else [],
                )
            )
        return results

    @staticmethod
    def _prompt(text: str, instance: MessageInstance) -> str:
        required_open = {
            "saludo": "saludo" in instance.required_structure,
            "cuerpo": "cuerpo" in instance.required_structure,
            "cierre": "cierre" in instance.required_structure,
            "cierre_agradecimiento": instance.must_end_with_gratitude,
        }
        return (
            "Evalúa si el mensaje candidato cumple restricciones lingüísticas abiertas.\n"
            "No uses listas cerradas de frases aceptables. Acepta formulaciones naturales equivalentes.\n"
            "No evalúes longitud exacta, conceptos obligatorios ni términos prohibidos; otro módulo los verifica.\n\n"
            "Criterios:\n"
            "- saludo: el mensaje abre con una forma natural de dirigirse al destinatario.\n"
            "- cuerpo: existe un desarrollo central coherente, no solo saludo y cierre.\n"
            "- cierre: el mensaje termina con una despedida, cierre o conclusión comunicativa natural.\n"
            "- cierre_agradecimiento: si se requiere, la última parte del mensaje expresa agradecimiento claro.\n\n"
            "Devuelve exactamente este JSON, con booleanos JSON true/false:\n"
            "{"
            '"saludo":{"passed":true,"missing":[],"details":"..."},'
            '"cuerpo":{"passed":true,"missing":[],"details":"..."},'
            '"cierre":{"passed":true,"missing":[],"details":"..."},'
            '"cierre_agradecimiento":{"passed":true,"missing":[],"details":"..."}'
            "}\n\n"
            f"Restricciones abiertas requeridas: {json.dumps(required_open, ensure_ascii=False)}\n"
            f"Tipo de mensaje: {instance.message_type}\n"
            f"Destinatario: {instance.recipient}\n"
            f"Tono obligatorio: {instance.tone}\n"
            f"Mensaje candidato:\n{text}"
        )


class HeuristicOpenConstraintJudge:
    """Respaldo local aproximado para pruebas sin Groq.

    Este modo no intenta enumerar saludos ni agradecimientos; para resultados
    finales debe usarse `open_constraint_evaluator=llm`.
    """

    def evaluate(self, text: str, instance: MessageInstance) -> list[ConstraintResult]:
        segments = _split_segments(text)
        required = set(instance.required_structure)
        results: list[ConstraintResult] = []

        if "saludo" in required:
            first_words = word_count(segments[0]) if segments else 0
            ok = bool(segments) and len(segments) >= 2 and first_words <= 12
            results.append(
                ConstraintResult(
                    "saludo",
                    ok,
                    "primer segmento breve y separado" if ok else "no hay apertura breve separada",
                    "heuristic_open_constraints",
                    [] if ok else ["saludo"],
                )
            )
        else:
            results.append(ConstraintResult("saludo", True, "no requerido", "heuristic_open_constraints"))

        if "cuerpo" in required:
            ok = word_count(text) >= max(12, instance.min_words // 2) and len(segments) >= 2
            results.append(
                ConstraintResult(
                    "cuerpo",
                    ok,
                    "hay desarrollo central suficiente" if ok else "desarrollo central insuficiente",
                    "heuristic_open_constraints",
                    [] if ok else ["cuerpo"],
                )
            )
        else:
            results.append(ConstraintResult("cuerpo", True, "no requerido", "heuristic_open_constraints"))

        if "cierre" in required:
            last_words = word_count(segments[-1]) if segments else 0
            ok = len(segments) >= 2 and 1 <= last_words <= 18
            results.append(
                ConstraintResult(
                    "cierre",
                    ok,
                    "último segmento breve" if ok else "no hay cierre breve separado",
                    "heuristic_open_constraints",
                    [] if ok else ["cierre"],
                )
            )
        else:
            results.append(ConstraintResult("cierre", True, "no requerido", "heuristic_open_constraints"))

        if instance.must_end_with_gratitude:
            results.append(
                ConstraintResult(
                    "cierre_agradecimiento",
                    False,
                    "requiere juez LLM; no se valida con listas cerradas en modo heurístico",
                    "heuristic_open_constraints",
                    ["cierre_agradecimiento"],
                )
            )
        else:
            results.append(
                ConstraintResult("cierre_agradecimiento", True, "no requerido", "heuristic_open_constraints")
            )

        return results


class GroqSemanticJudge:
    """Evalúa calidad semántica mediante una rúbrica respondida en JSON."""

    def __init__(self, model: str | None = None, groq_base_url: str | None = None):
        self.client = GroqGenerator(model=model, base_url=groq_base_url, temperature=0.0)
        self._cache: dict[tuple[str, str], tuple[float, List[str]]] = {}

    def score(self, text: str, instance: MessageInstance) -> tuple[float, List[str]]:
        cache_key = (instance.to_json(), text)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un evaluador técnico de mensajes en español. "
                    "Devuelve solamente JSON válido, sin markdown ni explicaciones externas."
                ),
            },
            {
                "role": "user",
                "content": self._prompt(text, instance),
            },
        ]
        raw = self.client.chat(messages, max_completion_tokens=320)
        obj = _extract_json(raw, "evaluación semántica")
        naturalness = self._score_value(obj.get("naturalidad"))
        tone = self._score_value(obj.get("tono"))
        coherence = self._score_value(obj.get("coherencia"))
        intent = self._score_value(obj.get("intencion"))
        score = 0.25 * naturalness + 0.25 * tone + 0.25 * coherence + 0.25 * intent
        notes = [
            "evaluador_semantico=llm",
            f"naturalidad={naturalness:.2f}",
            f"tono={tone:.2f}",
            f"coherencia={coherence:.2f}",
            f"intencion={intent:.2f}",
        ]
        comment = str(obj.get("comentario", "")).strip()
        if comment:
            notes.append(f"comentario_llm={comment[:180]}")
        result = (max(0.0, min(1.0, score)), notes)
        self._cache[cache_key] = result
        return result

    @staticmethod
    def _prompt(text: str, instance: MessageInstance) -> str:
        return (
            "Evalúa el siguiente mensaje candidato para una tarea de generación bajo restricciones.\n"
            "No decidas restricciones duras como longitud exacta, conceptos obligatorios, términos prohibidos, "
            "saludo, cuerpo, cierre o agradecimiento final; esas ya las verifica otro módulo. "
            "Evalúa solamente calidad semántica y comunicativa.\n\n"
            "Rúbrica, todos los valores deben estar entre 0.0 y 1.0:\n"
            "- naturalidad: qué tan humano y fluido suena.\n"
            "- tono: ajuste al tono obligatorio.\n"
            "- coherencia: claridad interna y continuidad del mensaje.\n"
            "- intencion: ajuste al tipo de mensaje y destinatario.\n\n"
            "Devuelve exactamente este JSON:\n"
            '{"naturalidad":0.0,"tono":0.0,"coherencia":0.0,"intencion":0.0,"comentario":"..."}\n\n'
            f"Tipo de mensaje: {instance.message_type}\n"
            f"Destinatario: {instance.recipient}\n"
            f"Tono obligatorio: {instance.tone}\n"
            f"Mensaje candidato:\n{text}"
        )

    @staticmethod
    def _score_value(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class MessageEvaluator:
    """Evalúa restricciones duras y criterios blandos.

    Las restricciones objetivas se verifican con código determinista. Las
    restricciones lingüísticas abiertas se evalúan por LLM por defecto, porque
    no se pueden enumerar todas las formas válidas de saludar, cerrar o agradecer.
    """

    def __init__(
        self,
        open_constraint_evaluator: str = "llm",
        semantic_evaluator: str = "llm",
        model: str | None = None,
        groq_base_url: str | None = None,
    ):
        if open_constraint_evaluator not in {"heuristic", "llm"}:
            raise ValueError(f"Evaluador de restricciones abiertas desconocido: {open_constraint_evaluator}")
        if semantic_evaluator not in {"heuristic", "llm"}:
            raise ValueError(f"Evaluador semántico desconocido: {semantic_evaluator}")

        self.open_constraint_evaluator = open_constraint_evaluator
        self.semantic_evaluator = semantic_evaluator
        self._open_judge = (
            GroqOpenConstraintJudge(model=model, groq_base_url=groq_base_url)
            if open_constraint_evaluator == "llm"
            else HeuristicOpenConstraintJudge()
        )
        self._llm_judge = (
            GroqSemanticJudge(model=model, groq_base_url=groq_base_url)
            if semantic_evaluator == "llm"
            else None
        )

    def evaluate(self, text: str, instance: MessageInstance) -> Evaluation:
        constraint_start = time.perf_counter()
        constraint_results = check_deterministic(text, instance)
        open_results = self._open_judge.evaluate(text, instance)
        constraint_results.extend(open_results)
        constraint_eval_time_ms = (time.perf_counter() - constraint_start) * 1000

        semantic_start = time.perf_counter()
        semantic_score, notes = self.semantic_score(text, instance)
        semantic_eval_time_ms = (time.perf_counter() - semantic_start) * 1000

        hard_score = sum(r.passed for r in constraint_results) / len(constraint_results)
        total_score = 0.70 * hard_score + 0.30 * semantic_score
        valid = all(r.passed for r in constraint_results)
        notes = [f"evaluador_restricciones_abiertas={self.open_constraint_evaluator}", *notes]
        return Evaluation(
            hard_score=round(hard_score, 4),
            semantic_score=round(semantic_score, 4),
            total_score=round(total_score, 4),
            valid=valid,
            constraint_results=constraint_results,
            open_constraint_evaluator=self.open_constraint_evaluator,
            semantic_evaluator=self.semantic_evaluator,
            constraint_eval_time_ms=round(constraint_eval_time_ms, 3),
            semantic_eval_time_ms=round(semantic_eval_time_ms, 3),
            notes=notes,
        )

    def semantic_score(self, text: str, instance: MessageInstance) -> tuple[float, List[str]]:
        if self.semantic_evaluator == "llm":
            assert self._llm_judge is not None
            return self._llm_judge.score(text, instance)
        return self.heuristic_semantic_score(text, instance)

    def heuristic_semantic_score(self, text: str, instance: MessageInstance) -> tuple[float, List[str]]:
        nt = normalize(text)
        notes: List[str] = ["evaluador_semantico=heuristic"]
        score = 0.0

        tone_score = self._tone_score(nt, instance.tone)
        if tone_score < 0.6:
            notes.append(f"tono {instance.tone} débil")
        score += 0.40 * tone_score

        wc = word_count(text)
        midpoint = (instance.min_words + instance.max_words) / 2
        length_fit = max(0.0, 1.0 - abs(wc - midpoint) / max(midpoint, 1))
        score += 0.20 * length_fit

        connectors = ["porque", "por ello", "ademas", "sin embargo", "por tanto", "para", "con el fin"]
        connector_score = min(1.0, sum(c in nt for c in connectors) / 2)
        punctuation_score = 1.0 if any(p in text for p in ".,;:") else 0.5
        score += 0.20 * ((connector_score + punctuation_score) / 2)

        type_score = self._type_score(nt, instance.message_type)
        if type_score < 0.6:
            notes.append(f"intención de {instance.message_type} poco explícita")
        score += 0.20 * type_score

        return max(0.0, min(1.0, score)), notes

    def _tone_score(self, nt: str, tone: str) -> float:
        tone = normalize(tone)
        if tone == "formal":
            return min(1.0, 0.35 + 0.18 * sum(m in nt for m in FORMAL_MARKERS))
        if tone == "informal":
            return min(1.0, 0.40 + 0.18 * sum(m in nt for m in INFORMAL_MARKERS))
        if tone == "empatico":
            return min(1.0, 0.30 + 0.20 * sum(m in nt for m in EMPATHY_MARKERS))
        if tone == "persuasivo":
            return min(1.0, 0.30 + 0.20 * sum(m in nt for m in PERSUASIVE_MARKERS))
        return 0.7

    def _type_score(self, nt: str, msg_type: str) -> float:
        msg_type = normalize(msg_type)
        markers = {
            "disculpa": ["disculpa", "disculpe", "lamento", "perdon"],
            "solicitud": ["solicito", "pido", "quisiera", "agradeceria"],
            "invitacion": ["invito", "invitamos", "acompanes", "participar"],
            "aviso": ["informo", "avisamos", "comunico", "notifico"],
            "recordatorio": ["recuerdo", "recordamos", "pendiente", "fecha"],
            "felicitacion": ["felicito", "felicitaciones", "enhorabuena", "logro"],
        }
        selected = markers.get(msg_type, [])
        if not selected:
            return 0.7
        return min(1.0, 0.25 + 0.25 * sum(m in nt for m in selected))
