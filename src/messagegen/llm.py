from __future__ import annotations

import json
import os
import re
import time
import http.client
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

from .models import MessageInstance
from .utils import contains_concept, normalize, word_count


PROMPT_VERSION = "constraint-first-prompts-v15"


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, instance: MessageInstance, attempt: int = 1) -> str:
        raise NotImplementedError

    def repair(self, text: str, instance: MessageInstance, feedback: list[str]) -> str:
        """Reparación genérica. Las subclases pueden mejorarla."""
        return text


class GroqConnectionError(RuntimeError):
    """Error legible cuando Groq no está disponible o falta configuración."""


class GroqRateLimitError(GroqConnectionError):
    """Error específico para límites de uso de Groq."""


def _dotenv_paths() -> list[Path]:
    project_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / ".env", project_root / ".env"]
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return paths


def load_dotenv() -> None:
    """Carga variables simples KEY=VALUE desde .env sin dependencia externa."""
    for path in _dotenv_paths():
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class GroqGenerator(BaseGenerator):
    """Generador real basado en la API compatible OpenAI de Groq.

    Usa la librería estándar de Python contra:
        https://api.groq.com/openai/v1/chat/completions

    Requiere GROQ_API_KEY y GROQ_MODEL en variables de entorno o en .env.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        timeout: int = 120,
        max_retries: int = 6,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL")
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise GroqConnectionError(
                "Falta GROQ_API_KEY. Crea un archivo .env con GROQ_API_KEY=... "
                "o exporta la variable antes de usar --generator groq."
            )
        if not self.model:
            raise GroqConnectionError(
                "Falta GROQ_MODEL. Agrega GROQ_MODEL=... al .env o usa --model."
            )

    def generate(self, instance: MessageInstance, attempt: int = 1) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._generation_prompt(instance, attempt)},
        ]
        text = self._chat(messages)
        return self._clean_message(text)

    def repair(self, text: str, instance: MessageInstance, feedback: list[str]) -> str:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": self._repair_prompt(text, instance, feedback),
            },
        ]
        repaired = self._chat(messages)
        return self._clean_message(repaired)

    def healthcheck(self) -> str:
        """Devuelve una respuesta corta del modelo para verificar configuración."""
        msg = self._chat([
            {"role": "system", "content": "Responde solamente con la palabra OK."},
            {"role": "user", "content": "Prueba de conexión."},
        ])
        return self._clean_message(msg)

    def chat(self, messages: list[dict[str, str]], max_completion_tokens: int = 220) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": 0.9,
            "max_completion_tokens": max_completion_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        raw = self._post_chat(data)

        try:
            obj = json.loads(raw)
            return obj.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        except json.JSONDecodeError as exc:
            raise GroqConnectionError(f"Respuesta no válida de Groq: {raw[:200]}") from exc

    def _chat(self, messages: list[dict[str, str]]) -> str:
        return self.chat(messages)

    def _post_chat(self, data: bytes) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            sleep_seconds = 0.75 * (attempt + 1)
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Connection": "close",
                    "User-Agent": "messagegen-constraints/0.1",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and self._is_daily_or_hard_rate_limit(raw):
                    raise GroqRateLimitError(self._format_http_error(exc.code, raw)) from exc
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise GroqConnectionError(self._format_http_error(exc.code, raw)) from exc
                if exc.code == 429:
                    sleep_seconds = self._rate_limit_sleep_seconds(
                        exc.headers,
                        raw,
                        fallback=max(2.0, 1.5 * (attempt + 1)),
                    )
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
            except TimeoutError as exc:
                last_error = exc
            except (http.client.RemoteDisconnected, http.client.HTTPException, ConnectionResetError) as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(sleep_seconds)

        detail = f" Detalle técnico: {last_error}" if last_error else ""
        raise GroqConnectionError(
            "No se pudo completar la llamada a Groq tras varios intentos. "
            "Puede ser un corte temporal de red o de la API; vuelve a ejecutar el comando."
            f"{detail}"
        )

    @staticmethod
    def _format_http_error(status: int, raw: str) -> str:
        try:
            obj = json.loads(raw)
            detail = obj.get("error", {}).get("message") or obj.get("message") or raw[:200]
        except json.JSONDecodeError:
            detail = raw[:200]
        if status == 401:
            return "Groq rechazó la autenticación. Revisa GROQ_API_KEY en .env."
        if status == 404:
            return f"Groq no encontró el recurso o modelo configurado. Revisa GROQ_MODEL. Detalle: {detail}"
        return f"Groq devolvió HTTP {status}. Detalle: {detail}"

    @staticmethod
    def _is_daily_or_hard_rate_limit(raw: str) -> bool:
        text = raw.lower()
        return (
            "rate limit reached" in text
            and (
                "tokens per day" in text
                or "requests per day" in text
                or "per day" in text
                or "daily" in text
                or "(tpd)" in text
                or "(rpd)" in text
            )
        )

    @staticmethod
    def _rate_limit_sleep_seconds(headers: object, raw: str, fallback: float) -> float:
        candidates = [fallback]
        retry_after = headers.get("Retry-After") if hasattr(headers, "get") else None
        if retry_after:
            try:
                candidates.append(float(retry_after))
            except ValueError:
                pass

        match = re.search(
            r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|seconds?|m|min|minutes?)",
            raw,
            flags=re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit in {"ms", "millisecond", "milliseconds"}:
                value = value / 1000
            elif unit in {"m", "min", "minute", "minutes"}:
                value = value * 60
            candidates.append(value)

        return min(90.0, max(1.0, max(candidates) + 0.25))

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Eres un generador de mensajes en español bajo restricciones verificadas automáticamente. "
            "La validez depende primero de restricciones duras: longitud, conceptos obligatorios literales "
            "y términos prohibidos. La naturalidad importa, pero nunca por encima de esas restricciones. "
            "Responde SOLO con el mensaje final, sin explicaciones, sin comillas, sin listas, "
            "sin markdown y sin decir que cumpliste las reglas. "
            "El mensaje debe sonar natural y tener saludo, cuerpo y cierre cuando se pidan."
        )

    @staticmethod
    def _target_words(instance: MessageInstance) -> int:
        span = instance.max_words - instance.min_words
        target = instance.min_words + round(span * 0.75)
        return max(instance.min_words, min(instance.max_words - 1, target))

    @staticmethod
    def _sentence_plan(instance: MessageInstance) -> str:
        target = GroqGenerator._target_words(instance)
        sentence_count = 3 if instance.max_words <= 63 else 4
        target_per_sentence = max(8, round(target / sentence_count))
        max_per_sentence = max(8, instance.max_words // sentence_count)
        low = max(8, min(max_per_sentence, target_per_sentence - 2))
        high = max(low, min(max_per_sentence, target_per_sentence + 1))
        return f"exactamente {sentence_count} oraciones desarrolladas de {low} a {high} palabras cada una"

    @staticmethod
    def _structure_rules(instance: MessageInstance) -> list[str]:
        required = set(instance.required_structure)
        rules: list[str] = []
        if "saludo" in required:
            rules.append(
                "La primera oración debe comenzar con un saludo claro al destinatario, "
                "por ejemplo 'Estimado cliente,' u 'Hola cliente,'."
            )
        if "cuerpo" in required:
            rules.append(
                "El cuerpo debe desarrollar una idea central con motivo, contexto, beneficio, petición "
                "o siguiente paso; no uses frases sueltas sin desarrollo."
            )
        if "cierre" in required:
            if instance.must_end_with_gratitude:
                rules.append(
                    "La última oración debe funcionar como cierre y expresar agradecimiento claro; "
                    "no termines con una descripción o palabra de relleno."
                )
            else:
                rules.append(
                    "La última oración debe cerrar la interacción de forma inequívoca con una despedida, "
                    "invitación, conclusión o siguiente paso; no termines con una descripción o palabra de relleno."
                )
        elif instance.must_end_with_gratitude:
            rules.append("La última parte del mensaje debe expresar agradecimiento claro.")
        return rules

    @staticmethod
    def _format_structure_rules(instance: MessageInstance) -> str:
        rules = GroqGenerator._structure_rules(instance)
        if not rules:
            return "- No hay estructura abierta adicional."
        return "\n".join(f"- {rule}" for rule in rules)

    @staticmethod
    def _length_repair_instruction(text: str, instance: MessageInstance) -> str:
        current = word_count(text)
        target = GroqGenerator._target_words(instance)
        low, high = GroqGenerator._repair_word_window(text, instance)
        if current < instance.min_words:
            missing = instance.min_words - current
            if missing <= 2:
                return (
                    f"El mensaje actual tiene {current} palabras y está CORTO por solo {missing}. "
                    f"Conserva casi todo y agrega {missing + 1} palabra(s) útiles, sin añadir una frase completa. "
                    f"Rango operativo para esta reparación: {low}-{high} palabras."
                )
            midpoint = (low + high) // 2
            needed = max(missing, midpoint - current)
            return (
                f"El mensaje actual tiene {current} palabras y está CORTO. "
                f"No basta con agregar {missing} palabra(s) al final: reescribe una versión completa "
                f"con alrededor de {needed} palabras netas adicionales. "
                f"Rango operativo para esta reparación: {low}-{high} palabras. "
                "La reparación debe ser claramente más amplia, no un parche mínimo."
            )
        if current > instance.max_words:
            extra = current - instance.max_words
            return (
                f"El mensaje actual tiene {current} palabras y está LARGO. "
                f"Elimina al menos {extra} palabras secundarias y apunta a {target}. "
                f"Rango operativo para esta reparación: {low}-{high} palabras. "
                "Reescribe una versión más compacta; no agregues ideas nuevas ni amplíes el texto."
            )
        return (
            f"El mensaje actual tiene {current} palabras, ya está dentro del rango. "
            "Conserva la longitud aproximada mientras corriges los demás fallos."
        )

    @staticmethod
    def _repair_word_window(text: str, instance: MessageInstance) -> tuple[int, int]:
        current = word_count(text)
        target = GroqGenerator._target_words(instance)
        if current < instance.min_words:
            missing = instance.min_words - current
            if missing <= 2:
                return instance.min_words, min(instance.max_words, instance.min_words + 3)
            if missing <= 8:
                return instance.min_words, min(instance.max_words, instance.min_words + 6)
            return target, instance.max_words
        if current > instance.max_words:
            return instance.min_words, min(target, instance.max_words - 1)
        return instance.min_words, instance.max_words

    @staticmethod
    def _literal_check_summary(text: str, instance: MessageInstance) -> str:
        missing = [concept for concept in instance.required_concepts if not contains_concept(text, concept)]
        nt = normalize(text)
        found_forbidden = [term for term in instance.forbidden_terms if normalize(term) in nt]
        missing_text = GroqGenerator._format_terms(missing) if missing else "ninguno"
        forbidden_text = GroqGenerator._format_terms(found_forbidden) if found_forbidden else "ninguno"
        return (
            f"Conceptos obligatorios faltantes detectados: {missing_text}\n"
            f"Términos prohibidos detectados: {forbidden_text}"
        )

    @staticmethod
    def _format_terms(values: list[str]) -> str:
        if not values:
            return "ninguno"
        return ", ".join(f"'{value}'" for value in values)

    @staticmethod
    def _format_instance(instance: MessageInstance) -> str:
        return (
            f"Tipo de mensaje: {instance.message_type}\n"
            f"Destinatario: {instance.recipient}\n"
            f"Tono obligatorio: {instance.tone}\n"
            f"Longitud obligatoria: mínimo {instance.min_words}, máximo {instance.max_words} palabras\n"
            f"Meta de longitud: {GroqGenerator._target_words(instance)} palabras\n"
            f"Plan recomendado: {GroqGenerator._sentence_plan(instance)}\n"
            f"Conceptos obligatorios literales: {GroqGenerator._format_terms(instance.required_concepts)}\n"
            f"Términos prohibidos ausentes: {GroqGenerator._format_terms(instance.forbidden_terms)}\n"
            f"Estructura requerida: {instance.required_structure}\n"
            f"Debe terminar con agradecimiento: {instance.must_end_with_gratitude}"
        )

    def _generation_prompt(self, instance: MessageInstance, attempt: int) -> str:
        variation = (
            "Haz una versión natural y directa."
            if attempt == 1
            else "Haz una versión alternativa, más natural y menos mecánica que las anteriores."
        )
        return (
            "Genera un único mensaje que cumpla exactamente estas restricciones:\n"
            f"{self._format_instance(instance)}\n\n"
            "Reglas duras de validez:\n"
            f"- {variation}\n"
            f"- Escribe {self._sentence_plan(instance)}.\n"
            f"- Cuenta mentalmente las palabras antes de responder y entrega entre {instance.min_words} "
            f"y {instance.max_words}; apunta a {self._target_words(instance)} palabras.\n"
            f"- Si el borrador mental queda con menos de {instance.min_words} palabras, agrega contexto útil antes de responder.\n"
            f"- Si el borrador mental queda con más de {instance.max_words} palabras, recorta antes de responder.\n"
            "- No respondas con oraciones telegráficas de 3 a 7 palabras; cada oración debe aportar información concreta.\n"
            "- Copia cada concepto obligatorio exactamente como aparece, con sus tildes si las tiene.\n"
            "- No sustituyas conceptos obligatorios por verbos, plurales, sinónimos ni fechas concretas.\n"
            "- Usa cada concepto obligatorio dentro de una oración natural, no como lista aislada.\n"
            "- Una aparición literal de cada concepto obligatorio basta; evita repetirlo en todas las oraciones.\n"
            "- Ejemplo: si el concepto es 'confirmación', debe aparecer 'confirmación', no solo 'confirmar' o 'confirme'.\n"
            "- No uses términos prohibidos, ni siquiera para decir que no los usarás.\n"
            "- No uses la frase 'Debe considerarse'.\n"
            "- No menciones las reglas ni la cantidad de palabras.\n"
            "\nReglas de estructura abierta:\n"
            f"{self._format_structure_rules(instance)}\n\n"
            "Checklist mental obligatorio antes de responder:\n"
            f"- Longitud dentro de {instance.min_words}-{instance.max_words} palabras.\n"
            f"- Conceptos literales presentes: {self._format_terms(instance.required_concepts)}.\n"
            f"- Términos prohibidos ausentes: {self._format_terms(instance.forbidden_terms)}.\n"
            "- Saludo, cuerpo, cierre y agradecimiento presentes cuando se pidan.\n"
            "- Responde únicamente con el mensaje final."
        )

    def _repair_prompt(self, text: str, instance: MessageInstance, feedback: list[str]) -> str:
        formatted_feedback = "\n".join(f"- {item}" for item in feedback) if feedback else "- Sin detalles."
        low, high = self._repair_word_window(text, instance)
        return (
            "Reescribe el mensaje actual para que sea VÁLIDO. Devuelve solo el mensaje final.\n\n"
            "Restricciones duras:\n"
            f"- Longitud final obligatoria: {instance.min_words}-{instance.max_words} palabras.\n"
            f"- Rango operativo para esta reparación: {low}-{high} palabras.\n"
            f"- Forma: {self._sentence_plan(instance)}.\n"
            f"- Conceptos exactos que deben aparecer: {self._format_terms(instance.required_concepts)}.\n"
            f"- Términos prohibidos que no pueden aparecer: {self._format_terms(instance.forbidden_terms)}.\n"
            "- Los conceptos obligatorios deben aparecer literalmente; no sirven sinónimos, verbos, plurales ni fechas concretas.\n\n"
            "Diagnóstico automático:\n"
            f"- {self._length_repair_instruction(text, instance)}\n"
            f"{self._literal_check_summary(text, instance)}\n"
            f"{formatted_feedback}\n\n"
            "Reglas de reescritura:\n"
            "- No repitas el mensaje actual si ya fue inválido.\n"
            "- Si está largo, elimina detalles secundarios y conserva solo saludo, propósito, conceptos obligatorios y cierre.\n"
            "- Si está corto por 1 o 2 palabras, haz un ajuste mínimo y no añadas una frase completa.\n"
            "- Si está corto por más de 2 palabras, agrega contexto útil dentro del rango operativo.\n"
            "- Si falta un concepto, inserta esa palabra exacta en una oración natural.\n"
            "- Si sobra un término prohibido, elimínalo por completo.\n"
            f"{self._format_structure_rules(instance)}\n\n"
            f"Mensaje actual:\n{text}"
        )

    @staticmethod
    def _clean_message(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip().strip('"').strip("'").strip()
        # Si el modelo devuelve prefijos tipo "Mensaje:", se eliminan.
        text = re.sub(r"^(mensaje|respuesta|versión)\s*:\s*", "", text, flags=re.IGNORECASE)
        return " ".join(text.split())


def build_generator(
    name: str = "groq",
    seed: int | None = None,
    model: str | None = None,
    groq_base_url: str | None = None,
    temperature: float = 0.2,
) -> BaseGenerator:
    _ = seed
    if name == "groq":
        return GroqGenerator(model=model, base_url=groq_base_url, temperature=temperature)
    raise ValueError(f"Generador desconocido: {name}")
