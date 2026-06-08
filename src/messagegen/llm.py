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


PROMPT_VERSION = "balanced-hard-length-v5"


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
            "Eres un generador de mensajes en español. "
            "Debes cumplir primero las restricciones objetivas de longitud, conceptos obligatorios "
            "y términos prohibidos. "
            "Responde SOLO con el mensaje final, sin explicaciones, sin comillas, sin listas, "
            "sin markdown y sin decir que cumpliste las reglas. "
            "Escribe mensajes claros, compactos, naturales y adecuados al destinatario."
        )

    @staticmethod
    def _target_words(instance: MessageInstance) -> int:
        width = max(0, instance.max_words - instance.min_words)
        # Un objetivo algo por debajo del centro reduce salidas apenas largas sin
        # empujar los rangos estrechos por debajo del mínimo.
        return min(instance.max_words, instance.min_words + max(2, round(width * 0.45)))

    @staticmethod
    def _sentence_plan(instance: MessageInstance) -> str:
        target = GroqGenerator._target_words(instance)
        if target <= 42:
            return "3 oraciones breves"
        if target <= 58:
            return "4 oraciones breves"
        return "4 oraciones de longitud media"

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
            f"Longitud: entre {instance.min_words} y {instance.max_words} palabras\n"
            f"Objetivo recomendado: cerca de {GroqGenerator._target_words(instance)} palabras\n"
            f"Forma recomendada: {GroqGenerator._sentence_plan(instance)}\n"
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
        gratitude_rule = (
            "La última oración debe ser un agradecimiento claro."
            if instance.must_end_with_gratitude
            else "Cierra de forma adecuada."
        )
        return (
            "Genera un único mensaje que cumpla exactamente estas restricciones:\n"
            f"{self._format_instance(instance)}\n\n"
            "Reglas:\n"
            f"- {variation}\n"
            f"- Usa {self._sentence_plan(instance)} y evita frases largas.\n"
            f"- Apunta a unas {self._target_words(instance)} palabras; nunca salgas del rango {instance.min_words}-{instance.max_words}.\n"
            "- Si dudas, prefiere un mensaje más corto y directo antes que uno elaborado.\n"
            "- Copia cada concepto obligatorio exactamente como aparece, con sus tildes si las tiene.\n"
            "- No lo sustituyas por verbos, plurales, sinónimos ni fechas concretas.\n"
            "- Ejemplo: si el concepto es 'confirmación', debe aparecer 'confirmación', no solo 'confirmar' o 'confirme'.\n"
            "- No uses términos prohibidos, ni siquiera para decir que no los usarás.\n"
            "- Evita repetir conceptos, elogios o justificaciones.\n"
            "- No uses la frase 'Debe considerarse'.\n"
            "- No menciones las reglas ni la cantidad de palabras.\n"
            "- Si la estructura pide saludo, abre con una forma natural de dirigirte al destinatario.\n"
            "- Si la estructura pide cuerpo, incluye desarrollo central suficiente y coherente.\n"
            "- Si la estructura pide cierre, termina con una despedida, conclusión o cierre comunicativo natural.\n"
            f"- {gratitude_rule}\n"
            "- Responde únicamente con el mensaje final."
        )

    def _repair_prompt(self, text: str, instance: MessageInstance, feedback: list[str]) -> str:
        formatted_feedback = "\n".join(f"- {item}" for item in feedback) if feedback else "- Sin detalles."
        return (
            "Repara el siguiente mensaje para que cumpla todas las restricciones. "
            "Conserva la intención y deja una versión compacta y natural.\n\n"
            f"Restricciones:\n{self._format_instance(instance)}\n\n"
            f"Mensaje actual:\n{text}\n\n"
            f"Fallos detectados por el verificador:\n{formatted_feedback}\n\n"
            "Instrucciones:\n"
            "- Devuelve solo el mensaje reparado.\n"
            f"- Usa {self._sentence_plan(instance)}.\n"
            f"- Apunta a unas {self._target_words(instance)} palabras y respeta siempre el rango {instance.min_words}-{instance.max_words}.\n"
            "- Si el mensaje es largo, recorta primero repeticiones, elogios genéricos, incisos y fórmulas extensas de cortesía.\n"
            "- Si el mensaje es corto, añade solo una frase breve y útil.\n"
            f"- Deben quedar visibles estos conceptos exactos: {self._format_terms(instance.required_concepts)}.\n"
            "- Copia conceptos faltantes exactamente; no los cambies por verbos, plurales, sinónimos ni ejemplos.\n"
            f"- Estos términos no pueden aparecer: {self._format_terms(instance.forbidden_terms)}.\n"
            "- Si aparece un término prohibido, elimínalo o reemplázalo por otra idea que no contenga esa palabra.\n"
            "- Si falla saludo, abre con una forma natural de dirigirte al destinatario.\n"
            "- Si falla cuerpo, agrega desarrollo central claro y coherente.\n"
            "- Si falla cierre, termina con una despedida, conclusión o cierre comunicativo natural.\n"
            "- Si falla cierre_agradecimiento, haz que la última parte exprese agradecimiento claro.\n"
            "- No uses listas ni explicaciones."
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
