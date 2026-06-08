from __future__ import annotations

import random
from typing import List

from .models import MessageInstance
from .utils import poisson


MESSAGE_TYPES = ["disculpa", "solicitud", "invitacion", "aviso", "recordatorio", "felicitacion"]
RECIPIENTS = ["profesor", "cliente", "compañero", "equipo de trabajo", "coordinador", "amigo"]
TONES = ["formal", "informal", "empatico", "persuasivo"]
CONCEPT_BANK = {
    "disculpa": ["disculpa", "responsabilidad", "solución", "compromiso", "inconveniente"],
    "solicitud": ["petición", "razón", "plazo", "apoyo", "respuesta"],
    "invitacion": ["fecha", "participación", "actividad", "lugar", "confirmación"],
    "aviso": ["información", "cambio", "fecha", "organización", "coordinación"],
    "recordatorio": ["fecha", "pendiente", "entrega", "revisión", "tiempo"],
    "felicitacion": ["logro", "esfuerzo", "reconocimiento", "éxito", "motivación"],
}
FORBIDDEN_BANK = ["odio", "insulto", "mentira", "fracaso", "culpa", "amenaza", "tonto", "inútil"]


class InstanceSimulator:
    """Genera instancias aleatorias controladas para la fase experimental."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate(self, n: int) -> List[MessageInstance]:
        return [self.generate_one(i) for i in range(1, n + 1)]

    def generate_balanced(self, per_difficulty: int) -> List[MessageInstance]:
        instances: List[MessageInstance] = []
        idx = 1
        for difficulty in ["baja", "media", "alta"]:
            for _ in range(per_difficulty):
                instances.append(self.generate_one(idx, difficulty=difficulty))
                idx += 1
        return instances

    def generate_one(self, idx: int = 1, difficulty: str | None = None) -> MessageInstance:
        if difficulty is None:
            difficulty = self.rng.choices(["baja", "media", "alta"], weights=[0.35, 0.45, 0.20])[0]
        if difficulty not in {"baja", "media", "alta"}:
            raise ValueError(f"Dificultad desconocida: {difficulty}")
        msg_type = self.rng.choice(MESSAGE_TYPES)
        recipient = self.rng.choice(RECIPIENTS)
        tone = self.rng.choice(TONES)

        if difficulty == "baja":
            min_words = self.rng.choice([25, 30, 35])
            max_words = min_words + self.rng.choice([25, 30, 35])
            required_k = max(1, poisson(1.2, self.rng))
            forbidden_k = self.rng.choice([0, 1])
        elif difficulty == "media":
            min_words = self.rng.choice([35, 40, 45])
            max_words = min_words + self.rng.choice([15, 20, 25])
            required_k = max(2, poisson(2.0, self.rng))
            forbidden_k = self.rng.choice([1, 2])
        else:
            min_words = self.rng.choice([45, 50, 55])
            max_words = min_words + self.rng.choice([8, 12, 15])
            required_k = max(3, poisson(3.0, self.rng))
            forbidden_k = self.rng.choice([2, 3])

        concepts = self.rng.sample(CONCEPT_BANK[msg_type], k=min(required_k, len(CONCEPT_BANK[msg_type])))
        forbidden = self.rng.sample(FORBIDDEN_BANK, k=forbidden_k)
        must_gratitude = self.rng.random() < (0.25 if difficulty != "alta" else 0.50)

        raw = (
            f"Genera un mensaje de {msg_type} para {recipient}, con tono {tone}, "
            f"entre {min_words} y {max_words} palabras. Debe incluir {concepts} "
            f"y evitar {forbidden}."
        )
        if must_gratitude:
            raw += " Debe terminar con agradecimiento."

        return MessageInstance(
            instance_id=f"inst_{idx:04d}",
            message_type=msg_type,
            recipient=recipient,
            tone=tone,
            min_words=min_words,
            max_words=max_words,
            required_concepts=concepts,
            forbidden_terms=forbidden,
            required_structure=["saludo", "cuerpo", "cierre"],
            must_end_with_gratitude=must_gratitude,
            difficulty=difficulty,
            raw_request=raw,
        )
