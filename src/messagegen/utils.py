from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, List

from .models import MessageInstance


def normalize(text: str) -> str:
    text = text.lower().strip()
    replacements = str.maketrans("áéíóúüñ", "aeiouun")
    text = text.translate(replacements)
    text = re.sub(r"\s+", " ", text)
    return text


def words(text: str) -> List[str]:
    return re.findall(r"[\wáéíóúÁÉÍÓÚñÑüÜ]+", text, flags=re.UNICODE)


def word_count(text: str) -> int:
    return len(words(text))


def contains_concept(text: str, concept: str) -> bool:
    """Búsqueda simple con normalización; se puede reemplazar por embeddings/LLM."""
    nt = normalize(text)
    nc = normalize(concept)
    if nc in nt:
        return True
    # fallback: todas las palabras relevantes del concepto aparecen en algún punto
    cwords = [w for w in words(nc) if len(w) > 2]
    return bool(cwords) and all(w in nt for w in cwords)


def read_jsonl(path: str | Path) -> List[MessageInstance]:
    instances: List[MessageInstance] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(MessageInstance.from_json(line))
    return instances


def write_jsonl(path: str | Path, instances: Iterable[MessageInstance]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for inst in instances:
            f.write(inst.to_json() + "\n")


def poisson(lmbda: float, rng: random.Random) -> int:
    """Algoritmo de Knuth para variable Poisson."""
    import math

    L = math.exp(-lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1
