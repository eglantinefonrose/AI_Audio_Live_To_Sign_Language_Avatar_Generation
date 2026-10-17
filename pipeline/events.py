"""Structures d'événements circulant dans le pipeline.

Ces dataclasses constituent le contrat de données entre les composants. Elles
ne dépendent d'aucune bibliothèque tierce afin de pouvoir être consommées par
n'importe quel module en aval (y compris le futur moteur d'avatar 3D).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TranscriptEvent:
    """Phrase brute validée par la VAD, émise par le moteur STT."""

    text: str
    language: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GlossEvent:
    """Résultat de la traduction : gloses LSF dans l'ordre chronologique."""

    source_text: str
    glosses: list[str]
    language: str | None = None
    timestamp: float = field(default_factory=time.time)


__all__ = ["TranscriptEvent", "GlossEvent"]
