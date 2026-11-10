"""Ré-export des contrats de données du pipeline.

Les types sont désormais définis dans le module neutre `contracts` (partagé par
les deux composants). On les ré-exporte ici pour compatibilité : les imports
historiques `from pipeline.events import TranscriptEvent, GlossEvent` continuent
de fonctionner et désignent exactement les mêmes classes.
"""
from __future__ import annotations

from src.contracts import GlossEvent, Sentence, TranscriptEvent

__all__ = ["TranscriptEvent", "GlossEvent", "Sentence"]
