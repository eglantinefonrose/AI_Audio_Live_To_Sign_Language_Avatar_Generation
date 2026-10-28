"""Contrats de données partagés entre les composants (interface neutre).

Ces dataclasses forment l'**interface** entre les deux composants :

    [Transcription]  --TranscriptEvent-->  [Traduction LSF]  --GlossEvent-->

Elles vivent ici, dans un module neutre sans dépendance, pour que ni le
composant de transcription (`stt_engine`) ni celui de traduction
(`lsf_translator`) n'aient à s'importer l'un l'autre : chacun ne connaît que ces
types. C'est ce qui rend les deux composants réellement découplés et testables
séparément.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TranscriptEvent:
    """Une phrase complète produite par le composant de transcription.

    La segmentation (découpe en phrases) est assurée par la VAD de Kyutai.
    """

    text: str
    language: str | None = None
    timestamp: float = field(default_factory=time.time)


# Alias lisible : la sortie du composant de transcription est une « phrase ».
Sentence = TranscriptEvent


@dataclass
class GlossEvent:
    """Séquence de gloses LSF produite par le composant de traduction."""

    source_text: str
    glosses: list[str]
    language: str | None = None
    timestamp: float = field(default_factory=time.time)


__all__ = ["TranscriptEvent", "GlossEvent", "Sentence"]
