"""Composant 1 — Transcription de la parole en phrases (Kyutai STT + VAD).

Façade autonome au-dessus de `STTEngine`. Elle transforme une source audio
(micro ou fichier) en un flux de **phrases complètes** (`TranscriptEvent`), la
segmentation étant assurée par la VAD de Kyutai.

Deux usages, pour rester utilisable seul **ou** branché dans l'orchestrateur :

    # Autonome (générateur) :
    tr = Transcriber(cfg.stt, cfg.audio)
    for phrase in tr.transcribe(FileAudioSource("v.mp4", cfg.audio)):
        print(phrase.text)

    # Bas niveau (callback, bloquant) — utilisé par l'orchestrateur :
    tr.run(source, on_sentence=handler, should_stop=flag.is_set)

Ce composant **ne connaît pas** le composant de traduction : il ne produit que
des `TranscriptEvent` (l'interface neutre définie dans `contracts`).
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Iterator, Optional

from config import AudioConfig, STTConfig
from contracts import TranscriptEvent
from stt_engine.engine import STTEngine
from stt_engine.segmenter import split_into_segments

log = logging.getLogger(__name__)

_SENTINEL = object()


class Transcriber:
    """Composant de transcription : audio -> phrases segmentées (VAD)."""

    def __init__(self, config: STTConfig, audio_config: AudioConfig) -> None:
        self._config = config
        self._engine = STTEngine(config, audio_config)
        self._stop = threading.Event()

    # -- chargement ------------------------------------------------------
    def load(self) -> None:
        """Charge le modèle STT (bloquant, coûteux).

        Lève `STTUnavailableError` si torch/moshi sont absents.
        """
        self._engine.load()

    # -- API bas niveau (callback, bloquant) ----------------------------
    def run(
        self,
        source,
        on_sentence: Callable[[TranscriptEvent], None],
        should_stop: Optional[Callable[[], bool]] = None,
        language: str | None = None,
    ) -> None:
        """Transcrit `source` (déjà démarrée) et appelle `on_sentence` par phrase.

        Bloquant : conçu pour tourner dans un thread (c'est ce que fait
        l'orchestrateur). Ne gère **pas** le cycle de vie de la source — c'est
        à l'appelant de faire `source.start()` / `source.stop()`.
        """
        self.load()
        stop = should_stop or self._stop.is_set
        max_words = self._config.max_sentence_words

        def handle(text: str) -> None:
            # Un « pavé » de la VAD peut contenir plusieurs phrases : on le
            # recoupe en segments courts, chacun émis comme un TranscriptEvent.
            for segment in split_into_segments(text, max_words=max_words):
                on_sentence(TranscriptEvent(text=segment, language=language))

        self._engine.run(
            source.frames(),
            on_sentence=handle,
            should_stop=stop,
        )

    # -- API haut niveau (générateur) -----------------------------------
    def transcribe(
        self,
        source,
        language: str | None = None,
        should_stop: Optional[Callable[[], bool]] = None,
        manage_source: bool = True,
    ) -> Iterator[TranscriptEvent]:
        """Génère les phrases transcrites au fil de l'eau (usage autonome).

        :param manage_source: si vrai (défaut), démarre et arrête `source`
            automatiquement. Mettre à faux si l'appelant gère déjà la source.
        """
        self.load()
        if manage_source:
            source.start()

        q: "queue.Queue" = queue.Queue()

        def combined_stop() -> bool:
            return self._stop.is_set() or bool(should_stop and should_stop())

        def worker() -> None:
            try:
                self.run(source, on_sentence=q.put, should_stop=combined_stop,
                         language=language)
            except Exception as exc:  # remontée au consommateur
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        thread = threading.Thread(target=worker, name="transcriber", daemon=True)
        thread.start()
        try:
            while True:
                # Timeout court : laisse passer un KeyboardInterrupt du thread
                # principal même quand aucune phrase n'arrive (micro silencieux).
                try:
                    item = q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self._stop.set()
            if manage_source:
                source.stop()
            thread.join(timeout=10)

    def stop(self) -> None:
        """Demande l'arrêt de la transcription en cours."""
        self._stop.set()


__all__ = ["Transcriber"]
