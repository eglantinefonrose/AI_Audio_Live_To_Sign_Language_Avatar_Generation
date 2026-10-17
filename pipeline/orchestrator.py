"""Orchestrateur : relie STT -> Traducteur LSF -> consommateurs (avatars).

Modèle « écouteur d'événements » basé sur deux `asyncio.Queue` :

    [AudioCapture] --(thread)--> STTEngine
          |  on_sentence (thread-safe)
          v
    transcript_queue  --(tâche async)-->  LSFTranslator
          |                                      |
          |                                      v
          |                                gloss_queue  --(tâche async)-->
          |                                      |        renderers + CLI
          v

Le moteur STT (bloquant, CPU/GPU) tourne dans un thread dédié ; il publie sur
`transcript_queue` via `loop.call_soon_threadsafe`. Les étapes aval sont de
simples coroutines, ce qui garde chaque composant isolé et testable.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Awaitable, Callable, Sequence

from avatar_adapter.interface import AvatarRendererInterface
from config import AppConfig
from lsf_translator.translator import LSFTranslator
from pipeline.events import GlossEvent, TranscriptEvent
from stt_engine.audio_capture import AudioCapture
from stt_engine.engine import STTEngine

log = logging.getLogger(__name__)

# Sentinelles de fin de flux pour chaque file.
_TRANSCRIPT_DONE = object()
_GLOSS_DONE = object()

# Callback optionnel pour l'affichage CLI : (TranscriptEvent, GlossEvent).
OnGloss = Callable[[TranscriptEvent, GlossEvent], Awaitable[None] | None]


class Pipeline:
    """Assemble et pilote l'ensemble des composants."""

    def __init__(
        self,
        config: AppConfig,
        renderers: Sequence[AvatarRendererInterface] | None = None,
        on_gloss: OnGloss | None = None,
        audio_source=None,
    ) -> None:
        """:param audio_source: toute source exposant ``start/frames/stop``
        (ex. ``AudioCapture`` pour le micro ou ``FileAudioSource`` pour un
        fichier vidéo/audio). Par défaut : le micro.
        """
        self.config = config
        self.renderers: list[AvatarRendererInterface] = list(renderers or [])
        self.on_gloss = on_gloss

        self.transcript_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.pipeline.transcript_queue_size
        )
        self.gloss_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.pipeline.gloss_queue_size
        )

        self.audio = audio_source if audio_source is not None else AudioCapture(
            config.audio
        )
        self.stt = STTEngine(config.stt, config.audio)
        self.translator = LSFTranslator(config.ollama)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = threading.Event()
        self._stt_thread: threading.Thread | None = None
        self._tasks: list[asyncio.Task] = []

    # -- démarrage -------------------------------------------------------
    async def start(self) -> None:
        """Initialise les ressources puis lance threads et tâches.

        L'ordre importe : on valide d'abord les dépendances externes (Ollama,
        modèle STT, périphérique audio) pour échouer tôt et clairement.
        """
        self._loop = asyncio.get_running_loop()

        # 1) Backend LLM (échoue vite si Ollama est éteint).
        await self.translator.connect()

        # 2) Chargement du modèle STT (bloquant => executor).
        log.info("Chargement du moteur STT…")
        await self._loop.run_in_executor(None, self.stt.load)

        # 3) Renderers aval.
        for renderer in self.renderers:
            await renderer.start()

        # 4) Capture audio.
        self.audio.start()

        # 5) Thread STT + tâches asynchrones.
        self._stt_thread = threading.Thread(
            target=self._run_stt, name="stt-engine", daemon=True
        )
        self._stt_thread.start()

        self._tasks = [
            asyncio.create_task(self._translator_worker(), name="translator"),
            asyncio.create_task(self._render_worker(), name="renderer"),
        ]
        log.info("Pipeline démarré.")

    # -- thread STT ------------------------------------------------------
    def _run_stt(self) -> None:
        """Cible du thread STT : convertit les phrases en événements enfilés."""
        def emit(sentence: str) -> None:
            event = TranscriptEvent(
                text=sentence, language=self._source_language()
            )
            # Passage thread -> boucle asyncio de façon sûre.
            self._loop.call_soon_threadsafe(self._enqueue_transcript, event)

        try:
            self.stt.run(
                self.audio.frames(),
                on_sentence=emit,
                should_stop=self._stop_event.is_set,
            )
        except Exception as exc:  # pragma: no cover
            log.exception("Le moteur STT s'est arrêté sur une erreur : %s", exc)
        finally:
            self._loop.call_soon_threadsafe(
                self.transcript_queue.put_nowait, _TRANSCRIPT_DONE
            )

    def _source_language(self) -> str | None:
        lang = self.config.pipeline.language
        return None if lang == "auto" else lang

    def _enqueue_transcript(self, event: TranscriptEvent) -> None:
        try:
            self.transcript_queue.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("File de transcriptions saturée, phrase ignorée.")

    # -- étage traduction ------------------------------------------------
    async def _translator_worker(self) -> None:
        while True:
            item = await self.transcript_queue.get()
            if item is _TRANSCRIPT_DONE:
                await self.gloss_queue.put(_GLOSS_DONE)
                break
            transcript: TranscriptEvent = item
            glosses = await self.translator.translate(
                transcript.text, transcript.language
            )
            gloss_event = GlossEvent(
                source_text=transcript.text,
                glosses=glosses,
                language=transcript.language,
                timestamp=transcript.timestamp,
            )
            await self.gloss_queue.put((transcript, gloss_event))

    # -- étage sortie / rendu -------------------------------------------
    async def _render_worker(self) -> None:
        while True:
            item = await self.gloss_queue.get()
            if item is _GLOSS_DONE:
                break
            transcript, gloss_event = item

            if self.on_gloss is not None:
                result = self.on_gloss(transcript, gloss_event)
                if asyncio.iscoroutine(result):
                    await result

            for renderer in self.renderers:
                try:
                    await renderer.render(gloss_event)
                except Exception as exc:  # un renderer défaillant n'arrête rien
                    log.error("Erreur du renderer %r : %s", renderer, exc)

    # -- attente de fin de flux -----------------------------------------
    async def wait(self) -> None:
        """Bloque jusqu'à épuisement du flux (utile pour une source *finie*).

        Pour une source fichier, les tâches se terminent naturellement à la fin
        du média. Pour le micro (flux infini), cette méthode ne rend jamais la
        main : l'appelant utilise alors une boucle interruptible à la place.
        """
        if self._tasks:
            await asyncio.gather(*self._tasks)

    # -- arrêt -----------------------------------------------------------
    async def stop(self) -> None:
        log.info("Arrêt du pipeline…")
        self._stop_event.set()
        self.audio.stop()  # débloque l'itérateur de trames du thread STT

        if self._stt_thread is not None:
            await self._loop.run_in_executor(None, self._stt_thread.join, 10.0)

        # Laisse les tâches asynchrones drainer les files puis les annule.
        for task in self._tasks:
            try:
                await asyncio.wait_for(task, timeout=15.0)
            except asyncio.TimeoutError:
                task.cancel()
            except asyncio.CancelledError:  # pragma: no cover
                pass

        for renderer in self.renderers:
            try:
                await renderer.stop()
            except Exception as exc:  # pragma: no cover
                log.error("Erreur à l'arrêt du renderer %r : %s", renderer, exc)
        log.info("Pipeline arrêté.")


__all__ = ["Pipeline"]
