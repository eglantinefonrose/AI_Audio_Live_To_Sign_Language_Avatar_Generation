"""Moteur STT streaming basé sur Kyutai STT (pile Moshi).

Le modèle `kyutai/stt-1b-en_fr` transcrit en flux continu et émet ses jetons de
texte au rythme de la parole : les silences se traduisent par des jetons de
remplissage (padding). On exploite ce comportement — piloté par la VAD
sémantique de Kyutai — pour segmenter en phrases : lorsqu'aucun nouveau mot
n'apparaît pendant `vad_pause_seconds`, la phrase accumulée est considérée
complète et émise.

Dépendances (`torch`, `moshi`) importées paresseusement : le reste du pipeline
reste importable même sans elles (utile pour les tests unitaires de la CLI).
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator

import numpy as np

from config import AudioConfig, STTConfig

log = logging.getLogger(__name__)


class STTUnavailableError(RuntimeError):
    """Levée quand `torch`/`moshi` ne sont pas installés ou chargeables."""


# Jetons texte à ignorer lors de la reconstruction de la transcription.
# 0 = padding (silence/pause), 3 = fin de segment selon la convention Moshi STT.
_IGNORED_TEXT_TOKENS = {0, 3}


class STTEngine:
    """Encapsule le modèle Moshi STT et la logique de segmentation VAD.

    Conçu pour tourner dans un thread dédié : `run()` est bloquant et pousse les
    phrases complètes via un callback thread-safe fourni par l'orchestrateur.
    """

    def __init__(self, config: STTConfig, audio_config: AudioConfig) -> None:
        self.config = config
        self.audio_config = audio_config
        self._loaded = False
        # Objets Moshi, résolus dans load().
        self._torch = None
        self._mimi = None
        self._lm_gen = None
        self._text_tokenizer = None
        self._device = None
        self._frame_size = 0
        # Durée d'une trame Mimi (s) ; sert au compteur de pause VAD.
        self._frame_seconds = (
            audio_config.block_size / audio_config.sample_rate
        )

    # -- chargement des modèles -----------------------------------------
    def _resolve_device(self, torch) -> str:
        requested = self.config.device
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            log.warning(
                "Utilisation du backend MPS (expérimental dans Moshi). "
                "Pour de meilleures performances sur M1, voir la variante MLX "
                "dans le README."
            )
            return "mps"
        return "cpu"

    def load(self) -> None:
        """Importe et charge les modèles. Bloquant et coûteux (à lancer tôt)."""
        if self._loaded:
            return
        try:
            import torch
            from moshi.models import loaders
            from moshi.models.lm import LMGen
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            raise STTUnavailableError(
                "torch/moshi introuvables. Installez les dépendances "
                "(voir requirements.txt / README)."
            ) from exc

        self._torch = torch
        self._device = self._resolve_device(torch)
        log.info(
            "Chargement du modèle STT '%s' sur %s…",
            self.config.hf_repo,
            self._device,
        )

        checkpoint = loaders.CheckpointInfo.from_hf_repo(self.config.hf_repo)
        dtype = torch.float32 if self._device != "cuda" else torch.bfloat16
        self._mimi = checkpoint.get_mimi(device=self._device)
        self._text_tokenizer = checkpoint.get_text_tokenizer()
        lm = checkpoint.get_moshi(device=self._device, dtype=dtype)
        self._lm_gen = LMGen(
            lm, temp=self.config.temp, temp_text=self.config.temp_text
        )
        self._frame_size = int(self._mimi.sample_rate / self._mimi.frame_rate)
        self._loaded = True
        log.info("Modèle STT chargé (trame=%d échantillons).", self._frame_size)

    # -- décodage d'un jeton texte --------------------------------------
    def _decode_token(self, token_id: int) -> str:
        if token_id in _IGNORED_TEXT_TOKENS:
            return ""
        try:
            piece = self._text_tokenizer.id_to_piece(token_id)
        except Exception:  # jeton de contrôle non décodable
            return ""
        # SentencePiece : "▁" marque une frontière de mot.
        return piece.replace("▁", " ")

    # -- boucle de streaming --------------------------------------------
    def run(
        self,
        frame_iter: Iterator[np.ndarray],
        on_sentence: Callable[[str], None],
        should_stop: Callable[[], bool],
    ) -> None:
        """Consomme les trames audio et émet les phrases complètes.

        :param frame_iter: itérateur bloquant de trames `float32`.
        :param on_sentence: callback thread-safe appelé avec chaque phrase.
        :param should_stop: prédicat interrogé régulièrement pour l'arrêt.
        """
        if not self._loaded:
            self.load()

        torch = self._torch
        pause_frames = max(
            1, int(round(self.config.vad_pause_seconds / self._frame_seconds))
        )
        buffer = ""
        frames_since_word = 0

        def flush() -> None:
            nonlocal buffer
            sentence = buffer.strip()
            if len(sentence) >= self.config.min_sentence_chars:
                on_sentence(sentence)
            buffer = ""

        log.info("Boucle STT démarrée (pause VAD = %d trames).", pause_frames)
        with torch.no_grad(), self._mimi.streaming(1), self._lm_gen.streaming(1):
            for frame in frame_iter:
                if should_stop():
                    break
                audio = torch.from_numpy(
                    np.ascontiguousarray(frame, dtype=np.float32)
                ).to(self._device)[None, None]
                codes = self._mimi.encode(audio)  # [1, K, T]
                for t in range(codes.shape[-1]):
                    tokens = self._lm_gen.step(codes[:, :, t : t + 1])
                    if tokens is None:
                        # Phase de latence initiale du modèle.
                        continue
                    piece = self._decode_token(int(tokens[0, 0, 0].item()))
                    if piece.strip():
                        buffer += piece
                        frames_since_word = 0
                    else:
                        frames_since_word += 1
                        # Silence prolongé + phrase en cours => phrase complète.
                        if frames_since_word >= pause_frames and buffer.strip():
                            flush()

        # Émet le reste éventuel à l'arrêt.
        flush()
        log.info("Boucle STT terminée.")


__all__ = ["STTEngine", "STTUnavailableError"]
