"""Capture micro via `sounddevice`, exposée comme un flux de trames numpy.

Le callback de `sounddevice` s'exécute dans un thread temps réel géré par
PortAudio : on y fait le minimum (copie + enfilage) et on expose les trames via
une `queue.Queue` thread-safe consommée par le moteur STT.
"""
from __future__ import annotations

import logging
import queue
from typing import Iterator

import numpy as np

try:  # import paresseux tolérant : la CLI peut démarrer sans PortAudio
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - dépend de l'environnement
    sd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from src.config import AudioConfig

log = logging.getLogger(__name__)

# Sentinelle enfilée pour signaler la fin du flux aux consommateurs.
_SENTINEL = object()


class AudioDeviceError(RuntimeError):
    """Levée quand aucun périphérique d'entrée exploitable n'est disponible."""


def list_input_devices() -> list[dict]:
    """Retourne la liste des périphériques d'entrée (nom + index)."""
    if sd is None:
        raise AudioDeviceError(
            f"sounddevice indisponible : {_IMPORT_ERROR}. "
            "Installez-le (voir README, PortAudio requis sur macOS)."
        )
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append({"index": idx, "name": dev["name"]})
    return devices


class AudioCapture:
    """Capture micro en continu et fournit un itérateur de trames mono."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._queue: "queue.Queue" = queue.Queue(maxsize=256)
        self._stream = None

    # -- callback PortAudio (thread temps réel) --------------------------
    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.warning("Statut de l'entrée audio : %s", status)
        # indata : (block_size, channels) float32 ; on ne garde que le mono.
        mono = indata[:, 0].copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # On préfère perdre une trame que de bloquer le thread audio.
            log.warning("File audio saturée, trame ignorée (STT trop lent ?).")

    # -- cycle de vie ----------------------------------------------------
    def start(self) -> None:
        if sd is None:
            raise AudioDeviceError(
                f"sounddevice indisponible : {_IMPORT_ERROR}. "
                "Installez-le (voir README, PortAudio requis sur macOS)."
            )
        if not list_input_devices():
            raise AudioDeviceError(
                "Aucun périphérique d'entrée audio détecté. "
                "Vérifiez votre micro et les autorisations macOS."
            )
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.block_size,
                dtype=self.config.dtype,
                device=self.config.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # erreur d'ouverture du périphérique
            raise AudioDeviceError(
                f"Impossible d'ouvrir le flux audio : {exc}"
            ) from exc
        log.info(
            "Capture audio démarrée (%d Hz, %d canal, bloc=%d).",
            self.config.sample_rate,
            self.config.channels,
            self.config.block_size,
        )

    def frames(self) -> Iterator[np.ndarray]:
        """Itérateur bloquant de trames `float32` de taille `block_size`."""
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        # Débloque un éventuel consommateur en attente.
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:  # pragma: no cover
            pass
        log.info("Capture audio arrêtée.")


__all__ = ["AudioCapture", "AudioDeviceError", "list_input_devices"]
