"""Source audio lisant un fichier vidéo/audio (hors ligne) via ffmpeg.

Décode n'importe quel conteneur géré par ffmpeg (mp4, mov, mkv, webm, wav,
mp3…) en PCM float32 mono au débit attendu par le STT, et l'expose comme un
flux de trames. L'interface (`start` / `frames` / `stop`) est identique à celle
de `AudioCapture`, si bien que l'orchestrateur n'a pas à savoir d'où vient
l'audio : micro ou fichier, c'est transparent.

Contrairement au micro, le flux est *fini* : l'itérateur `frames()` se termine
à la fin du fichier, ce qui laisse le pipeline se vider puis s'arrêter seul.
La segmentation VAD du moteur STT repose sur le nombre de trames (durée audio),
pas sur l'horloge murale : décoder plus vite que le temps réel ne fausse donc
pas le découpage en phrases.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from src.config import AudioConfig

log = logging.getLogger(__name__)

# float32 little-endian => 4 octets par échantillon.
_BYTES_PER_SAMPLE = 4


class AudioFileError(RuntimeError):
    """Levée si le fichier est introuvable, illisible, ou si ffmpeg échoue."""


class FileAudioSource:
    """Lit un fichier média et fournit des trames mono `float32`."""

    def __init__(
        self,
        path: str | Path,
        audio_config: AudioConfig,
        realtime: bool = False,
    ) -> None:
        """:param realtime: si vrai, débite les trames à la vitesse réelle du
        média (simulation « live ») au lieu d'aussi vite que possible.
        """
        self.path = Path(path).expanduser()
        self.config = audio_config
        self.realtime = realtime
        # Durée audio représentée par une trame (s).
        self._frame_seconds = audio_config.block_size / audio_config.sample_rate
        self._proc: subprocess.Popen | None = None

    # -- cycle de vie ----------------------------------------------------
    def start(self) -> None:
        if shutil.which("ffmpeg") is None:
            raise AudioFileError(
                "ffmpeg introuvable. Installez-le : `brew install ffmpeg`."
            )
        if not self.path.exists():
            raise AudioFileError(f"Fichier introuvable : {self.path}")
        if not self.path.is_file():
            raise AudioFileError(f"Le chemin n'est pas un fichier : {self.path}")

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "error",
            "-i", str(self.path),
            "-vn",                       # on ignore la piste vidéo
            "-f", "f32le",               # PCM float32 little-endian brut
            "-acodec", "pcm_f32le",
            "-ac", str(self.config.channels),
            "-ar", str(self.config.sample_rate),
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception as exc:  # pragma: no cover - dépend du système
            raise AudioFileError(f"Échec du lancement de ffmpeg : {exc}") from exc

        log.info(
            "Décodage du fichier '%s' via ffmpeg (%d Hz, %d canal).",
            self.path.name,
            self.config.sample_rate,
            self.config.channels,
        )

    def frames(self) -> Iterator[np.ndarray]:
        """Itérateur de trames `float32` de taille `block_size`.

        Se termine à la fin du fichier (EOF de la sortie ffmpeg).
        """
        if self._proc is None or self._proc.stdout is None:
            raise AudioFileError("frames() appelé avant start().")

        frame_bytes = self.config.block_size * _BYTES_PER_SAMPLE
        read_size = frame_bytes * 16  # lectures groupées pour l'efficacité
        buf = b""
        stdout = self._proc.stdout

        # Cadencement temps réel : on vise `start + i * frame_seconds` pour la
        # i-ème trame. On ne dort que si l'on est en avance ; si le STT est plus
        # lent que le temps réel, aucun ralentissement n'est ajouté.
        start = time.monotonic()
        index = 0

        def pace() -> None:
            nonlocal index
            if self.realtime:
                delay = (start + index * self._frame_seconds) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            index += 1

        while True:
            chunk = stdout.read(read_size)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= frame_bytes:
                raw, buf = buf[:frame_bytes], buf[frame_bytes:]
                pace()
                yield np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)

        # Dernière trame partielle éventuelle, complétée par des zéros.
        if buf:
            tail = np.frombuffer(buf, dtype="<f4")
            frame = np.zeros(self.config.block_size, dtype=np.float32)
            frame[: len(tail)] = tail
            pace()
            yield frame

        self._finalize()

    def _finalize(self) -> None:
        if self._proc is None:
            return
        err = b""
        if self._proc.stderr is not None:
            try:
                err = self._proc.stderr.read() or b""
            except Exception:  # pragma: no cover
                pass
        code = self._proc.wait()
        if code:
            message = err.decode("utf-8", "replace").strip() or "erreur inconnue"
            raise AudioFileError(f"ffmpeg a échoué (code {code}) : {message}")
        log.info("Fin du décodage du fichier '%s'.", self.path.name)

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
        self._proc = None
        log.info("Source fichier arrêtée.")


__all__ = ["FileAudioSource", "AudioFileError"]
