"""Outil CLI : transcription parole -> gloses LSF en temps réel.

Affiche en continu :  [STT Raw] <phrase> -> [LSF Gloss] ['GLOSE', ...]

Usage :
    python cli.py                      # micro par défaut, config par défaut
    python cli.py videos/demo.mp4      # transcrit un fichier vidéo/audio
    python cli.py --list-devices       # liste les micros disponibles
    python cli.py --model qwen2.5:3b --language fr --stt-device cpu

    # Tester UNIQUEMENT la transcription (sans Ollama) et voir la découpe
    # en phrases envoyée au traducteur :
    python cli.py videos/demo.mp4 --stt-only
    python cli.py --stt-only --vad-pause 1.5   # micro, seuil de pause ajusté
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from config import CONFIG, replace
from pipeline.events import GlossEvent, TranscriptEvent

log = logging.getLogger("cli")


def setup_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_config(args: argparse.Namespace):
    """Applique les surcharges CLI sur la configuration par défaut."""
    cfg = CONFIG
    if args.model:
        cfg = cfg.with_overrides(ollama=replace(cfg.ollama, model=args.model))
    if args.postprocess:
        cfg = cfg.with_overrides(ollama=replace(cfg.ollama, postprocess=True))
    if args.ollama_host:
        cfg = cfg.with_overrides(
            ollama=replace(cfg.ollama, host=args.ollama_host)
        )
    if args.stt_device:
        cfg = cfg.with_overrides(stt=replace(cfg.stt, device=args.stt_device))
    if args.vad_pause is not None:
        cfg = cfg.with_overrides(
            stt=replace(cfg.stt, vad_pause_seconds=args.vad_pause)
        )
    if args.max_words is not None:
        cfg = cfg.with_overrides(
            stt=replace(cfg.stt, max_sentence_words=args.max_words)
        )
    if args.hf_repo:
        cfg = cfg.with_overrides(stt=replace(cfg.stt, hf_repo=args.hf_repo))
    if args.language:
        cfg = cfg.with_overrides(
            pipeline=replace(cfg.pipeline, language=args.language)
        )
    if args.input_device is not None:
        cfg = cfg.with_overrides(
            audio=replace(cfg.audio, device=args.input_device)
        )
    if args.verbose:
        cfg = cfg.with_overrides(log_level=logging.DEBUG)
    return cfg


async def print_transcript(transcript: TranscriptEvent) -> None:
    """Affiche le sous-titre DÈS que la phrase est transcrite (avant le LLM)."""
    print(f"[STT Raw] {transcript.text}", flush=True)


async def print_gloss(transcript: TranscriptEvent, gloss: GlossEvent) -> None:
    """Affiche les gloses LSF une fois la traduction du LLM terminée."""
    print(f"       -> [LSF Gloss] {gloss.glosses}\n", flush=True)


async def run(cfg, media: str | None = None, realtime: bool = False) -> int:
    # Imports tardifs : permet `--list-devices` sans torch/moshi/ollama.
    from avatar_adapter.interface import ConsoleAvatarRenderer
    from lsf_translator.translator import OllamaUnavailableError
    from pipeline.orchestrator import Pipeline
    from stt_engine.audio_capture import AudioDeviceError
    from stt_engine.engine import STTUnavailableError
    from stt_engine.file_source import AudioFileError, FileAudioSource

    # Source audio : fichier vidéo/audio si fourni, sinon micro.
    audio_source = (
        FileAudioSource(media, cfg.audio, realtime=realtime) if media else None
    )

    pipeline = Pipeline(
        cfg,
        renderers=[ConsoleAvatarRenderer()],
        on_transcript=print_transcript,   # sous-titre dès la transcription
        on_gloss=print_gloss,             # gloses une fois le LLM terminé
        audio_source=audio_source,
    )

    try:
        await pipeline.start()
    except OllamaUnavailableError as exc:
        log.error("Ollama indisponible : %s", exc)
        return 2
    except (AudioDeviceError, AudioFileError) as exc:
        log.error("Problème audio : %s", exc)
        return 3
    except STTUnavailableError as exc:
        log.error("Moteur STT indisponible : %s", exc)
        return 4

    try:
        if media:
            # Flux fini : on transcrit jusqu'à la fin du média puis on s'arrête.
            cadence = "temps réel" if realtime else "aussi vite que possible"
            print(f"\n🎬  Transcription de « {media} » ({cadence})…\n", flush=True)
            await pipeline.wait()
            print("\n✅  Transcription terminée.", flush=True)
        else:
            # Flux micro infini : tourne jusqu'à interruption (Ctrl+C).
            print("\n🎙️  Parlez… (Ctrl+C pour quitter)\n", flush=True)
            while True:
                await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nArrêt demandé…", flush=True)
    finally:
        await pipeline.stop()
    return 0


def run_stt_only(cfg, media: str | None = None, realtime: bool = False) -> int:
    """Exerce SEUL le composant de transcription et affiche les phrases.

    N'interroge PAS Ollama : montre la découpe (VAD) qui alimente le traducteur.
    Illustre l'usage autonome du composant `Transcriber` (générateur).
    """
    from stt_engine.audio_capture import AudioCapture, AudioDeviceError
    from stt_engine.engine import STTUnavailableError
    from stt_engine.file_source import AudioFileError, FileAudioSource
    from stt_engine.transcriber import Transcriber

    source = (
        FileAudioSource(media, cfg.audio, realtime=realtime)
        if media else AudioCapture(cfg.audio)
    )
    transcriber = Transcriber(cfg.stt, cfg.audio)
    language = None if cfg.pipeline.language == "auto" else cfg.pipeline.language

    try:
        transcriber.load()   # charge le modèle Kyutai (peut être long)
    except STTUnavailableError as exc:
        log.error("Moteur STT indisponible : %s", exc)
        return 4

    if media:
        print(f"\n🎬  STT seul sur « {media} » "
              f"(pause VAD = {cfg.stt.vad_pause_seconds}s, pas de traduction)…",
              flush=True)
    else:
        print(f"\n🎙️  STT seul — parlez "
              f"(pause VAD = {cfg.stt.vad_pause_seconds}s, Ctrl+C pour quitter)…",
              flush=True)

    count = 0
    try:
        # Le composant gère lui-même le cycle de vie de la source.
        for sentence in transcriber.transcribe(source, language=language):
            count += 1
            print(f"\n  ┌─ phrase #{count:02d}  →  [envoyée au LLM]\n"
                  f"  └─ {sentence.text!r}", flush=True)
    except (AudioDeviceError, AudioFileError) as exc:
        log.error("Problème audio : %s", exc)
        return 3
    except KeyboardInterrupt:
        print("\nArrêt demandé…", flush=True)

    print(f"\n✅  Terminé — {count} phrase(s) détectée(s).", flush=True)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcription parole -> gloses LSF (micro ou fichier)."
    )
    p.add_argument("media", nargs="?",
                   help="Chemin d'un fichier vidéo/audio à transcrire "
                        "(mp4, mov, mkv, wav, mp3…). Si omis : micro en direct.")
    p.add_argument("--list-devices", action="store_true",
                   help="Liste les périphériques d'entrée audio et quitte.")
    p.add_argument("--model", help="Modèle Ollama (défaut: config).")
    p.add_argument("--postprocess", action="store_true",
                   help="Post-traitement déterministe des gloses (lsf_translator/postprocess.py).")
    p.add_argument("--ollama-host", help="URL de l'instance Ollama.")
    p.add_argument("--hf-repo", help="Dépôt HF du modèle STT Kyutai.")
    p.add_argument("--stt-device", choices=["auto", "cpu", "cuda", "mps"],
                   help="Périphérique de calcul STT.")
    p.add_argument("--language", choices=["auto", "fr", "en"],
                   help="Langue source (défaut: auto).")
    p.add_argument("--stt-only", action="store_true",
                   help="Teste la transcription seule (sans Ollama) et affiche "
                        "la découpe en phrases envoyée au traducteur.")
    p.add_argument("--vad-pause", type=float, metavar="SECONDES",
                   help="Seuil de silence (s) validant une phrase complète "
                        "(défaut: %.1f). Pour régler la découpe VAD."
                        % CONFIG.stt.vad_pause_seconds)
    p.add_argument("--max-words", type=int, metavar="N",
                   help="Taille cible (en mots) des segments envoyés au LLM : "
                        "les phrases plus longues sont recoupées sur la "
                        "ponctuation (défaut: %d ; 0 = désactivé)."
                        % CONFIG.stt.max_sentence_words)
    p.add_argument("--realtime", action="store_true",
                   help="Mode fichier : cadence la lecture à la vitesse réelle "
                        "du média (simulation live) au lieu d'aussi vite que "
                        "possible.")
    p.add_argument("--input-device", type=int,
                   help="Index du périphérique micro (voir --list-devices).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Journalisation détaillée (DEBUG).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.list_devices:
        setup_logging(logging.INFO)
        from stt_engine.audio_capture import AudioDeviceError, list_input_devices
        try:
            devices = list_input_devices()
        except AudioDeviceError as exc:
            log.error("%s", exc)
            return 1
        print("Périphériques d'entrée disponibles :")
        for dev in devices:
            print(f"  [{dev['index']}] {dev['name']}")
        return 0

    cfg = build_config(args)
    setup_logging(cfg.log_level)

    if args.stt_only:
        # Transcription seule, sans Ollama ni pipeline asyncio.
        return run_stt_only(cfg, media=args.media, realtime=args.realtime)

    try:
        return asyncio.run(run(cfg, media=args.media, realtime=args.realtime))
    except KeyboardInterrupt:  # pragma: no cover
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
