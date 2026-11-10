"""Configuration centralisée du pipeline de transcription LSF temps réel.

Tous les paramètres réglables (modèles, points de terminaison API, audio) sont
regroupés ici sous forme de dataclasses immuables afin que les modules métier
(`stt_engine`, `lsf_translator`, `pipeline`) restent découplés de la source de
configuration. Un seul objet `CONFIG` est exposé et peut être surchargé par la
CLI au démarrage (voir `cli.py`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class AudioConfig:
    """Paramètres de capture micro.

    Kyutai/Mimi attend un flux mono 24 kHz. Mimi fonctionne à 12.5 Hz, soit
    1920 échantillons par trame (24000 / 12.5) : on aligne `block_size` dessus
    pour fournir exactement une trame Mimi par bloc audio.
    """

    sample_rate: int = 24_000
    channels: int = 1
    block_size: int = 1_920
    dtype: str = "float32"
    device: int | None = None  # None => périphérique d'entrée par défaut


@dataclass(frozen=True)
class STTConfig:
    """Paramètres du moteur STT Kyutai (Moshi)."""

    hf_repo: str = "kyutai/stt-1b-en_fr"
    # auto | cpu | cuda | mps. Sur M1, `auto` tentera `mps` (expérimental dans
    # Moshi) sinon `cpu`. Voir le README pour la variante MLX, plus rapide.
    device: str = "auto"
    # Segmentation VAD : durée de silence (sans nouveau mot) qui valide une
    # phrase complète. Exploite le rythme d'émission des jetons du modèle, lui
    # même piloté par la VAD sémantique de Kyutai.
    vad_pause_seconds: float = 1.0
    # Longueur minimale (en caractères) avant d'autoriser un flush, pour éviter
    # d'émettre des bribes parasites.
    min_sentence_chars: int = 2
    # Découpe des phrases trop longues AVANT envoi au LLM : si la VAD produit un
    # « pavé » (locuteur sans pause), il est recoupé sur la ponctuation en
    # segments d'environ `max_sentence_words` mots, car les LLM décrochent sur
    # les entrées longues. Mettre à 0 pour désactiver le redécoupage.
    max_sentence_words: int = 25
    # Échantillonnage déterministe (latence + reproductibilité).
    temp: float = 0.0
    temp_text: float = 0.0


@dataclass(frozen=True)
class OllamaConfig:
    """Paramètres du backend LLM local (Ollama)."""

    host: str = "http://localhost:11434"
    # Doit être un modèle réellement installé (`ollama list`). Surchargeable
    # via --model. gemma2:9b retenu après comparaison : meilleur en français et
    # plus fidèle au texte source (~5.4 Go, tient sur M1 16 Go avec le STT).
    model: str = "gemma2:9b"
    request_timeout: float = 30.0
    # Maintient le modèle chargé entre deux appels => latence minimale.
    keep_alive: str = "10m"
    temperature: float = 0.0
    # Prompt de prod (règles + few-shot) ≈ 2 300 tokens : 2 048 tronquait le
    # début du prompt (les règles). 4 096 laisse la marge pour la réponse.
    num_ctx: int = 4_096
    # Post-traitement déterministe des gloses (lsf_translator/postprocess.py),
    # mesuré dans eval/RESULTS.md (eval/evaluate.py --postprocess).
    postprocess: bool = False
    # Exemples few-shot ciblés ajoutés au prompt (prompts.py: FEW_SHOT_TARGETED),
    # mesurés dans eval/RESULTS.md (eval/evaluate.py --few-shot-cible).
    few_shot_targeted: bool = False
    # Few-shot dynamique : nb d'exemples choisis phrase par phrase dans la banque
    # (lsf_translator/retrieval.py) ; 0 = désactivé (eval/evaluate.py --few-shot-dynamique).
    few_shot_dynamic: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """Paramètres de l'orchestrateur et des files d'attente."""

    transcript_queue_size: int = 64
    gloss_queue_size: int = 64
    # auto | fr | en. `auto` laisse le LLM détecter la langue source.
    language: str = "auto"


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_level: int = logging.INFO

    def with_overrides(self, **sections) -> "AppConfig":
        """Retourne une copie avec certaines sous-sections remplacées.

        Exemple: ``CONFIG.with_overrides(stt=replace(CONFIG.stt, device="cpu"))``
        """
        return replace(self, **sections)


# Instance par défaut, importable partout : `from config import CONFIG`.
CONFIG = AppConfig()

# Ré-export pratique pour la CLI.
__all__ = [
    "AudioConfig",
    "STTConfig",
    "OllamaConfig",
    "PipelineConfig",
    "AppConfig",
    "CONFIG",
    "replace",
]
