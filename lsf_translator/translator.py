"""Traducteur français/anglais -> glossaire LSF via un LLM local (Ollama).

Utilise le client asynchrone d'Ollama avec sortie structurée (JSON Schema) pour
garantir un `list[str]` exploitable sans parsing fragile. Inclut une
vérification de disponibilité au démarrage (instance lancée + modèle présent).
"""
from __future__ import annotations

import json
import logging

from config import OllamaConfig
from contracts import GlossEvent, TranscriptEvent
from lsf_translator.prompts import (
    GLOSS_JSON_SCHEMA,
    build_messages,
)

try:  # import paresseux
    from ollama import AsyncClient, ResponseError
except Exception as exc:  # pragma: no cover - dépend de l'environnement
    AsyncClient = None
    ResponseError = Exception
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

log = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """Levée si le paquet `ollama` manque ou si le serveur est injoignable."""


class LSFTranslator:
    """Convertit une phrase en gloses LSF en interrogeant Ollama."""

    def __init__(self, config: OllamaConfig) -> None:
        self.config = config
        self._client = None

    # -- cycle de vie ----------------------------------------------------
    async def connect(self) -> None:
        """Instancie le client et vérifie que l'instance Ollama répond."""
        if AsyncClient is None:
            raise OllamaUnavailableError(
                f"Paquet 'ollama' indisponible : {_IMPORT_ERROR}. "
                "Installez-le (pip install ollama)."
            )
        self._client = AsyncClient(
            host=self.config.host, timeout=self.config.request_timeout
        )
        try:
            listing = await self._client.list()
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Impossible de joindre Ollama sur {self.config.host} : {exc}. "
                "Lancez-le avec `ollama serve`."
            ) from exc

        available = self._available_models(listing)
        if not self._model_available(available):
            # Échec immédiat et explicite : sinon chaque traduction renverrait
            # un 404 silencieux => [LSF Gloss] [] en boucle.
            raise OllamaUnavailableError(
                f"Le modèle '{self.config.model}' n'est pas installé sur Ollama.\n"
                f"  Modèles disponibles : {', '.join(available) or '(aucun)'}\n"
                f"  → `ollama pull {self.config.model}`  ou  "
                f"relancez avec `--model <un modèle de la liste>`."
            )
        log.info(
            "Traducteur LSF connecté à Ollama (%s, modèle=%s).",
            self.config.host,
            self.config.model,
        )

    @staticmethod
    def _available_models(listing) -> list[str]:
        """Noms des modèles installés (gère objets pydantic et dicts)."""
        models = getattr(listing, "models", None)
        if models is None and isinstance(listing, dict):
            models = listing.get("models", [])
        names = []
        for m in models or []:
            name = getattr(m, "model", None)
            if name is None and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                names.append(name)
        return names

    def _model_available(self, names: list[str]) -> bool:
        """Vrai si le modèle configuré correspond à un modèle installé."""
        target = self.config.model
        return any(n == target or n.split(":")[0] == target.split(":")[0]
                   for n in names)

    # -- traduction ------------------------------------------------------
    async def translate(self, text: str, language: str | None = None) -> list[str]:
        """Retourne la liste ordonnée des gloses LSF pour `text`.

        Ne lève pas en cas d'erreur LLM : journalise et renvoie `[]` pour ne pas
        interrompre le flux temps réel.
        """
        if self._client is None:
            raise OllamaUnavailableError("translate() appelé avant connect().")
        if not text or not text.strip():
            return []

        try:
            response = await self._client.chat(
                model=self.config.model,
                messages=build_messages(text, language),
                format=GLOSS_JSON_SCHEMA,
                options={
                    "temperature": self.config.temperature,
                    "num_ctx": self.config.num_ctx,
                },
                keep_alive=self.config.keep_alive,
            )
        except ResponseError as exc:
            log.error("Erreur Ollama lors de la traduction : %s", exc)
            return []
        except Exception as exc:  # timeout, réseau…
            log.error("Échec de l'appel à Ollama : %s", exc)
            return []

        content = self._extract_content(response)
        return self._parse_glosses(content)

    async def translate_event(self, sentence: TranscriptEvent) -> GlossEvent:
        """Traduit une phrase (interface neutre) en `GlossEvent`.

        C'est le point d'entrée utilisé par l'orchestrateur ; il conserve le
        texte source et l'horodatage de la phrase d'origine.
        """
        glosses = await self.translate(sentence.text, sentence.language)
        return GlossEvent(
            source_text=sentence.text,
            glosses=glosses,
            language=sentence.language,
            timestamp=sentence.timestamp,
        )

    @staticmethod
    def _extract_content(response) -> str:
        message = getattr(response, "message", None)
        if message is not None:
            return getattr(message, "content", "") or ""
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "") or ""
        return ""

    @staticmethod
    def _parse_glosses(content: str) -> list[str]:
        if not content:
            return []
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            log.warning("Réponse LLM non-JSON, ignorée : %r", content[:200])
            return []
        glosses = data.get("glosses", []) if isinstance(data, dict) else []
        # Normalisation défensive : chaînes non vides uniquement.
        return [str(g).strip() for g in glosses if str(g).strip()]


__all__ = ["LSFTranslator", "OllamaUnavailableError"]
