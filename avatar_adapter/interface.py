"""Interface de rendu avatar — point d'extension pour le futur avatar 3D.

Le pipeline publie des `GlossEvent` sur une file d'attente asynchrone. Tout
consommateur implémentant `AvatarRendererInterface` peut s'y brancher sans que
l'orchestrateur, le STT ou le traducteur n'aient à être modifiés : c'est le
découplage exigé par l'architecture.

Un moteur d'avatar 3D (Unity/Unreal/Blender via socket, WebSocket, gRPC…)
n'aura qu'à sous-classer cette interface et à être enregistré auprès du
`Pipeline`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pipeline.events import GlossEvent

log = logging.getLogger(__name__)


class AvatarRendererInterface(ABC):
    """Classe de base abstraite pour un consommateur de gloses LSF.

    Cycle de vie : ``start()`` une fois au démarrage, ``render(event)`` pour
    chaque phrase traduite, ``stop()`` à l'arrêt. Les trois sont asynchrones
    pour pouvoir dialoguer avec un moteur distant sans bloquer le pipeline.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialise les ressources (connexion socket, fenêtre, etc.)."""

    @abstractmethod
    async def render(self, event: GlossEvent) -> None:
        """Rend / transmet une séquence de gloses LSF à l'avatar."""

    @abstractmethod
    async def stop(self) -> None:
        """Libère les ressources."""


class ConsoleAvatarRenderer(AvatarRendererInterface):
    """Implémentation de référence : journalise les gloses dans la console.

    Sert à la fois de démonstration de l'interface et de « renderer » par
    défaut tant que le véritable moteur 3D n'est pas branché.
    """

    def __init__(self, name: str = "console") -> None:
        self.name = name

    async def start(self) -> None:
        log.info("Avatar renderer '%s' démarré.", self.name)

    async def render(self, event: GlossEvent) -> None:
        log.debug("[avatar:%s] %s", self.name, event.glosses)

    async def stop(self) -> None:
        log.info("Avatar renderer '%s' arrêté.", self.name)


__all__ = ["AvatarRendererInterface", "ConsoleAvatarRenderer"]
