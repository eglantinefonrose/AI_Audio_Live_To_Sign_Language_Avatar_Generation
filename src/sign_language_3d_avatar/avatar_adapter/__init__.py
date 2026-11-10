"""Point d'extension pour le rendu par avatar 3D (consommateur de gloses)."""
from src.sign_language_3d_avatar.avatar_adapter.interface import (
    AvatarRendererInterface,
    ConsoleAvatarRenderer,
)

__all__ = ["AvatarRendererInterface", "ConsoleAvatarRenderer"]
