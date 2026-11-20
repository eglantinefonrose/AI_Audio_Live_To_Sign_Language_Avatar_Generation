#!/bin/bash
# Lance l'outil Avatar LSFB (macOS) : double-cliquer. Laisser la fenêtre ouverte pendant l'utilisation.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../../.." || exit 1
if ! command -v uv >/dev/null; then
  echo "L'outil n'est pas installé : double-cliquez d'abord sur « Installer.command »."
  read -r -p "Appuyez sur Entrée pour fermer."; exit 1
fi
uv run --quiet --no-project --python 3.12 --with numpy python src/sign_language_3d_avatar/local_website/serveur.py --ouvrir
