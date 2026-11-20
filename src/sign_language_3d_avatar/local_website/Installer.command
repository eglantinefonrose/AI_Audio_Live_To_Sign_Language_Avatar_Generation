#!/bin/bash
# Installe ce dont l'outil Avatar LSFB a besoin (macOS) : double-cliquer, une seule fois.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../../.." || exit 1
echo "Installation de l'outil Avatar LSFB…"
if ! command -v uv >/dev/null; then
  echo "- Installation de uv (gestionnaire Python)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Échec : vérifiez la connexion Internet."; read -r -p "Entrée pour fermer."; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "- Préparation de Python et numpy…"
uv run --quiet --no-project --python 3.12 --with numpy python -c "import numpy" || { echo "Échec de la préparation de Python."; read -r -p "Entrée pour fermer."; exit 1; }
if [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then
  echo "- Blender : trouvé."
else
  echo "- ATTENTION : Blender est introuvable dans Applications. Installez-le depuis blender.org (voir la notice)."
fi
echo
echo "Installation terminée. Vous pouvez fermer cette fenêtre, puis double-cliquer sur « Avatar LSFB.command »."
read -r -p "Appuyez sur Entrée pour fermer."
