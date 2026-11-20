#!/bin/bash
# Fabrique l'archive à distribuer de l'outil Avatar LSFB : dist/avatar-lsfb.zip (dist/ est
# ignoré par git : l'archive contient le dictionnaire dérivé de LSFB et l'avatar).
#
# L'archive reprend l'arborescence du projet, réduite à ce dont l'outil a besoin, pour que
# les lanceurs de local_website/ marchent tels quels ; à sa racine : les lanceurs de
# racine_archive/ (ils appellent ceux de local_website/), la notice et LICENCE.txt.
#
#     bash src/sign_language_3d_avatar/local_website/fabriquer_archive.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
A=src/sign_language_3d_avatar
AVATAR=avatar_sign_4.vrm                       # comme AVATAR dans serveur.py
DIST="$ROOT/dist"
WORK="$(mktemp -d)"
OUT="$WORK/avatar-lsfb"
trap 'rm -rf "$WORK"' EXIT
cd "$ROOT"

[ -f "$A/avatar_animation/data/dictionnaire/index.json" ] || {
  echo "Dictionnaire absent : construisez-le d'abord (voir avatar_animation/README.md)."; exit 1; }

mkdir -p "$OUT/$A/local_website" "$OUT/$A/avatar_animation/data/dictionnaire" "$OUT/src/lsf_translator"
cp "$A/local_website/serveur.py" "$A/local_website/index.html" "$A/local_website/NOTICE.md" \
   "$A/local_website/Installer.command" "$A/local_website/Installer.bat" \
   "$A/local_website/Avatar LSFB.command" "$A/local_website/Avatar LSFB.bat" "$A/local_website/config.json" \
   "$OUT/$A/local_website/"
cp "$A/avatar_animation/sequence.py" "$A/avatar_animation/pose3d.py" "$A/avatar_animation/avatar_rig.py" \
   "$A/avatar_animation/demo_phrase_blender.py" "$OUT/$A/avatar_animation/"
cp "$A/avatar_animation/data/$AVATAR" "$OUT/$A/avatar_animation/data/"
cp "$A/avatar_animation/data/dictionnaire/index.json" "$OUT/$A/avatar_animation/data/dictionnaire/"
cp -R "$A/avatar_animation/data/dictionnaire/signes" "$OUT/$A/avatar_animation/data/dictionnaire/"
cp src/lsf_translator/synonymes_lsfb.csv "$OUT/src/lsf_translator/"
cp "$A/local_website/racine_archive/"* "$OUT/"
# Notice à la racine, sans l'en-tête destiné aux développeurs (commentaire HTML).
perl -0pe 's/\A<!--.*?-->\s*//s' "$A/local_website/NOTICE.md" > "$OUT/NOTICE.md"

mkdir -p "$DIST"
rm -f "$DIST/avatar-lsfb.zip"
(cd "$WORK" && ditto -c -k --keepParent avatar-lsfb "$DIST/avatar-lsfb.zip")
grep -q '"url_avis": *"http' "$A/local_website/config.json" \
  || echo "⚠️  config.json sans url_avis : les avis ne seront pas envoyés au tableau (voir GOOGLE_SHEETS.md)."
echo "Archive : $DIST/avatar-lsfb.zip ($(du -h "$DIST/avatar-lsfb.zip" | cut -f1)) ;" \
     "$(ls "$OUT/$A/avatar_animation/data/dictionnaire/signes" | wc -l | tr -d ' ') signes, avatar $AVATAR"
