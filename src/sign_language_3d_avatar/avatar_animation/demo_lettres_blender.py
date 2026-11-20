"""Démo : l'avatar signe chaque lettre de l'alphabet manuel ; par lettre, une vidéo en pied
(`<LETTRE>.mp4`) et une en gros plan sur la main droite (`<LETTRE>_main.mp4`).

Chaque lettre est jouée comme dans un mot épelé (partie centrale du clip, `sequence._letter`),
entre deux poses de repos ; c'est ce que voit le lecteur d'un nom propre ou d'un terme
technique. Les vidéos servent à vérifier les lettres à l'œil
(`dataset_generation/verif_lettres.py`, qui lance ce script).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \\
        --python src/sign_language_3d_avatar/avatar_animation/demo_lettres_blender.py -- \\
        --sortie dossier/ [--lettres N H P] [--angle 15]
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_phrase_blender import render  # noqa: E402
from sequence import load_index  # noqa: E402


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="demo_lettres_blender.py")
    ap.add_argument("--sortie", required=True, help="dossier des vidéos <LETTRE>.mp4")
    ap.add_argument("--lettres", nargs="*", default=list(string.ascii_uppercase))
    ap.add_argument("--avatar", help="fichier .vrm (VRoid Studio) ; défaut : mannequin")
    ap.add_argument("--angle", type=float, default=15.0, help="angle de vue (0 = de face)")
    args = ap.parse_args(argv)

    index = load_index()
    out = Path(args.sortie).expanduser().resolve()
    for letter in (l.upper() for l in args.lettres):
        if f"FS:{letter}" not in index["signes"]:
            print(f"[lettres] {letter} : absente du dictionnaire")
            continue
        # Glose « épelée » d'une seule lettre : joue la lettre comme dans un mot épelé.
        index["gloses"][f"LETTRE-{letter}"] = {"type": "épelé", "signes": [f"FS:{letter}"]}
        render([f"LETTRE-{letter}"], out / f"{letter}.mp4", args.avatar, args.angle, index)
        render([f"LETTRE-{letter}"], out / f"{letter}_main.mp4", args.avatar, args.angle, index, gros_plan=True)


if __name__ == "__main__":
    main()
