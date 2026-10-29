"""Test interactif : taper une phrase en gloses LSF, voir l'avatar la signer.

Chaque glose est cherchée dans les lexiques SiGML (tous les `.sigml` de
`dataset_generation/sigml/`) ; la phrase est animée par
Blender sans interface (`avatar_animation/sigml_blender.py`), puis la vidéo
s'ouvre dans le lecteur par défaut. Les signes sont des BROUILLONS non validés.

    uv run python src/sign_language_3d_avatar/tester_phrase_sigml.py
    uv run python src/sign_language_3d_avatar/tester_phrase_sigml.py --phrase "MOI MERCI"

Blender : /Applications/Blender.app par défaut, sinon --blender ou la variable BLENDER.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).resolve().parent
LEXICON = BASE / "dataset_generation" / "sigml"
ANIMATOR = BASE / "avatar_animation" / "sigml_blender.py"
OUT = BASE / "avatar_animation" / "data" / "sigml" / "phrases"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def load_glosses(path: Path) -> list[str]:
    """Gloses d'un fichier .sigml, ou de tous les .sigml d'un dossier."""
    files = sorted(path.glob("*.sigml")) if path.is_dir() else [path]
    glosses = [s.get("gloss") for f in files for s in ET.parse(f).getroot().iter("hns_sign")]
    return list(dict.fromkeys(glosses))


def parse_phrase(text: str, known: list[str]) -> tuple[list[str], list[str]]:
    """(gloses reconnues dans l'ordre, gloses inconnues). Insensible à la casse."""
    by_upper = {g.upper(): g for g in known}
    words = [by_upper.get(w.upper(), w.upper()) for w in text.split()]
    return [w for w in words if w in known], [w for w in words if w not in known]


def render(words: list[str], blender: str, lexicon: Path) -> Path:
    slug = re.sub(r"[^\w-]+", "_", "_".join(words))[:80]
    out = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.mp4"
    cmd = [blender, "-b", "--factory-startup", "--python", str(ANIMATOR), "--",
           str(lexicon), "--phrase", " ".join(words), "--sortie", str(out)]
    print(f"  rendu de {len(words)} signe(s) avec Blender…", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        errors = [l for l in (proc.stdout + proc.stderr).splitlines()
                  if re.search(r"Error|Traceback|\[sigml\]", l)]
        raise RuntimeError("échec du rendu Blender :\n    " + "\n    ".join(errors[-15:] or ["(pas de message)"]))
    print(f"  vidéo prête en {time.time() - t0:.0f} s : {out.relative_to(BASE.parent.parent)}")
    return out


def open_video(path: Path) -> None:
    opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
    subprocess.run([*opener, str(path)], check=False)


def handle(text: str, known: list[str], args) -> None:
    words, unknown = parse_phrase(text, known)
    if unknown:
        print(f"  ⚠️  absentes du lexique, ignorées : {' '.join(unknown)}")
    if not words:
        print(f"  aucune glose connue. Lexique : {' '.join(known)}")
        return
    print(f"  phrase signée : {' '.join(words)}")
    try:
        video = render(words, args.blender, args.lexique)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        return
    if not args.ne_pas_ouvrir:
        open_video(video)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phrase", help="phrase à signer directement (sinon : mode interactif)")
    ap.add_argument("--lexique", type=Path, default=LEXICON, help="fichier .sigml ou dossier (défaut : dataset_generation/sigml/)")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER), help="exécutable Blender")
    ap.add_argument("--ne-pas-ouvrir", action="store_true", help="générer la vidéo sans l'ouvrir")
    args = ap.parse_args()

    if not Path(args.blender).exists():
        sys.exit(f"Blender introuvable : {args.blender} (option --blender ou variable BLENDER)")
    known = load_glosses(args.lexique)
    if args.phrase:
        handle(args.phrase, known, args)
        return

    print("Avatar LSF (SiGML) — tapez une phrase en gloses, par ex. « TOI DONNER-À-MOI MERCI ».")
    print(f"Gloses disponibles : {' '.join(known)}")
    print("Brouillons non validés par un locuteur LSF. Entrée vide ou « q » pour quitter.\n")
    while True:
        try:
            text = input("gloses> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("", "q", "quit", "exit"):
            break
        handle(text, known, args)


if __name__ == "__main__":
    main()
