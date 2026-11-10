"""Test : l'avatar signe une phrase en LSFB, rendue en vidéo par Blender.

Entrée : des gloses (`--gloses "MOI MAISON ALLER"`) ou une phrase en français
(`--texte "Je vais à la maison."`, traduite en gloses par le traducteur du projet, ce
qui demande Ollama). Sans option : mode interactif (gloses, ou `t: phrase` pour traduire).

Chaque glose est cherchée dans le dictionnaire LSFB (`avatar_animation/data/dictionnaire/`) :
signe LSFB nettoyé, pointage généré (MOI, TOI…) ou, à défaut, mot affiché à l'écran.
Avatar : mannequin, ou modèle VRoid Studio avec `--avatar mon_avatar.vrm`.

    uv run python src/sign_language_3d_avatar/tester_phrase_lsfb.py --gloses "MOI MAISON AIDER-À-TOI MERCI"
    uv run python src/sign_language_3d_avatar/tester_phrase_lsfb.py --texte "Demain, je vais à Paris."

Blender : /Applications/Blender.app par défaut, sinon --blender ou la variable BLENDER.
Données dérivées de LSFB-ISOL (CC BY-NC-SA 4.0) : usage non commercial.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE.parent
DEMO = BASE / "avatar_animation" / "demo_phrase_blender.py"
OUT = BASE / "avatar_animation" / "data" / "dictionnaire" / "phrases"
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def translate(text: str) -> list[str]:
    """Phrase française -> gloses, avec le traducteur du projet (Ollama)."""
    sys.path.insert(0, str(SRC.parent))          # le projet importe ses modules en « src.… »
    from src.config import CONFIG
    from src.lsf_translator.translator import LSFTranslator

    async def run() -> list[str]:
        tr = LSFTranslator(CONFIG.ollama)
        await tr.connect()
        return await tr.translate(text, "fr")

    return asyncio.run(run())


def render(glosses: list[str], args) -> Path:
    slug = re.sub(r"[^\w-]+", "_", "_".join(glosses))[:80]
    out = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}_{slug}.mp4"
    cmd = [args.blender, "-b", "--factory-startup", "--python", str(DEMO), "--",
           "--gloses", " ".join(glosses), "--sortie", str(out)]
    if args.avatar:
        cmd += ["--avatar", str(Path(args.avatar).expanduser().resolve())]
    print(f"  rendu avec Blender…", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("[démo]") and "vidéo" not in line:
            print("  " + line[7:])
    if proc.returncode != 0 or not out.exists():
        errors = [l for l in (proc.stdout + proc.stderr).splitlines() if re.search(r"Error|Traceback", l)]
        raise RuntimeError("échec du rendu Blender :\n    " + "\n    ".join(errors[-15:] or ["(pas de message)"]))
    print(f"  vidéo prête en {time.time() - t0:.0f} s : {out.relative_to(SRC.parent)}")
    return out


def handle(text: str, args, translated: bool = False) -> None:
    if translated:
        try:
            glosses = translate(text)
        except Exception as e:
            print(f"  ❌ traduction impossible : {e}")
            return
        print(f"  gloses : {' '.join(glosses) or '(aucune)'}")
    else:
        glosses = text.upper().split()
    if not glosses:
        return
    try:
        video = render(glosses, args)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        return
    if not args.ne_pas_ouvrir:
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.run([*opener, str(video)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gloses", help="gloses séparées par des espaces")
    ap.add_argument("--texte", help="phrase en français, traduite en gloses (Ollama)")
    ap.add_argument("--avatar", help="fichier .vrm exporté de VRoid Studio (défaut : mannequin)")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER), help="exécutable Blender")
    ap.add_argument("--ne-pas-ouvrir", action="store_true", help="générer la vidéo sans l'ouvrir")
    args = ap.parse_args()
    if not Path(args.blender).exists():
        sys.exit(f"Blender introuvable : {args.blender} (option --blender ou variable BLENDER)")
    if args.gloses or args.texte:
        handle(args.texte or args.gloses, args, translated=bool(args.texte))
        return
    print("Avatar LSFB — tapez des gloses (« MOI MAISON ALLER ») ou « t: » + une phrase en français.")
    print("Entrée vide ou « q » pour quitter.\n")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in ("", "q", "quit", "exit"):
            break
        if line.lower().startswith("t:"):
            handle(line[2:].strip(), args, translated=True)
        else:
            handle(line, args)


if __name__ == "__main__":
    main()
