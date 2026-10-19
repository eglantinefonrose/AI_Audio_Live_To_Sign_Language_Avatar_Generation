"""Rangement des datasets d'entraînement par hash : finetune/data/<train_sha>/.

Même logique que eval/datasets/ : un dossier par contenu, en lecture seule, jamais
réécrit. Le dossier porte le hash de train.jsonl (`train_sha`, 12 caractères du
SHA-256, celui affiché dans eval/RESULTS.md) et contient train.jsonl,
valid.jsonl, manifest.json et pairs.jsonl du même jeu.

- build_dataset.py écrit chaque nouveau dataset dans un dossier temporaire puis le
  range ici (`freeze`).
- train.sh appelle ce script sur le dataset choisi (par défaut le plus récent) :
  un dossier déjà rangé est simplement vérifié ; un dossier extérieur est copié.

Usage :
    uv run python finetune/snapshot_data.py              # range/vérifie le plus récent
    uv run python finetune/snapshot_data.py <dossier>    # range/vérifie ce dossier
    uv run python finetune/snapshot_data.py --latest     # chemin du plus récent
    # -> affiche "<train_sha> <dossier rangé>"
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "finetune" / "data"
FILES = ("train.jsonl", "valid.jsonl", "manifest.json", "pairs.jsonl")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def datasets() -> list[Path]:
    """Datasets rangés (dossiers <hash>/ contenant un train.jsonl)."""
    if not DATA_DIR.exists():
        return []
    return [d for d in DATA_DIR.iterdir() if d.is_dir() and (d / "train.jsonl").exists()
            and not d.name.startswith(".")]


def latest() -> Path:
    """Dataset le plus récent, d'après `created` du manifeste (sinon date du fichier)."""
    def created(d: Path) -> str:
        m = d / "manifest.json"
        if m.exists():
            c = json.loads(m.read_text(encoding="utf-8")).get("created")
            if c:
                return c
        return "0000"   # sans manifeste : passe après les datasets datés
    found = datasets()
    if not found:
        sys.exit(f"Aucun dataset dans {_rel(DATA_DIR)}/ : lance finetune/build_dataset.py.")
    return max(found, key=lambda d: (created(d), (d / "train.jsonl").stat().st_mtime))


def freeze(src: Path, move: bool = False) -> tuple[str, Path]:
    """Range `src` sous finetune/data/<train_sha>/ et renvoie (train_sha, dossier).

    Si ce contenu est déjà rangé, vérifie seulement qu'il est intact (et identique à
    `src`) ; `move=True` déplace `src` au lieu de le copier (dossier temporaire).
    """
    src = src.resolve()
    train = src / "train.jsonl"
    if not train.exists():
        sys.exit(f"train.jsonl introuvable dans {_rel(src)}")
    sha = _sha(train.read_bytes())
    dest = DATA_DIR / sha
    present = [f for f in FILES if (src / f).exists()]
    if dest.exists():
        if _sha((dest / "train.jsonl").read_bytes()) != sha:
            sys.exit(f"Dataset altéré : {_rel(dest)}/train.jsonl ne correspond plus au hash {sha}.")
        if src != dest:
            for f in present:
                if (dest / f).exists() and (dest / f).read_bytes() != (src / f).read_bytes():
                    sys.exit(f"Même train.jsonl ({sha}) mais {f} différent de {_rel(dest)}/{f} : "
                             f"régénère le dataset plutôt que de modifier {f} seul.")
            if move:
                shutil.rmtree(src)   # doublon exact d'un dataset déjà rangé
        return sha, dest
    if move:
        src.rename(dest)
    else:
        dest.mkdir(parents=True)
        for f in present:
            shutil.copyfile(src / f, dest / f)
    for f in dest.iterdir():
        f.chmod(0o444)   # lecture seule : un dataset rangé ne se modifie pas
    return sha, dest


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--latest"]:
        print(_rel(latest()))
    else:
        sha, dest = freeze(Path(args[0]) if args else latest())
        print(sha, _rel(dest))
