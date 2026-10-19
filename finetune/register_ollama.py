"""Importe un modèle fine-tuné dans Ollama ET enregistre sa provenance.

Ollama ne stocke aucune métadonnée libre (pas de date d'entraînement, pas de
commentaire conservé du Modelfile). On garde donc le lien dans un petit fichier
versionné : `finetune/models/<tag>.json` = tag Ollama + digest + provenance de
l'entraînement (copie de `<adapters>/training_meta.json`, écrit par train.sh).

`eval/evaluate.py` lit ce fichier (et vérifie le digest) pour inscrire la date et
la durée d'entraînement dans chaque record d'éval ; `publish_hf.py` les reprend
dans la model card.

Usage :
    uv run python finetune/register_ollama.py lsf-qwen3b:v3
        # = ollama create lsf-qwen3b:v3 -f finetune/Modelfile + enregistrement
    uv run python finetune/register_ollama.py lsf-qwen7b:v1 \\
        --modelfile finetune/Modelfile.7b --meta finetune/adapters-7b/training_meta.json \\
        --quantize q4_K_M
        # import quantifié : ~4,7 Go dans Ollama au lieu de ~15 Go en fp16
    uv run python finetune/register_ollama.py lsf-qwen3b:v2 --no-create \\
        --meta chemin/vers/training_meta.json
        # enregistre un tag DÉJÀ créé (rattrapage)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ollama import Client

from config import CONFIG

MODELS_DIR = PROJECT / "finetune" / "models"
ADAPTERS_DIR = PROJECT / "finetune" / "adapters"
MODELFILE = PROJECT / "finetune" / "Modelfile"


def registry_path(tag: str) -> Path:
    return MODELS_DIR / (tag.replace(":", "-").replace("/", "-") + ".json")


def ollama_digests(client: Client) -> dict[str, str]:
    listing = client.list()
    models = listing.get("models") if isinstance(listing, dict) \
        else getattr(listing, "models", [])
    out = {}
    for m in models or []:
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        dg = m.get("digest") if isinstance(m, dict) else getattr(m, "digest", None)
        if name:
            out[name] = (dg or "")[:19]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ollama create + provenance d'entraînement.")
    ap.add_argument("tag", help="Tag Ollama versionné, ex. lsf-qwen3b:v3")
    ap.add_argument("--modelfile", default=str(MODELFILE))
    ap.add_argument("--meta", default=str(ADAPTERS_DIR / "training_meta.json"),
                    help="training_meta.json écrit par finetune/train.sh")
    ap.add_argument("--quantize", default=None, metavar="NIVEAU",
                    help="Quantifie à l'import (ollama create -q), ex. q4_K_M. "
                         "Défaut : fp16 tel que produit par mlx_lm.fuse --de-quantize.")
    ap.add_argument("--no-create", action="store_true",
                    help="Ne lance pas `ollama create` (tag déjà importé).")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        sys.exit(f"Provenance introuvable : {meta_path}\n"
                 "Entraîne via ./finetune/train.sh (qui l'écrit), ou passe --meta.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    client = Client(host=CONFIG.ollama.host)
    existing = ollama_digests(client)
    if not args.no_create:
        if args.tag in existing:
            sys.exit(f"Le tag {args.tag} existe déjà : ne réécris jamais un tag, "
                     "prends une nouvelle version (ou --no-create pour l'enregistrer).")
        cmd = ["ollama", "create", args.tag, "-f", args.modelfile]
        if args.quantize:
            cmd += ["--quantize", args.quantize]
        subprocess.run(cmd, check=True)
        existing = ollama_digests(client)
    if args.tag not in existing:
        sys.exit(f"Tag {args.tag} absent d'Ollama.")

    record = {
        "tag": args.tag,
        "ollama_digest": existing[args.tag],
        # None = poids fp16 (ou inconnu pour un tag enregistré avec --no-create).
        "quantization": args.quantize,
        "registered_at": datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "training": meta,
    }
    out = registry_path(args.tag)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    approx = " (approximatif)" if meta.get("approximate") else ""
    print(f"✅ {args.tag} ({record['ollama_digest']}) — entraîné le "
          f"{meta.get('started_at') or meta.get('ended_at') or '?'}, "
          f"{meta.get('duration_s') or '?'} s{approx}")
    print(f"   Provenance : {out.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
