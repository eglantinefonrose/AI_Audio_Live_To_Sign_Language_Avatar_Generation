"""Variante « gemma » d'un dataset rangé : le message system fusionné dans le 1er user.

Le chat template de Gemma 2 n'a pas de rôle `system` : il lève « System role not
supported » dès le premier exemple. On produit donc un nouveau dataset où le
system est placé en tête du premier message user (séparé par une ligne vide),
contenu par ailleurs identique. C'est la même transformation que fait Ollama à
l'inférence avec son template gemma2 (SYSTEM placé avant le message user).

Le résultat est un dataset à part entière, rangé sous son propre hash
(finetune/data/<train_sha>/, cf. snapshot_data.py). Son manifeste reprend celui
de la source + `variant` et `derived_from` ; `latest()` ignore les variantes, pour
que `./finetune/train.sh` sans DATA ne les choisisse pas pour un modèle qwen.

Usage :
    uv run python finetune/gemma_variant.py finetune/data/2184f8f3751e
    # -> affiche "<train_sha> <dossier rangé>"
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from snapshot_data import freeze

VARIANT = "gemma-no-system"
SEP = "\n\n"


def fold_system(record: dict) -> dict:
    messages = record["messages"]
    if messages[0]["role"] != "system":
        return record
    system, first, *rest = messages
    if first["role"] != "user":
        sys.exit("Message system non suivi d'un message user : format inattendu.")
    merged = {"role": "user", "content": system["content"] + SEP + first["content"]}
    return {**record, "messages": [merged, *rest]}


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).resolve()
    src_sha, src = freeze(src)   # vérifie que la source est intacte

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "variant"
        out.mkdir()
        for name in ("train.jsonl", "valid.jsonl"):
            lines = (src / name).read_text(encoding="utf-8").splitlines()
            rows = [fold_system(json.loads(line)) for line in lines if line.strip()]
            (out / name).write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        if (src / "pairs.jsonl").exists():
            shutil.copyfile(src / "pairs.jsonl", out / "pairs.jsonl")

        source_manifest = src / "manifest.json"
        manifest = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else {}
        manifest.update({
            "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "variant": VARIANT,
            "variant_note": "system fusionné en tête du 1er message user (Gemma 2 n'a pas de rôle system)",
            "derived_from": src_sha,
        })
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        sha, dest = freeze(out, move=True)
    print(sha, dest.relative_to(Path(__file__).resolve().parent.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
