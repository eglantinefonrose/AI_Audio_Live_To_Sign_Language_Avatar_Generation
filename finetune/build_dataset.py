"""Génère un dataset synthétique FR -> glose LSF pour fine-tuner (LoRA).

⚠️  100 % synthétique = DISTILLATION. Le modèle fine-tuné n'ira pas au-delà du
modèle "professeur" (`--teacher`) qui génère les données, et héritera de ses
erreurs. Choisis donc un professeur MEILLEUR que l'élève que tu veux entraîner
(mesure-les d'abord avec `eval/evaluate.py` et prends le meilleur F1). Sans
validation humaine, la qualité est plafonnée par le professeur + les filtres
ci-dessous — c'est un point de départ, pas une vérité terrain.

Pipeline :
  1. génère des phrases FR variées (par phénomène + domaine) avec le professeur ;
  2. traduit chaque phrase en gloses via le MÊME prompt de production ;
  3. filtre (contrôle qualité auto, remplace l'annotateur humain) ;
  4. dédup + anti-fuite contre eval/dataset.jsonl (test held-out) ;
  5. écrit train/val au format chat (SANS few-shot : la connaissance passe dans
     les poids) + un manifeste de provenance.

Usage :
  uv run python finetune/build_dataset.py --teacher qwen2.5:14b --per-category 25
  uv run python finetune/build_dataset.py --teacher gemma2:9b --per-category 3   # test rapide
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ollama import Client

from config import CONFIG
from lsf_translator.prompts import (
    GLOSS_JSON_SCHEMA,
    TRAIN_SYSTEM,
    build_messages,
    build_user_prompt,
)

OUT_DIR = PROJECT / "finetune" / "data"
EVAL_DATASET = PROJECT / "eval" / "dataset.jsonl"

# Gloses "mots-outils" interdites (mêmes que l'éval) : une sortie qui en contient
# est rejetée par le contrôle qualité.
FORBIDDEN = {
    "DE", "DANS", "POUR", "LE", "LA", "LES", "UN", "UNE", "DES", "À", "AU",
    "ET", "OU", "QUE", "AVEC", "EN", "DU", "CE", "CETTE", "SE", "NE", "ON", "VA",
    "QUE", "QUI", "SUR", "SES", "MON", "MA", "TON", "TA",
}

# Catégories de génération : (étiquette-phénomène, consigne pour le professeur).
# Couvre la grammaire LSF ET le domaine ML (tes vidéos).
CATEGORIES = [
    ("negation", "des phrases contenant une négation"),
    ("question-ouverte", "des questions ouvertes (qui, quoi, où, quand, pourquoi, comment)"),
    ("question-ouinon", "des questions fermées (réponse oui/non)"),
    ("temps", "des phrases situées dans le temps (hier, demain, la semaine prochaine, tout à l'heure)"),
    ("lieu", "des phrases mentionnant un lieu (au bureau, à Paris, à la maison)"),
    ("connecteur", "des phrases commençant par un connecteur de conclusion (pour finir, en conclusion, pour conclure)"),
    ("quotidien", "des phrases simples de la vie quotidienne"),
    ("domaine-ml", "des phrases issues d'un exposé technique sur le machine learning (modèles, datasets, entraînement, fine tuning, détection, flux de données)"),
]

# Schéma pour la génération de phrases sources.
SENT_SCHEMA = {
    "type": "object",
    "properties": {"sentences": {"type": "array", "items": {"type": "string"}}},
    "required": ["sentences"],
}


# -- utilitaires ---------------------------------------------------------
def normalize(text: str) -> str:
    """Clé de déduplication / anti-fuite : minuscule, espaces et ponctuation finale réduits."""
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(" .!?")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def prompt_hash() -> str:
    from lsf_translator.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
    return sha((SYSTEM_PROMPT + repr(FEW_SHOT_EXAMPLES)).encode("utf-8"))


# -- appels au professeur ------------------------------------------------
def gen_sentences(client: Client, teacher: str, desc: str, k: int) -> list[str]:
    """Demande au professeur k phrases FR variées illustrant `desc`."""
    msg = (
        f"Génère exactement {k} phrases en français, variées, naturelles et "
        f"courtes à moyennes, illustrant {desc}. Évite les répétitions. "
        f'Réponds en JSON : {{"sentences": ["...", "..."]}}.'
    )
    try:
        r = client.chat(
            model=teacher, messages=[{"role": "user", "content": msg}],
            format=SENT_SCHEMA, options={"temperature": 0.9},
        )
        data = json.loads(r["message"]["content"])
        return [str(s).strip() for s in data.get("sentences", []) if str(s).strip()]
    except Exception as exc:  # pragma: no cover
        print(f"    (génération échouée : {exc})", file=sys.stderr)
        return []


def to_gloss(client: Client, teacher: str, text: str) -> list[str]:
    """Traduit `text` en gloses avec le prompt de PRODUCTION (few-shot inclus)."""
    try:
        r = client.chat(
            model=teacher, messages=build_messages(text, "fr"),
            format=GLOSS_JSON_SCHEMA,
            options={"temperature": 0.0, "num_ctx": CONFIG.ollama.num_ctx},
        )
        data = json.loads(r["message"]["content"])
        glosses = data.get("glosses", []) if isinstance(data, dict) else []
        return [str(g).strip() for g in glosses if str(g).strip()]
    except Exception:
        return []


# -- contrôle qualité (remplace l'annotateur) ----------------------------
def quality_reason(text: str, glosses: list[str]) -> str | None:
    """Retourne None si la paire est acceptable, sinon la raison du rejet."""
    if not glosses:
        return "glose vide"
    if any(g in FORBIDDEN for g in glosses):
        return "mot-outil présent"
    if any(g != g.upper() for g in glosses):
        return "glose non MAJUSCULE"
    if any(not re.fullmatch(r"[A-ZÀ-ÖØ-Þ0-9'\-]+", g) for g in glosses):
        return "caractère inattendu"
    nw = len(text.split())
    if len(glosses) > max(3, round(nw * 1.8)):
        return "trop de gloses vs source"
    return None


# -- écriture ------------------------------------------------------------
def training_record(text: str, glosses: list[str]) -> dict:
    """Format chat pour LoRA — SANS few-shot (la connaissance va dans les poids)."""
    return {"messages": [
        {"role": "system", "content": TRAIN_SYSTEM},
        {"role": "user", "content": build_user_prompt(text, "fr")},
        {"role": "assistant",
         "content": json.dumps({"glosses": glosses}, ensure_ascii=False)},
    ]}


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


# -- pipeline ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Builder de dataset synthétique LSF.")
    ap.add_argument("--teacher", default="qwen2.5:14b",
                    help="Modèle générateur (DOIT être meilleur que l'élève visé).")
    ap.add_argument("--per-category", type=int, default=25,
                    help="Nb de phrases demandées par catégorie (avant filtrage).")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(host=CONFIG.ollama.host)
    rng = random.Random(args.seed)

    # Anti-fuite : textes du test held-out à exclure du train.
    eval_norm = set()
    if EVAL_DATASET.exists():
        for line in EVAL_DATASET.read_text(encoding="utf-8").splitlines():
            if line.strip():
                eval_norm.add(normalize(json.loads(line)["text"]))

    print(f"Professeur : {args.teacher}  —  {args.per_category}/catégorie  "
          f"({len(CATEGORIES)} catégories)\n")

    # 1) Génération des phrases sources.
    seen: set[str] = set()
    sources: list[tuple[str, str]] = []  # (texte, phénomène)
    for phen, desc in CATEGORIES:
        got = gen_sentences(client, args.teacher, desc, args.per_category)
        kept = 0
        for s in got:
            key = normalize(s)
            if not key or key in seen or key in eval_norm:
                continue
            seen.add(key)
            sources.append((s, phen))
            kept += 1
        print(f"  {phen:18s} : {kept} phrases retenues (sur {len(got)} générées)")

    # 2) Traduction + 3) contrôle qualité.
    accepted: list[dict] = []
    drops: dict[str, int] = {}
    pair_keys: set[str] = set()
    print("\nTraduction + contrôle qualité…")
    for text, phen in sources:
        glosses = to_gloss(client, args.teacher, text)
        reason = quality_reason(text, glosses)
        if reason:
            drops[reason] = drops.get(reason, 0) + 1
            continue
        pkey = normalize(text) + "|" + "|".join(glosses)
        if pkey in pair_keys:
            drops["doublon"] = drops.get("doublon", 0) + 1
            continue
        pair_keys.add(pkey)
        accepted.append({"text": text, "glosses": glosses, "phenomenon": phen})

    # 4) Split train/val.
    rng.shuffle(accepted)
    n_val = max(1, round(len(accepted) * args.val_frac)) if accepted else 0
    val, train = accepted[:n_val], accepted[n_val:]

    write_jsonl(out / "train.jsonl", [training_record(p["text"], p["glosses"]) for p in train])
    write_jsonl(out / "valid.jsonl", [training_record(p["text"], p["glosses"]) for p in val])
    # Copie "lisible" (texte + gloses + phénomène) pour inspection/correction future.
    write_jsonl(out / "pairs.jsonl", accepted)

    manifest = {
        "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "teacher": args.teacher,
        "prompt_hash": prompt_hash(),
        "synthetic": True, "human_validated": False,
        "generated": len(sources), "accepted": len(accepted),
        "train": len(train), "valid": len(val),
        "drops": drops,
        "per_category": args.per_category, "seed": args.seed,
        "eval_excluded": len(eval_norm),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n── Bilan ─────────────────────────────────")
    print(f"  Générées   : {len(sources)}")
    print(f"  Acceptées  : {len(accepted)}  (train {len(train)} / valid {len(val)})")
    print(f"  Rejetées   : {sum(drops.values())}  {drops or ''}")
    print(f"  Sorties    : {out.relative_to(PROJECT)}/ (train.jsonl, valid.jsonl, "
          f"pairs.jsonl, manifest.json)")
    if not accepted:
        print("  ⚠️  Rien d'accepté — vérifie qu'Ollama et le professeur répondent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
